"""
Three-arm fusion pipeline for the LTWR business/SEC-filing study:

  Arm A: RRF   -- unweighted reciprocal-rank fusion (baseline)
  Arm B: Static TWR -- hand-set alpha/beta/gamma/delta (analogue of Eq. 2)
  Arm C: LTWR (Learned Trust-Weighted Ranking) -- learned fusion weights
         over the SAME closed, scalar feature set as static TWR (rrf score,
         w1, w2, w3). No document text, no new embeddings, no cross-encoder
         pass -- this is a learned combination FUNCTION, not a reranker,
         and is the architectural distinction the paper must state
         explicitly to preserve TWR's zero-overhead claim.

Both the retrieval stage (BM25 + FAISS) and the candidate pool are IDENTICAL
across all three arms, exactly mirroring the original TWR paper's design so
that any measured difference is attributable to the fusion step alone.
"""
import re
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from infra.business_document import BusinessDocument
from business_domain.gains import recency_weight

DENSE_MODEL_NAME = "all-MiniLM-L6-v2"

FILING_W1 = {"10-K": 1.0, "10-Q": 0.75, "8-K": 0.5, "DEF14A": 0.35}
AUDIT_W2 = {"Big4": 1.0, "OtherAudited": 0.7, "Unaudited": 0.3, "Unknown": 0.3}


class BusinessTWRPipeline:
    def __init__(self, corpus: List[BusinessDocument], k_doc: int = 5, ltwr_model=None):
        self.corpus = corpus
        self.k_doc = k_doc
        self.k_rrf = 60
        self.current_year = 2026
        self.ltwr_model = ltwr_model  # trained LGBMRanker/booster, or None

        self.embedder = SentenceTransformer(DENSE_MODEL_NAME)
        self._build_bm25_index()
        self._build_dense_index()

    def _build_bm25_index(self):
        tokenized_corpus = [re.findall(r"\b\w+\b", doc.text.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_dense_index(self):
        texts = [doc.text for doc in self.corpus]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def _embed_query(self, query: str):
        emb = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return emb

    def hybrid_retrieval(self, query: str, top_n: int = 10):
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranking = np.argsort(bm25_scores)[::-1][:top_n]

        query_emb = self._embed_query(query)
        faiss.normalize_L2(query_emb)
        _, faiss_indices = self.faiss_index.search(query_emb, top_n)
        faiss_ranking = faiss_indices[0]

        return bm25_ranking, bm25_scores, faiss_ranking

    def calculate_metadata_components(self, doc: BusinessDocument):
        w1 = FILING_W1.get(doc.filing_type, 0.2)
        w2 = AUDIT_W2.get(doc.audit_tier, 0.3)
        w3 = recency_weight(doc.filing_year, self.current_year)
        return w1, w2, w3

    def _accumulate_rrf(self, bm25_ranking, faiss_ranking, alpha=1.0):
        rrf_scores = {}
        for rank_list in [bm25_ranking, faiss_ranking]:
            for rank, doc_idx in enumerate(rank_list):
                contrib = 1.0 / (self.k_rrf + rank + 1)
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + alpha * contrib
        return rrf_scores

    # ---- Arm A: RRF -------------------------------------------------
    def rrf_only(self, bm25_ranking, faiss_ranking) -> List[int]:
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking)
        return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # ---- Arm B: static TWR (Eq. 2 analogue) -------------------------
    def static_twr(self, bm25_ranking, faiss_ranking,
                   alpha=1.0, beta=0.5, gamma=0.5, delta=0.3) -> List[int]:
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking, alpha=alpha)
        twr_scores = {}
        for doc_idx, rrf in rrf_scores.items():
            doc = self.corpus[doc_idx]
            w1, w2, w3 = self.calculate_metadata_components(doc)
            twr_scores[doc_idx] = rrf + beta * w1 + gamma * w2 + delta * w3
        return sorted(twr_scores.keys(), key=lambda x: twr_scores[x], reverse=True)

    # ---- Feature vector shared by static TWR and LTWR ------------
    def _feature_vector(self, doc_idx, rrf_score, bm25_score, faiss_score_lookup):
        doc = self.corpus[doc_idx]
        w1, w2, w3 = self.calculate_metadata_components(doc)
        dense_score = faiss_score_lookup.get(doc_idx, 0.0)
        return [rrf_score, bm25_score, dense_score, w1, w2, w3]

    def build_features_from_rankings(self, bm25_ranking, bm25_scores, faiss_ranking):
        """Constructs scalar features directly from pre-computed rankings to bypass re-retrieval."""
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking)
        faiss_score_lookup = {idx: 1.0 / (r + 1) for r, idx in enumerate(faiss_ranking)}
        feats = {}
        for doc_idx, rrf in rrf_scores.items():
            feats[doc_idx] = self._feature_vector(doc_idx, rrf, bm25_scores[doc_idx], faiss_score_lookup)
        return feats

    def build_features(self, query: str, top_n: int = 10):
        """Exposed for legacy training scripts."""
        bm25_ranking, bm25_scores, faiss_ranking = self.hybrid_retrieval(query, top_n=top_n)
        return self.build_features_from_rankings(bm25_ranking, bm25_scores, faiss_ranking)

    # ---- Arm C: LTWR (learned scalar-feature fusion) -------------
    def ltwr(self, bm25_ranking, bm25_scores, faiss_ranking) -> List[int]:
        """Runs fast, zero-overhead scalar ranking utilizing pre-computed ranking states."""
        if self.ltwr_model is None:
            raise RuntimeError("LTWR model not loaded -- train it first (business_domain/train_ltwr.py)")
        
        feats = self.build_features_from_rankings(bm25_ranking, bm25_scores, faiss_ranking)
        doc_idxs = list(feats.keys())
        if not doc_idxs:
            return []
            
        X = np.array([feats[i] for i in doc_idxs])
        scores = self.ltwr_model.predict(X)
        order = np.argsort(scores)[::-1]
        return [doc_idxs[i] for i in order]

    def provenance(self, indices: List[int]) -> List[Dict[str, Any]]:
        out = []
        for rank, idx in enumerate(indices[: self.k_doc]):
            doc = self.corpus[idx]
            out.append({
                "rank": rank + 1,
                "chunk_id": doc.chunk_id,
                "filing_type": doc.filing_type,
                "audit_tier": doc.audit_tier,
                "filing_year": doc.filing_year,
                "ticker": doc.ticker,
            })
        return out

    def retrieve(self, query: str, arm: str = "static_twr") -> Dict[str, Any]:
        t0 = time.perf_counter()
        bm25_ranking, bm25_scores, faiss_ranking = self.hybrid_retrieval(query)
        
        if arm == "rrf":
            indices = self.rrf_only(bm25_ranking, faiss_ranking)
        elif arm == "static_twr":
            indices = self.static_twr(bm25_ranking, faiss_ranking)
        elif arm == "ltwr":
            indices = self.ltwr(bm25_ranking, bm25_scores, faiss_ranking)
        else:
            raise ValueError(f"unknown arm: {arm}")
            
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"results": self.provenance(indices), "latency_ms": latency_ms}