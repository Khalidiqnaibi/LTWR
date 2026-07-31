"""
retrieval.py .. CVE/supply-chain-security analogue of
pipeline/academic_retrieval.py and pipeline/business_retrieval.py. Same
three-arm design:

  Arm A: RRF   .. unweighted reciprocal-rank fusion (baseline)
  Arm B: Static TWR .. hand-set alpha/beta/gamma/delta (Eq. 2, as specified:
         w1=severity, w2=cvss_version/scoring-methodology-currency,
         w3=recency decay)
  Arm C: LTWR (Learned Trust-Weighted Ranking) .. learned fusion weights
         over the SAME closed, scalar feature set as static TWR (rrf score,
         bm25 score, dense score, w1, w2, w3). No document text is a
         feature, no new embeddings, no cross-encoder pass.

Both the retrieval stage (BM25 + dense) and the candidate pool are IDENTICAL
across all three arms, so any measured difference is attributable to the
fusion step alone .. matching Section 3.1 of the TWR paper exactly.
"""
import re
import time
import numpy as np
import faiss ,json
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from infra.cve_document import CveDocument
from domain.gains import severity_weight, w2_weight, recency_weight

DENSE_MODEL_NAME = "all-MiniLM-L6-v2"


class CveTWRPipeline:
    def __init__(self, corpus: List[CveDocument], k_doc: int = 5, ltwr_model_path=None):
        self.corpus = corpus
        self.k_doc = k_doc
        self.k_rrf = 60
        self.current_year = 2026
        self.ltwr_model_path = ltwr_model_path  # path to the LTWR model file
        self.load_ltwr_model(self.ltwr_model_path) if self.ltwr_model_path else None

        self.rrf_norm = 2.0 / (self.k_rrf + 1)

        self.embedder = SentenceTransformer(DENSE_MODEL_NAME)
        self._build_bm25_index()
        self._build_dense_index()

    def load_ltwr_model(self, model_path) -> bool:
        """Loads LTWR linear weights strictly from a JSON file.

        Expected JSON format:
        {
            "coef": [c0, c1, c2, c3, c4, c5],
            "intercept": 0.123
        }
        """
        if not model_path.exists():
            raise FileNotFoundError(f"LTWR JSON model file not found: {model_path}")

        if model_path.suffix != ".json":
            raise ValueError(f"Only JSON model files are accepted. Got: {model_path.suffix}")

        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        if "coef" not in model_data or "intercept" not in model_data:
            raise KeyError("JSON model file must contain 'coef' and 'intercept' keys.")

        self.ltwr_coef = [float(c) for c in model_data["coef"]]
        self.ltwr_intercept = float(model_data["intercept"])
        return True

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
        return self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def hybrid_retrieval(self, query: str, top_n: int = 10):
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranking = np.argsort(bm25_scores)[::-1][:top_n]

        query_emb = self._embed_query(query)
        faiss.normalize_L2(query_emb)
        _, faiss_indices = self.faiss_index.search(query_emb, top_n)
        faiss_ranking = faiss_indices[0]

        return bm25_ranking, bm25_scores, faiss_ranking

    def calculate_metadata_components(self, doc: CveDocument):
        w1 = severity_weight(doc.severity)
        w2 = w2_weight(doc)
        w3 = recency_weight(doc.pub_year, self.current_year)
        return w1, w2, w3

    def _accumulate_rrf(self, bm25_ranking, faiss_ranking, alpha=1.0):
        rrf_scores = {}
        for rank_list in [bm25_ranking, faiss_ranking]:
            for rank, doc_idx in enumerate(rank_list):
                contrib = 1.0 / (self.k_rrf + rank + 1)
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + alpha * contrib
        return rrf_scores

    #  Arm A: RRF 
    def rrf_only(self, bm25_ranking, faiss_ranking) -> List[int]:
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking)
        return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    #  Arm B: static TWR (Eq. 2, exactly as specified) 
    def static_twr(self, bm25_ranking, faiss_ranking,
                    alpha=1.0, beta=0.6, gamma=0.6, delta=0.3) -> List[int]:
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking, alpha=1.0)
        twr_scores = {}
        for doc_idx, rrf in rrf_scores.items():
            doc = self.corpus[doc_idx]
            w1, w2, w3 = self.calculate_metadata_components(doc)
            rrf_norm_score = rrf / self.rrf_norm
            twr_scores[doc_idx] = alpha * rrf_norm_score + beta * w1 + gamma * w2 + delta * w3
        return sorted(twr_scores.keys(), key=lambda x: twr_scores[x], reverse=True)

    #  Feature vector shared by static TWR and LTWR 
    def _feature_vector(self, doc_idx, rrf_score, bm25_score, faiss_score_lookup):
        doc = self.corpus[doc_idx]
        w1, w2, w3 = self.calculate_metadata_components(doc)
        dense_score = faiss_score_lookup.get(doc_idx, 0.0)
        rrf_norm_score = rrf_score / self.rrf_norm
        return [rrf_norm_score, bm25_score, dense_score, w1, w2, w3]

    def build_features(self, query: str, top_n: int = 10):
        bm25_ranking, bm25_scores, faiss_ranking = self.hybrid_retrieval(query, top_n=top_n)
        return self.build_features_from_rankings(bm25_ranking, bm25_scores, faiss_ranking)

    def build_features_from_rankings(self, bm25_ranking, bm25_scores, faiss_ranking):
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking)
        faiss_score_lookup = {idx: 1.0 / (r + 1) for r, idx in enumerate(faiss_ranking)}

        candidate_idxs = set(bm25_ranking) | set(faiss_ranking)
        bm25_pool = [bm25_scores[idx] for idx in candidate_idxs]
        bm25_min, bm25_max = min(bm25_pool), max(bm25_pool)
        bm25_range = bm25_max - bm25_min

        def bm25_norm(idx):
            if bm25_range <= 0:
                return 0.0
            return (bm25_scores[idx] - bm25_min) / bm25_range

        feats = {}
        for doc_idx, rrf in rrf_scores.items():
            feats[doc_idx] = self._feature_vector(doc_idx, rrf, bm25_norm(doc_idx), faiss_score_lookup)
        return feats

    #  Arm C: LTWR (learned scalar-feature fusion) 
    def ltwr(self, bm25_ranking: List[int], bm25_scores: dict, faiss_ranking: List[int]) -> List[int]:
        """Pure Python scalar LTWR ranking using JSON-loaded weights (no scikit-learn / no fallback)."""
        if not hasattr(self, "ltwr_coef") or self.ltwr_coef is None:
            raise RuntimeError("LTWR JSON model not loaded .. call load_ltwr_model() first.")

        # 1. Compute baseline RRF scores
        rrf_scores = self._accumulate_rrf(bm25_ranking, faiss_ranking)
        if not rrf_scores:
            return []

        # 2. Build fast lookup tables for BM25 normalization and FAISS rank reciprocal
        candidate_idxs = set(bm25_ranking) | set(faiss_ranking)
        bm25_pool = [bm25_scores[idx] for idx in candidate_idxs]
        bm25_min, bm25_max = min(bm25_pool), max(bm25_pool)
        bm25_range = (bm25_max - bm25_min) if (bm25_max - bm25_min) > 0 else 1.0

        faiss_score_lookup = {idx: 1.0 / (r + 1) for r, idx in enumerate(faiss_ranking)}

        # 3. Pure scalar dot product loop (Zero NumPy matrix allocation)
        c0, c1, c2, c3, c4, c5 = self.ltwr_coef
        intercept = self.ltwr_intercept
        scores = {}

        for doc_idx, rrf in rrf_scores.items():
            doc = self.corpus[doc_idx]
            w1, w2, w3 = self.calculate_metadata_components(doc)
            
            rrf_norm = rrf / self.rrf_norm
            bm25_norm = (bm25_scores[doc_idx] - bm25_min) / bm25_range
            dense_val = faiss_score_lookup.get(doc_idx, 0.0)

            scores[doc_idx] = (c0 * rrf_norm + 
                               c1 * bm25_norm + 
                               c2 * dense_val + 
                               c3 * w1 + 
                               c4 * w2 + 
                               c5 * w3 + intercept)

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    def provenance(self, indices: List[int]) -> List[Dict[str, Any]]:
        out = []
        for rank, idx in enumerate(indices[: self.k_doc]):
            doc = self.corpus[idx]
            out.append({
                "rank": rank + 1,
                "chunk_id": doc.chunk_id,
                "cve_id": doc.cve_id,
                "severity": doc.severity,
                "vuln_status": doc.vuln_status,
                "cvss_version": doc.cvss_version,
                "pub_year": doc.pub_year,
                "package": doc.package,
                "ecosystem": doc.ecosystem,
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