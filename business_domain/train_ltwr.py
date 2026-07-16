"""
train_ltwr.py -- fits the LTWR fusion function on the CLOSED, scalar
feature set [rrf_score, bm25_score, dense_score, w1, w2, w3]. No document
text is used as a feature, so training/inference costs the same as static
TWR's fusion step (a handful of scalar lookups + one small model.predict
call) -- not a per-candidate text forward pass.

Split is COMPANY-DISJOINT (train tickers != test tickers) to avoid the
leakage a random query split would introduce.
"""
import json
import pickle
import numpy as np
import lightgbm as lgb

from infra.business_document import BusinessDocument
from pipeline.business_retrieval import BusinessTWRPipeline
from business_domain.gains import filing_gain, audit_gain, recency_weight
from sklearn.linear_model import Ridge

# Company-disjoint split, matching corpus_gen.py's 15-company pull.
# Sectors kept balanced across train/test (2-3 train, 1-2 test per sector).
TRAIN_TICKERS = ["AAPL", "MSFT", "CRM", "TGT", "WMT", "JNJ", "PFE", "XOM", "CAT"]
TEST_TICKERS = ["HON", "JPM", "BAC", "CVX", "NUE", "UNH"]


def load_corpus(path="data_in/business_corpus.json"):
    raw = json.load(open(path))
    return [BusinessDocument(**r) for r in raw]


def combined_label(doc: BusinessDocument, current_year=2026) -> float:
    """Graded relevance target for LambdaMART: sum of the same trust signals
    static TWR uses, on one combined 0-~12 scale."""
    raw = filing_gain(doc.filing_type) + audit_gain(doc.audit_tier) + 4.0 * recency_weight(doc.filing_year, current_year)
    return int(round(raw))


def build_training_set(pipeline: BusinessTWRPipeline, queries, tickers_filter):
    X_rows, y_rows, group_sizes = [], [], []
    for q in queries:
        if q["ticker"] not in tickers_filter:
            continue
        feats = pipeline.build_features(q["query"], top_n=10)
        if not feats:
            continue
        for doc_idx, fv in feats.items():
            X_rows.append(fv)
            y_rows.append(combined_label(pipeline.corpus[doc_idx]))
        group_sizes.append(len(feats))
    return np.array(X_rows), np.array(y_rows), group_sizes



def main():
    corpus = load_corpus()
    queries = json.load(open("data_in/business_queries.json"))
    pipeline = BusinessTWRPipeline(corpus)

    # Build the exact same training matrix
    X_train, y_train, _ = build_training_set(pipeline, queries, TRAIN_TICKERS)

    # Fit a simple linear model
    # L2 regularization (Ridge) keeps the coefficients stable and prevents overfitting
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    feature_names = ["rrf_score", "bm25_score", "dense_score", "w1_filing", "w2_audit", "w3_recency"]
    coefficients = dict(zip(feature_names, model.coef_.tolist()))
    
    print("=== LEARNED COEFFICIENTS ===")
    for feat, weight in coefficients.items():
        print(f"{feat:12s} : {weight:+.6f}")
        
    print(f"Intercept    : {model.intercept_:+.6f}")

if __name__ == "__main__":
    main()