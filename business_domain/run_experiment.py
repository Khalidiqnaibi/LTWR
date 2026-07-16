"""
run_experiment.py -- runs all three arms (RRF, static TWR, LTWR) on the
COMPANY-DISJOINT test-ticker queries only. Reports:
  - per-query metrics (filing-gain nDCG@3, audit-gain nDCG@3, MRR-authoritative)
  - Shapiro-Wilk-gated paired significance tests + Holm-Bonferroni correction
  - Cliff's delta effect sizes
  - per-candidate fusion-step latency (the zero-overhead check)
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from infra.business_document import BusinessDocument
from pipeline.business_retrieval import BusinessTWRPipeline
from business_domain.gains import filing_gain, audit_gain, is_authoritative
from business_domain.train_ltwr import TEST_TICKERS, load_corpus


def ndcg_at_k(gains, k=3):
    g = gains[:k]
    if not g:
        return 0.0
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(g))
    ideal = sorted(gains, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_authoritative(results):
    for rank, r in enumerate(results, start=1):
        if is_authoritative(r["filing_type"], r["audit_tier"]):
            return 1.0 / rank
    return 0.0


def query_metrics(results):
    fg = [filing_gain(r["filing_type"]) for r in results]
    ag = [audit_gain(r["audit_tier"]) for r in results]
    return {
        "ndcg3_filing": ndcg_at_k(fg, 3),
        "ndcg3_audit": ndcg_at_k(ag, 3),
        "mrr": mrr_authoritative(results),
    }


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (n_x * n_y)


def paired_test(a, b, label):
    diffs = np.array(a) - np.array(b)
    if np.allclose(diffs, 0):
        return {"metric": label, "mean_delta": 0.0, "p_raw": 1.0, "test": "none", "cliffs_delta": 0.0}
    _, p_norm = stats.shapiro(diffs) if len(diffs) >= 3 else (None, 0.0)
    if p_norm > 0.05:
        stat, p = stats.ttest_rel(a, b)
        test_name = "paired t-test"
    else:
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, 1.0
        test_name = "Wilcoxon"
    delta = cliffs_delta(a, b)
    return {"metric": label, "mean_delta": float(np.mean(diffs)), "p_raw": float(p),
            "test": test_name, "cliffs_delta": float(delta)}


def main():
    corpus = load_corpus()
    queries = [q for q in json.load(open("data_in/business_queries.json")) if q["ticker"] in TEST_TICKERS]
    print(f"Evaluating on {len(queries)} held-out test-ticker queries "
          f"(tickers: {sorted(set(q['ticker'] for q in queries))})")

    with open("business_domain/ltwr_model.pkl", "rb") as f:
        ltwr_model = pickle.load(f)

    pipeline = BusinessTWRPipeline(corpus, ltwr_model=ltwr_model)

    rows = []
    fusion_latency = {"rrf": [], "static_twr": [], "ltwr": []}

    for q in queries:
        # Pre-retrieval phase: executes outside the timed fusion blocks
        bm25_ranking, bm25_scores, faiss_ranking = pipeline.hybrid_retrieval(q["query"])

        # Arm A (RRF) latency measurement
        t0 = time.perf_counter()
        idx_rrf = pipeline.rrf_only(bm25_ranking, faiss_ranking)
        fusion_latency["rrf"].append((time.perf_counter() - t0) * 1e6)

        # Arm B (Static TWR) latency measurement
        t0 = time.perf_counter()
        idx_static = pipeline.static_twr(bm25_ranking, faiss_ranking)
        fusion_latency["static_twr"].append((time.perf_counter() - t0) * 1e6)

        # Arm C (LTWR) latency measurement -- ONLY measures feature vectorization & ML inference
        t0 = time.perf_counter()
        idx_ltwr = pipeline.ltwr(bm25_ranking, bm25_scores, faiss_ranking)
        fusion_latency["ltwr"].append((time.perf_counter() - t0) * 1e6)

        for arm, idxs in [("rrf", idx_rrf), ("static_twr", idx_static), ("ltwr", idx_ltwr)]:
            res = pipeline.provenance(idxs)
            m = query_metrics(res)
            m.update({"qid": q["id"], "dimension": q["ablation_dimension"], "arm": arm})
            rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv("eval_results/business_metrics_per_query.csv", index=False)

    pivot = {}
    for metric in ["ndcg3_filing", "ndcg3_audit", "mrr"]:
        pivot[metric] = df.pivot(index="qid", columns="arm", values=metric)

    print("\n=== Headline means by arm ===")
    for metric in ["ndcg3_filing", "ndcg3_audit", "mrr"]:
        print(metric, pivot[metric].mean().to_dict())

    print("\n=== Pairwise significance tests (raw p, before correction) ===")
    test_results = []
    for metric in ["ndcg3_filing", "ndcg3_audit", "mrr"]:
        p = pivot[metric]
        for a_arm, b_arm in [("static_twr", "rrf"), ("ltwr", "rrf"), ("ltwr", "static_twr")]:
            r = paired_test(p[a_arm], p[b_arm], f"{metric}::{a_arm}_vs_{b_arm}")
            test_results.append(r)

    pvals = [r["p_raw"] for r in test_results]
    reject, p_adj, _, _ = multipletests(pvals, method="holm")
    for r, p_a, sig in zip(test_results, p_adj, reject):
        r["p_holm"] = float(p_a)
        r["significant"] = bool(sig)

    stats_df = pd.DataFrame(test_results)
    stats_df.to_csv("eval_results/business_stats_report.csv", index=False)
    print(stats_df.to_string(index=False))

    print("\n=== Fusion-step latency (microseconds per query, candidate pool ~10-20 docs) ===")
    for arm, lat in fusion_latency.items():
        print(f"{arm:12s} mean={np.mean(lat):8.1f}us  p95={np.percentile(lat,95):8.1f}us")

    lat_df = pd.DataFrame(fusion_latency)
    lat_df.to_csv("eval_results/business_fusion_latency.csv", index=False)


if __name__ == "__main__":
    main()