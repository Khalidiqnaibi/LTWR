"""
run_experiment.py -- runs all three arms (RRF, static TWR, LTWR) on the
PACKAGE-DISJOINT test-package queries only (train packages were used to fit
the LTWR model in train_ltwr_cve.py). Reports:

  - Gain-based metrics (nDCG@3 severity, nDCG@3 vuln_status, MRR-authoritative)
    -- mirrors the metric family the clinical/academic/SEC papers report,
    kept for cross-domain comparability in the paper's Section 6 abstraction.

  - REAL ground-truth metrics (nDCG@3 vs. NVD's own CPE-matched top-k,
    MRR-to-first-ground-truth-hit) -- this is the metric family that
    actually answers "did trust-weighting improve retrieval of the true
    top documents," using data_in/ground_truth.json as the judgment
    source. This is the metric to lead with in the paper; the gain-based
    metrics above are secondary/comparability metrics, not proxies for
    this one.

  - Shapiro-Wilk-gated paired significance tests + Holm-Bonferroni
    correction, Cliff's delta effect sizes, and per-candidate fusion-step
    latency -- identical statistical protocol to the clinical paper and
    the earlier domains.
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from infra.cve_document import CveDocument
from pipeline.retrieval import CveTWRPipeline
from domain.gains import severity_gain, vuln_status_gain, is_authoritative
from domain.train_ltwr import TEST_PACKAGES, load_corpus, load_ground_truth
# Explicit import so a missing/renamed model_utils.py fails here with a clear
# ImportError, rather than as a confusing AttributeError inside pickle.load()
# below -- see domain/model_utils.py's docstring for why this class lives
# in its own module instead of inside train_ltwr_cve.py.
from domain.model_utils import BoundedLinearModel  # noqa: F401


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
        if is_authoritative(r["severity"], r["vuln_status"]):
            return 1.0 / rank
    return 0.0


def real_ndcg_at_k(results, true_order: list, k=3):
    """The real-ground-truth nDCG: gain for a retrieved doc at rank i is
    (len(true_order) - true_rank) if the doc appears in true_order, else 0
    -- i.e. a real graded relevance score derived directly from NVD's own
    CPE-matched ordering, not from this paper's w1/w2/w3 scoring. This is
    the metric that is NOT circular, unlike a metric built from
    severity_gain/vuln_status_gain (those measure 'does the ranking
    reflect the paper's own weights,' which was exactly the problem
    identified earlier in this project)."""
    cve_to_true_rank = {cve_id: r for r, cve_id in enumerate(true_order)}
    n = len(true_order)
    gains = []
    for r in results[:k]:
        true_rank = cve_to_true_rank.get(r["cve_id"])
        gains.append(float(n - true_rank) if true_rank is not None else 0.0)
    if not gains:
        return 0.0
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(gains))
    ideal_gains = sorted([float(n - r) for r in range(n)], reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def real_mrr(results, true_order: list):
    """MRR to the first document that appears anywhere in the real
    ground-truth top-k for this package (NOT restricted to true_order[0]
    specifically, since any ground-truth CVE surfaced early is a genuine
    retrieval success, matching how the clinical paper's MRR criterion is
    'any authoritative document', not 'the single most authoritative
    document')."""
    true_set = set(true_order)
    for rank, r in enumerate(results, start=1):
        if r["cve_id"] in true_set:
            return 1.0 / rank
    return 0.0


def query_metrics(results, true_order: list):
    sev_gains = [severity_gain(r["severity"]) for r in results]
    status_gains = [vuln_status_gain(r["vuln_status"]) for r in results]
    return {
        "ndcg3_severity": ndcg_at_k(sev_gains, 3),
        "ndcg3_vuln_status": ndcg_at_k(status_gains, 3),
        "mrr_authoritative": mrr_authoritative(results),
        "real_ndcg3": real_ndcg_at_k(results, true_order, 3),
        "real_mrr": real_mrr(results, true_order),
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
    if p_norm is not None and p_norm > 0.05:
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


METRICS = ["ndcg3_severity", "ndcg3_vuln_status", "mrr_authoritative", "real_ndcg3", "real_mrr"]


def main():
    corpus = load_corpus()
    ground_truth = load_ground_truth()
    all_queries = json.load(open("data_in/queries.json"))
    queries = [q for q in all_queries if q["package"] in TEST_PACKAGES]
    print(f"Evaluating on {len(queries)} held-out test-package queries "
          f"(packages: {sorted(set(q['package'] for q in queries))})")

    try:
        with open("domain/ltwr_cve_model.pkl", "rb") as f:
            ltwr_model = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            "domain/ltwr_cve_model.pkl not found -- run train_ltwr_cve.py "
            "first. Note: this must be the PAIRWISE-objective model (the "
            "default save target in train_ltwr_cve.py's main()), not "
            "ltwr_cve_model_ablation_ridge.pkl -- see that module's "
            "docstring for why the ablation model should not be reported "
            "as the headline LTWR arm."
        )

    pipeline = CveTWRPipeline(corpus, ltwr_model=ltwr_model)

    rows = []
    fusion_latency = {"rrf": [], "static_twr": [], "ltwr": []}

    for q in queries:
        true_order = ground_truth.get(q["package"], [])
        bm25_ranking, bm25_scores, faiss_ranking = pipeline.hybrid_retrieval(q["query"])

        t0 = time.perf_counter()
        idx_rrf = pipeline.rrf_only(bm25_ranking, faiss_ranking)
        fusion_latency["rrf"].append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        idx_static = pipeline.static_twr(bm25_ranking, faiss_ranking)
        fusion_latency["static_twr"].append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        idx_ltwr = pipeline.ltwr(bm25_ranking, bm25_scores, faiss_ranking)
        fusion_latency["ltwr"].append((time.perf_counter() - t0) * 1e6)

        for arm, idxs in [("rrf", idx_rrf), ("static_twr", idx_static), ("ltwr", idx_ltwr)]:
            res = pipeline.provenance(idxs)
            m = query_metrics(res, true_order)
            m.update({"qid": q["id"], "dimension": q["ablation_dimension"],
                      "package": q["package"], "arm": arm})
            rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv("eval_results/cve_metrics_per_query.csv", index=False)

    pivot = {}
    for metric in METRICS:
        pivot[metric] = df.pivot(index="qid", columns="arm", values=metric)

    print("\n=== Headline means by arm ===")
    for metric in METRICS:
        print(metric, pivot[metric].mean().to_dict())

    print("\n=== Pairwise significance tests (raw p, before correction) ===")
    test_results = []
    for metric in METRICS:
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
    stats_df.to_csv("eval_results/cve_stats_report.csv", index=False)
    print(stats_df.to_string(index=False))

    print("\n=== Fusion-step latency (microseconds per query) ===")
    for arm, lat in fusion_latency.items():
        print(f"{arm:12s} mean={np.mean(lat):8.1f}us  p95={np.percentile(lat,95):8.1f}us")

    lat_df = pd.DataFrame(fusion_latency)
    lat_df.to_csv("eval_results/cve_fusion_latency.csv", index=False)

    print("\nNOTE: 'real_ndcg3' and 'real_mrr' are the metrics validated "
          "against external ground truth (NVD's own CPE-match linkage). "
          "'ndcg3_severity', 'ndcg3_vuln_status', and 'mrr_authoritative' "
          "are gain-based metrics kept for comparability with the "
          "clinical/academic/SEC domains' metric family -- they measure "
          "agreement with this paper's own w1/w2/w3 scheme, not with "
          "independent relevance judgments. Lead with real_ndcg3/real_mrr "
          "in the paper's headline results table.")


if __name__ == "__main__":
    main()