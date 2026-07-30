"""
train_ltwr_cve.py -- CVE/supply-chain-security analogue of
domain/train_ltwr.py (academic) and business_domain/train_ltwr.py (SEC),
BUT with the key architectural fix those domains could not make without
new annotation infrastructure: this module fits beta/gamma/delta against
REAL per-query top-k relevance judgments (data_in/ground_truth.json,
produced by cve_corpus_gen.py from NVD's own CPE-match linkage), using a
pairwise ranking loss -- not a regression against a self-declared scalar
label.

TWO TRAINING OBJECTIVES ARE IMPLEMENTED, BOTH REPORTED, NEITHER HIDDEN:

  1. fit_pairwise_ranking() -- the real fix. For each query, every pair
     (doc_i ranked above doc_j in the real ground truth) becomes a
     training constraint: score(doc_i) should exceed score(doc_j). This
     is standard pairwise Learning-to-Rank (RankNet-style logistic
     pairwise loss), fit via gradient descent over a linear scoring
     function with the SAME closed feature set static TWR uses
     [rrf_score, bm25_score, dense_score, w1, w2, w3], coefficients
     bounded to [0,1] to stay comparable to static TWR's alpha/beta/
     gamma/delta range. This is the objective whose results should be
     reported as "LTWR" in the paper's headline comparison against
     static TWR and RRF.

  2. fit_bounded_ridge() -- kept from the earlier domains' approach
     (regression to combined_label, an equal-weighted normalized sum of
     w1+w2+w3) ONLY as a labeled ablation/reference point, per the
     project's earlier finding that this objective measures "does LTWR
     reproduce its own declared training target better than static
     weights do" -- NOT retrieval quality. If reported at all, it must be
     labeled exactly that way, not as a second confirmation of (1).

Split is PACKAGE-DISJOINT (train packages != test packages), the
CVE-domain analogue of the SEC domain's company-disjoint split and the
academic domain's field-disjoint split, to avoid a query about a package
seen during training leaking into evaluation.

TRAIN_PACKAGES/TEST_PACKAGES are now loaded from
data_in/train_test_split.json (written by corpus_gen.py's
generate_corpus() at pull time) rather than hardcoded here. This closes a
real gap found during development: a hand-copied split can silently go
stale the moment the corpus is regenerated with a different package list
or different CPE-match results, while the hardcoded Python constants
keep referring to packages that may no longer exist (or may no longer be
excluded) in the actual data on disk. See load_train_test_split() below.
"""
import json
import pickle
import numpy as np
from scipy.optimize import lsq_linear

from infra.cve_document import CveDocument
from pipeline.retrieval import CveTWRPipeline
from domain.gains import severity_gain, vuln_status_gain, recency_weight, SEVERITY_GAIN
from domain.model_utils import BoundedLinearModel

# Package-disjoint split is now loaded from data_in/train_test_split.json,
# written by corpus_gen.py's generate_corpus() at pull time -- derived
# from whichever packages actually survived the >=2-ground-truth-CVEs
# filter on THIS run, not hand-copied from a previous run's printout.
# The earlier hardcoded TRAIN_PACKAGES/TEST_PACKAGES here were exactly
# how express/struts2/requests/spring-framework could silently go stale
# the moment the corpus was regenerated with a different package list or
# different CPE-match results: the hardcoded names would still exist as
# Python constants even after the underlying data changed underneath
# them. See load_train_test_split() below.
_FALLBACK_TRAIN_PACKAGES = ["curl", "jackson-databind", "log4j-core", "lodash", "flask"]
_FALLBACK_TEST_PACKAGES = ["openssl", "django", "openssh", "xz-utils"]

FEATURE_NAMES = ["rrf_score", "bm25_score", "dense_score", "w1_severity", "w2_vuln_status", "w3_recency"]
COEF_BOUNDS = (0.0, 1.0)  # matches static TWR's beta/gamma/delta range

SEVERITY_GAIN_MAX = max(SEVERITY_GAIN.values())  # = 4
COMBINED_LABEL_MAX = 3.0  # max of a 3-term sum of [0,1]-normalized signals


def load_train_test_split(path="data_in/train_test_split.json"):
    """Loads the package-disjoint split written by corpus_gen.py's
    generate_corpus() at pull time. Falls back to a hardcoded split ONLY
    if the file is missing (e.g. running against an older corpus pull
    from before this file existed), and prints a loud warning when that
    happens, since a stale hardcoded fallback silently matching or not
    matching the actual corpus on disk is exactly the failure mode this
    loader exists to prevent."""
    try:
        split = json.load(open(path))
        return split["train_packages"], split["test_packages"]
    except FileNotFoundError:
        print(f"*** WARNING: {path} not found -- falling back to a "
              f"hardcoded train/test split ({_FALLBACK_TRAIN_PACKAGES} / "
              f"{_FALLBACK_TEST_PACKAGES}). This fallback may not match "
              f"the packages actually present in your current "
              f"data_in/ground_truth.json. Re-run corpus_gen.py's "
              f"generate_corpus() to produce a real train_test_split.json "
              f"before trusting results from this fallback. ***")
        return _FALLBACK_TRAIN_PACKAGES, _FALLBACK_TEST_PACKAGES


TRAIN_PACKAGES, TEST_PACKAGES = load_train_test_split()


def load_corpus(path="data_in/corpus.json"):
    raw = json.load(open(path))
    return [CveDocument(**r) for r in raw]


def load_ground_truth(path="data_in/ground_truth.json"):
    """Loads the REAL top-k judgment file: {package: [cve_id, ...]},
    ordered most-relevant-first, derived from NVD's own CPE-match linkage
    (see cve_corpus_gen.py's generate_corpus()/build_seed_corpus()).
    This is the external signal the earlier domains' LTWR training
    lacked."""
    return json.load(open(path))


# ---------------------------------------------------------------------
# Objective 1 (the fix): pairwise ranking loss against real ground truth
# ---------------------------------------------------------------------
def build_pairwise_training_set(pipeline: CveTWRPipeline, queries, ground_truth: dict, packages_filter):
    """For each query, builds (feature_i, feature_j) pairs where doc_i is
    ranked strictly above doc_j in the real ground-truth ordering for that
    query's package. Only pairs where BOTH documents were actually
    retrieved by the shared BM25+dense stage are used (a document the
    retrieval stage never surfaces can't be reordered by the fusion layer
    no matter how the coefficients are set, so including it would just add
    noise to the loss)."""
    pair_features_i, pair_features_j = [], []
    n_queries_used = 0

    for q in queries:
        if q["package"] not in packages_filter:
            continue
        true_order = ground_truth.get(q["package"], [])
        if len(true_order) < 2:
            continue  # no real pair to learn from for this package

        feats = pipeline.build_features(q["query"], top_n=10)
        if len(feats) < 2:
            continue

        # Map retrieved doc_idx -> its cve_id, so we can look up each
        # retrieved doc's position in the real ground-truth order.
        idx_to_cve = {idx: pipeline.corpus[idx].cve_id for idx in feats}
        cve_to_true_rank = {cve_id: r for r, cve_id in enumerate(true_order)}

        retrieved_with_rank = [
            (idx, cve_to_true_rank[cve_id])
            for idx, cve_id in idx_to_cve.items()
            if cve_id in cve_to_true_rank
        ]
        if len(retrieved_with_rank) < 2:
            continue

        n_queries_used += 1
        # Every (i, j) pair where i has a better (lower) true rank than j
        # becomes one training constraint: score(i) > score(j).
        for a in range(len(retrieved_with_rank)):
            for b in range(len(retrieved_with_rank)):
                idx_i, rank_i = retrieved_with_rank[a]
                idx_j, rank_j = retrieved_with_rank[b]
                if rank_i < rank_j:  # i strictly more relevant than j
                    pair_features_i.append(feats[idx_i])
                    pair_features_j.append(feats[idx_j])

    return np.array(pair_features_i), np.array(pair_features_j), n_queries_used


def fit_pairwise_ranking(X_i, X_j, bounds=COEF_BOUNDS, lr=0.05, n_epochs=300, l2=0.01, seed=13):
    """RankNet-style pairwise logistic loss: for each training pair
    (x_i, x_j) where x_i should outrank x_j, minimize
        -log(sigmoid(w . (x_i - x_j)))
    via projected gradient descent, clipping w to `bounds` after every
    step (projected gradient, not clamp-after-convergence, for the same
    reason fit_bounded_ridge() in the earlier domains used a real bounded
    solve rather than an unconstrained fit clamped afterward -- clamping
    only at the end would not let the other coefficients rebalance around
    the constraint during optimization).

    No intercept: only the RELATIVE ordering implied by w . (x_i - x_j)
    matters for a ranking loss, so an intercept term would be exactly
    unidentifiable (it cancels in every pairwise difference) -- omitted
    rather than fit-and-ignored, to avoid a misleading nonzero-but-
    meaningless value in the printed coefficients.
    """
    rng = np.random.default_rng(seed)
    n_features = X_i.shape[1]
    w = rng.uniform(bounds[0], bounds[1], size=n_features)  # random init, per the
    # "set random coefficients, then adjust" instinct from the project's
    # earlier discussion -- valid here because the adjustment direction is
    # now driven by a real external loss (pairwise ground-truth ordering),
    # not by a self-referential score. This is what makes the same
    # iterative-adjustment MECHANISM valid this time.

    n_pairs = X_i.shape[0]
    for epoch in range(n_epochs):
        diff = X_i - X_j  # shape (n_pairs, n_features)
        margin = diff @ w  # shape (n_pairs,)
        sigmoid = 1.0 / (1.0 + np.exp(-margin))
        # gradient of -log(sigmoid(margin)) w.r.t. w is -(1-sigmoid)*diff
        grad = -((1.0 - sigmoid)[:, None] * diff).mean(axis=0) + l2 * w
        w = w - lr * grad
        w = np.clip(w, bounds[0], bounds[1])  # projected gradient step

    # Report final pairwise accuracy (fraction of training pairs correctly
    # ordered) as a fit-quality diagnostic -- not a held-out metric, just a
    # sanity check that the optimization actually moved w somewhere useful.
    final_margin = (X_i - X_j) @ w
    train_pairwise_acc = float((final_margin > 0).mean())

    return BoundedLinearModel(w, intercept=0.0), train_pairwise_acc


# ---------------------------------------------------------------------
# Objective 2 (reference/ablation only): regression to combined_label
# ---------------------------------------------------------------------
def combined_label(doc: CveDocument, current_year=2026) -> float:
    """The 'beta*w1(d) + gamma*w2(d) + delta*w3(d)' block of Eq. 2,
    evaluated with beta=gamma=delta=1 (equal, neutral weight). Kept ONLY
    as a labeled ablation against fit_pairwise_ranking() above -- fitting
    to this target measures optimization fidelity to a self-declared
    target, not retrieval quality against real relevance. See this
    module's docstring and README's circularity note before reporting
    results from this objective as if they were validated against
    ground truth."""
    severity = severity_gain(doc.severity) / SEVERITY_GAIN_MAX
    status = vuln_status_gain(doc.vuln_status) / 4.0  # VULN_STATUS_GAIN max is 4
    recency = recency_weight(doc.pub_year, current_year)
    return round(severity + status + recency, 4)


def fit_bounded_ridge(X, y, alpha=1.0, bounds=COEF_BOUNDS):
    """Bounded ridge via the augmented-least-squares trick, identical
    method to the academic/SEC domains' version -- see that docstring for
    why bounded-by-construction rather than clamped after an unconstrained
    fit."""
    n_samples, n_features = X.shape
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean

    X_aug = np.vstack([X_centered, np.sqrt(alpha) * np.eye(n_features)])
    y_aug = np.concatenate([y_centered, np.zeros(n_features)])

    result = lsq_linear(X_aug, y_aug, bounds=bounds, method="bvls")
    coef = result.x
    intercept = y_mean - X_mean @ coef
    return BoundedLinearModel(coef, intercept)


def build_regression_training_set(pipeline: CveTWRPipeline, queries, packages_filter):
    X_rows, y_rows = [], []
    for q in queries:
        if q["package"] not in packages_filter:
            continue
        feats = pipeline.build_features(q["query"], top_n=10)
        for doc_idx, fv in feats.items():
            X_rows.append(fv)
            y_rows.append(combined_label(pipeline.corpus[doc_idx]))
    return np.array(X_rows), np.array(y_rows)


def main():
    corpus = load_corpus()
    queries = json.load(open("data_in/queries.json"))
    ground_truth = load_ground_truth()
    pipeline = CveTWRPipeline(corpus)

    print("=" * 70)
    print("OBJECTIVE 1 (primary): pairwise ranking loss vs. real NVD-derived "
          "ground truth")
    print("=" * 70)
    X_i, X_j, n_q_used = build_pairwise_training_set(pipeline, queries, ground_truth, TRAIN_PACKAGES)
    print(f"Training pairs: {X_i.shape[0]}, from {n_q_used} queries "
          f"(train packages: {TRAIN_PACKAGES})")

    if X_i.shape[0] == 0:
        print("WARNING: zero training pairs generated -- check that "
              "cve_ground_truth.json covers TRAIN_PACKAGES and that the "
              "retrieval stage is actually surfacing ground-truth CVEs for "
              "these queries. Skipping pairwise fit.")
        pairwise_model = None
    else:
        pairwise_model, train_acc = fit_pairwise_ranking(X_i, X_j)
        coefficients = dict(zip(FEATURE_NAMES, pairwise_model.coef_.tolist()))
        print(f"Training pairwise accuracy: {train_acc:.4f}")
        print(f"\n=== LEARNED COEFFICIENTS (pairwise objective, bounded to {COEF_BOUNDS}) ===")
        for feat in FEATURE_NAMES:
            print(f"{feat:15s} : {coefficients[feat]:+.6f}")

        with open("domain/ltwr_cve_model.pkl", "wb") as f:
            pickle.dump(pairwise_model, f)
        print("Saved -> domain/ltwr_cve_model.pkl  (this is the model "
              "run_experiment_cve.py's 'ltwr' arm loads by default)")

    print("\n" + "=" * 70)
    print("OBJECTIVE 2 (reference/ablation ONLY -- NOT validated against "
          "real relevance, see docstring above before reporting)")
    print("=" * 70)
    X_train, y_train = build_regression_training_set(pipeline, queries, TRAIN_PACKAGES)
    if X_train.shape[0] == 0:
        print("WARNING: zero rows for regression ablation -- skipping.")
    else:
        ridge_model = fit_bounded_ridge(X_train, y_train / COMBINED_LABEL_MAX, alpha=1.0, bounds=COEF_BOUNDS)
        ridge_coefficients = dict(zip(FEATURE_NAMES, ridge_model.coef_.tolist()))
        print(f"\n=== LEARNED COEFFICIENTS (combined_label regression, ablation only) ===")
        for feat in FEATURE_NAMES:
            print(f"{feat:15s} : {ridge_coefficients[feat]:+.6f}")
        print(f"Intercept       : {ridge_model.intercept_:+.6f}  (unconstrained)")

        with open("domain/ltwr_cve_model_ablation_ridge.pkl", "wb") as f:
            pickle.dump(ridge_model, f)
        print("Saved -> domain/ltwr_cve_model_ablation_ridge.pkl  "
              "(ablation reference only -- do not load this as the "
              "headline 'ltwr' arm)")


if __name__ == "__main__":
    main()