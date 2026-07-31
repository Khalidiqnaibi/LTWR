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
from domain.gains import severity_gain, w2_gain, recency_weight, SEVERITY_GAIN, CVSS_VERSION_GAIN
from domain.model_utils import BoundedLinearModel

# Fallback split kept in sync with the real data_in/train_test_split.json
# as of this project's most recent verified corpus pull (739 records, 16
# packages, post-CPE-hint-fix). Still only a fallback for the rare case
# the JSON file is missing -- load_train_test_split() prefers the real
# file whenever it exists, and warns loudly when it doesn't. This
# constant will go stale again the next time corpus_gen.py is re-run
# with a different package list; that's expected and fine, since the
# real file is always preferred when present -- this exists only so an
# accidental missing-file case degrades to something plausible instead
# of silently wrong package names like the original express/struts2/
# requests/spring-framework fallback did.
_FALLBACK_TRAIN_PACKAGES = ["curl", "django", "postgresql", "axios", "redis", "flask", "xz-utils", "guava"]
_FALLBACK_TEST_PACKAGES = ["openssl", "tomcat", "jackson-databind", "openssh", "log4j-core", "lodash", "spring-framework", "pyyaml"]

FEATURE_NAMES = ["rrf_score", "bm25_score", "dense_score", "w1_severity", "w2_cvss_version", "w3_recency"]
COEF_BOUNDS = (0.0, 1.0)

SEVERITY_GAIN_MAX = max(SEVERITY_GAIN.values())
COMBINED_LABEL_MAX = 3.0


def load_train_test_split(path="data_in/train_test_split.json"):
    try:
        split = json.load(open(path))
        return split["train_packages"], split["test_packages"]
    except FileNotFoundError:
        print(f"*** WARNING: {path} not found -- falling back to a "
              f"hardcoded train/test split. ***")
        return _FALLBACK_TRAIN_PACKAGES, _FALLBACK_TEST_PACKAGES


TRAIN_PACKAGES, TEST_PACKAGES = load_train_test_split()


def load_corpus(path="data_in/corpus.json"):
    raw = json.load(open(path))
    return [CveDocument(**r) for r in raw]


def load_ground_truth(path="data_in/ground_truth.json"):
    return json.load(open(path))


def build_pairwise_training_set(pipeline, queries, ground_truth, packages_filter):
    """
    an earlier version of this
    function ONLY formed pairs between two documents that were BOTH
    already confirmed members of the query's own package's ground truth
    (the `if cve_id in cve_to_true_rank` filter below used to apply to
    BOTH pair members). That meant every training pair was, by
    construction, already package-pure -- rrf_score/bm25_score/dense_score
    carried zero signal for the comparisons the model was actually fit on
    (only severity/w2/recency mattered, since true_order ranks by
    severity-then-recency), so gradient descent correctly learned to zero
    out retrieval-relevance features entirely. At test time,
    hybrid_retrieval() searches the WHOLE corpus with no package
    restriction, so an off-package document CAN leak into a query's
    candidate pool -- and a model with near-zero rrf/bm25/dense weight has
    no signal telling it that document doesn't belong, so it ranks purely
    on severity/recency regardless of package. This is why real_mrr
    collapsed to ~0.44 (below even plain RRF's ~0.86) despite training
    pairwise accuracy of 0.92: the model was never given a single training
    example teaching it "an off-package document should rank below an
    on-package one."

    FIX: retrieved candidates NOT in the query's package ground truth are
    no longer discarded -- they're kept as guaranteed-WORSE members of a
    pair against every genuine ground-truth document retrieved for that
    query. This directly supplies the missing training signal.
    """
    pair_features_i, pair_features_j = [], []
    n_queries_used = 0

    for q in queries:
        if q["package"] not in packages_filter:
            continue
        true_order = ground_truth.get(q["package"], [])
        if len(true_order) < 2:
            continue

        feats = pipeline.build_features(q["query"], top_n=10)
        if len(feats) < 2:
            continue

        idx_to_cve = {idx: pipeline.corpus[idx].cve_id for idx in feats}
        cve_to_true_rank = {cve_id: r for r, cve_id in enumerate(true_order)}

        # Ground-truth-ranked candidates (in-package, real relevance order).
        retrieved_with_rank = [
            (idx, cve_to_true_rank[cve_id])
            for idx, cve_id in idx_to_cve.items()
            if cve_id in cve_to_true_rank
        ]
        # NEW: off-package candidates that were retrieved but are NOT in
        # this package's ground truth at all -- these are the negative
        # examples the model needs to see, not discard.
        off_package_idxs = [idx for idx, cve_id in idx_to_cve.items() if cve_id not in cve_to_true_rank]

        if len(retrieved_with_rank) < 1:
            continue  # need at least one real ground-truth hit to anchor any pair against

        n_queries_used += 1

        # Original pairs: better-ranked-in-ground-truth beats worse-ranked
        # (only meaningful if >=2 ground-truth hits were retrieved).
        for a in range(len(retrieved_with_rank)):
            for b in range(len(retrieved_with_rank)):
                idx_i, rank_i = retrieved_with_rank[a]
                idx_j, rank_j = retrieved_with_rank[b]
                if rank_i < rank_j:
                    pair_features_i.append(feats[idx_i])
                    pair_features_j.append(feats[idx_j])

        # NEW: every in-package ground-truth hit beats every off-package
        # candidate that was retrieved alongside it for this query.
        for idx_i, _rank_i in retrieved_with_rank:
            for idx_j in off_package_idxs:
                pair_features_i.append(feats[idx_i])
                pair_features_j.append(feats[idx_j])

    return np.array(pair_features_i), np.array(pair_features_j), n_queries_used


def fit_pairwise_ranking(X_i, X_j, bounds=COEF_BOUNDS, lr=0.05, n_epochs=300, l2=0.01, seed=13):
    rng = np.random.default_rng(seed)
    n_features = X_i.shape[1]
    w = rng.uniform(bounds[0], bounds[1], size=n_features)

    n_pairs = X_i.shape[0]
    for epoch in range(n_epochs):
        diff = X_i - X_j
        margin = diff @ w
        sigmoid = 1.0 / (1.0 + np.exp(-margin))
        grad = -((1.0 - sigmoid)[:, None] * diff).mean(axis=0) + l2 * w
        w = w - lr * grad
        w = np.clip(w, bounds[0], bounds[1])

    final_margin = (X_i - X_j) @ w
    train_pairwise_acc = float((final_margin > 0).mean())

    return BoundedLinearModel(w, intercept=0.0), train_pairwise_acc


W2_GAIN_MAX = max(CVSS_VERSION_GAIN.values())  # = 3, for the currently-active w2 signal
# (was VULN_STATUS_GAIN's max of 4 when vuln_status was w2 -- this constant
# must track whichever signal domain.gains.W2_SIGNAL currently selects, or
# combined_label()'s status term silently compresses to the wrong range.)


def combined_label(doc, current_year=2026):
    """The 'beta*w1(d) + gamma*w2(d) + delta*w3(d)' block of Eq. 2,
    evaluated with beta=gamma=delta=1 (equal, neutral weight). Kept ONLY
    as a labeled ablation against fit_pairwise_ranking() above -- fitting
    to this target measures optimization fidelity to a self-declared
    target, not retrieval quality against real relevance. See this
    module's docstring and README's circularity note before reporting
    results from this objective as if they were validated against
    ground truth.

    Uses w2_gain(doc), which dispatches per domain.gains.W2_SIGNAL
    (currently cvss_version, previously vuln_status) -- normalized by
    W2_GAIN_MAX, the max of whichever gain table is currently active, not
    a hardcoded constant. Hardcoding /4.0 here (VULN_STATUS_GAIN's max)
    would silently under-scale the status term now that w2 dispatches to
    CVSS_VERSION_GAIN (max 3), compressing it to at most 0.75 instead of
    1.0 relative to the severity and recency terms."""
    severity = severity_gain(doc.severity) / SEVERITY_GAIN_MAX
    status = w2_gain(doc) / W2_GAIN_MAX
    recency = recency_weight(doc.pub_year, current_year)
    return round(severity + status + recency, 4)


def fit_bounded_ridge(X, y, alpha=1.0, bounds=COEF_BOUNDS):
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


def build_regression_training_set(pipeline, queries, packages_filter):
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

    X_i, X_j, n_q_used = build_pairwise_training_set(pipeline, queries, ground_truth, TRAIN_PACKAGES)
    print(f"Training pairs: {X_i.shape[0]}, from {n_q_used} queries "
          f"(train packages: {TRAIN_PACKAGES})")

    if X_i.shape[0] == 0:
        pairwise_model = None
    else:
        pairwise_model, train_acc = fit_pairwise_ranking(X_i, X_j)
        coefficients = dict(zip(FEATURE_NAMES, pairwise_model.coef_.tolist()))
        print(f"Training pairwise accuracy: {train_acc:.4f}")
        for feat in FEATURE_NAMES:
            print(f"{feat:15s} : {coefficients[feat]:+.6f}")

        with open("domain/ltwr_cve_model.pkl", "wb") as f:
            pickle.dump(pairwise_model, f)

    X_train, y_train = build_regression_training_set(pipeline, queries, TRAIN_PACKAGES)
    if X_train.shape[0] == 0:
        pass
    else:
        ridge_model = fit_bounded_ridge(X_train, y_train / COMBINED_LABEL_MAX, alpha=1.0, bounds=COEF_BOUNDS)
        ridge_coefficients = dict(zip(FEATURE_NAMES, ridge_model.coef_.tolist()))
        for feat in FEATURE_NAMES:
            print(f"{feat:15s} : {ridge_coefficients[feat]:+.6f}")

        with open("domain/ltwr_cve_model_ablation_ridge.pkl", "wb") as f:
            pickle.dump(ridge_model, f)


if __name__ == "__main__":
    main()