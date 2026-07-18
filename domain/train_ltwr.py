"""
train_ltwr.py -- fits the LTWR fusion function on the CLOSED, scalar
feature set [rrf_score, bm25_score, dense_score, w1, w2, w3] for the
academic-publishing domain. No document text is used as a feature.

Coefficients are BOUNDED to [0, 1] -- the same range static TWR's hand-set
alpha/beta/gamma/delta live in (see pipeline/academic_retrieval.py's
static_twr()). See fit_bounded_ridge()'s docstring for why bounded, not
unconstrained-then-clamped.

NOTE ON THE PICKLED MODEL CLASS: BoundedLinearModel lives in
domain/model_utils.py, NOT in this file, specifically so pickle can locate
it reliably regardless of how this script or run_experiment.py is invoked
(direct script run vs. `-m` vs. import) -- see that file's docstring for
why this matters. If you ever move or rename that class, you must retrain
(regenerate domain/ltwr_model.pkl) afterward; existing pickles reference
the old location and will NOT be fixed by changing the code.

Split is FIELD-DISJOINT (train fields != test fields) to avoid the leakage
a random query split would introduce -- the academic-domain analogue of the
SEC domain's company-disjoint split.
"""
import json
import pickle
import numpy as np
from scipy.optimize import lsq_linear

from infra.academic_document import AcademicDocument
from pipeline.academic_retrieval import AcademicTWRPipeline
from domain.gains import peer_review_gain, retraction_gain, recency_weight
from domain.model_utils import BoundedLinearModel

# Field-disjoint split, matching corpus_gen.py's 8-field pull.
TRAIN_FIELDS = ["machine_learning", "oncology", "climate_science", "psychology", "genomics"]
TEST_FIELDS = ["materials_science", "epidemiology", "economics"]

# Locked to the same 6-feature order _feature_vector() in
# pipeline/academic_retrieval.py produces -- do not reorder or shorten this
# list independently of that function, or coefficients will be silently
# mislabeled (this is exactly the bug that dropped w2_retraction from an
# earlier run's printout: the list and the array went out of sync).
FEATURE_NAMES = ["rrf_score", "bm25_score", "dense_score", "w1_peer_review", "w2_retraction", "w3_recency"]

COEF_BOUNDS = (0.0, 1.0)  # matches static TWR's beta/gamma/delta range


def fit_bounded_ridge(X, y, alpha=1.0, bounds=COEF_BOUNDS):
    """Bounded ridge via the augmented-least-squares trick: appending
    sqrt(alpha)*I rows to X (and zeros to y) applies an L2 penalty inside an
    ordinary bounded least-squares solve. This is a real bounded-ridge fit,
    not an unconstrained fit clamped after the fact -- clamping afterward
    would NOT recover the same (or a valid) optimum under the constraint,
    since the other coefficients wouldn't rebalance to compensate.
    Intercept is fit unconstrained via mean-centering, matching how sklearn
    handles intercepts for Ridge by default."""
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


def load_corpus(path="data_in/academic_corpus.json"):
    raw = json.load(open(path))
    return [AcademicDocument(**r) for r in raw]


def combined_label(doc: AcademicDocument, current_year=2026) -> float:
    """Graded relevance target: sum of the same trust signals static TWR
    uses, on one combined scale. Retraction gain is intentionally weighted
    ~2x, mirroring static TWR's higher default gamma -- a retracted work
    should pull the combined label down hard, not just nudge it."""
    raw = (peer_review_gain(doc.pub_type)
           + 2.0 * retraction_gain(doc.retracted)
           + 4.0 * recency_weight(doc.pub_year, current_year))
    return round(raw, 2)


def build_training_set(pipeline: AcademicTWRPipeline, queries, fields_filter):
    X_rows, y_rows, group_sizes = [], [], []
    for q in queries:
        if q["field"] not in fields_filter:
            continue
        feats = pipeline.build_features(q["query"], top_n=10)
        if not feats:
            continue
        for doc_idx, fv in feats.items():
            assert len(fv) == len(FEATURE_NAMES), (
                f"feature vector has {len(fv)} entries but FEATURE_NAMES has "
                f"{len(FEATURE_NAMES)} -- _feature_vector() and FEATURE_NAMES "
                f"have drifted out of sync, fix before training on mislabeled data."
            )
            X_rows.append(fv)
            y_rows.append(combined_label(pipeline.corpus[doc_idx]))
        group_sizes.append(len(feats))
    return np.array(X_rows), np.array(y_rows), group_sizes


def main():
    corpus = load_corpus()
    queries = json.load(open("data_in/academic_queries.json"))
    pipeline = AcademicTWRPipeline(corpus)

    X_train, y_train, group_train = build_training_set(pipeline, queries, TRAIN_FIELDS)
    print(f"Training rows: {X_train.shape}, groups: {len(group_train)}")

    model = fit_bounded_ridge(X_train, y_train, alpha=1.0, bounds=COEF_BOUNDS)

    coefficients = dict(zip(FEATURE_NAMES, model.coef_.tolist()))
    assert len(coefficients) == len(FEATURE_NAMES) == len(model.coef_), (
        "coefficient count mismatch -- FEATURE_NAMES and model.coef_ are out "
        "of sync, do not trust the printout below until this is fixed."
    )

    print(f"=== LEARNED COEFFICIENTS (bounded to {COEF_BOUNDS}) ===")
    for feat in FEATURE_NAMES:  # iterate FEATURE_NAMES directly, not the dict,
        print(f"{feat:15s} : {coefficients[feat]:+.6f}")  # so order is guaranteed
    print(f"Intercept       : {model.intercept_:+.6f}  (unconstrained)")

    with open("domain/ltwr_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved -> domain/ltwr_model.pkl")


if __name__ == "__main__":
    main()