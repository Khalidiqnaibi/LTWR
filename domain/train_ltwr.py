"""
train_ltwr.py .. CVE/supply-chain-security analogue of
domain/train_ltwr.py (academic) and business_domain/train_ltwr.py (SEC),
BUT with the key architectural fix those domains could not make without
new annotation infrastructure: this module fits beta/gamma/delta against
REAL per-query top-k relevance judgments (data_in/ground_truth.json,
produced by cve_corpus_gen.py from NVD's own CPE-match linkage), using a
pairwise ranking loss .. not a regression against a self-declared scalar
label.
"""
import json
import pickle
import numpy as np
from scipy.optimize import lsq_linear
from builtins import FileNotFoundError, dict, enumerate, float, len, max, open, print, range, round, zip
from pathlib import Path

from infra.cve_document import CveDocument
from pipeline.retrieval import CveTWRPipeline
from domain.gains import severity_gain, w2_gain, recency_weight, SEVERITY_GAIN, CVSS_VERSION_GAIN
from domain.model_utils import BoundedLinearModel

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
        print(f"*** WARNING: {path} not found .. falling back to a "
              f"hardcoded train/test split. ***")
        return _FALLBACK_TRAIN_PACKAGES, _FALLBACK_TEST_PACKAGES


TRAIN_PACKAGES, TEST_PACKAGES = load_train_test_split()


def load_corpus(path="data_in/corpus.json"):
    raw = json.load(open(path))
    return [CveDocument(**r) for r in raw]


def load_ground_truth(path="data_in/ground_truth.json"):
    return json.load(open(path))


def build_pairwise_training_set(pipeline, queries, ground_truth, packages_filter):
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

        retrieved_with_rank = [
            (idx, cve_to_true_rank[cve_id])
            for idx, cve_id in idx_to_cve.items()
            if cve_id in cve_to_true_rank
        ]
        off_package_idxs = [idx for idx, cve_id in idx_to_cve.items() if cve_id not in cve_to_true_rank]

        if len(retrieved_with_rank) < 1:
            continue

        n_queries_used += 1

        for a in range(len(retrieved_with_rank)):
            for b in range(len(retrieved_with_rank)):
                idx_i, rank_i = retrieved_with_rank[a]
                idx_j, rank_j = retrieved_with_rank[b]
                if rank_i < rank_j:
                    pair_features_i.append(feats[idx_i])
                    pair_features_j.append(feats[idx_j])

        for idx_i, _ in retrieved_with_rank:
            for idx_j in off_package_idxs:
                pair_features_i.append(feats[idx_i])
                pair_features_j.append(feats[idx_j])

    return np.array(pair_features_i), np.array(pair_features_j), n_queries_used


def fit_pairwise_ranking(X_i, X_j, bounds=COEF_BOUNDS, lr=0.05, n_epochs=300, l2=0.01, seed=13):
    rng = np.random.default_rng(seed)
    n_features = X_i.shape[1]
    w = rng.uniform(bounds[0], bounds[1], size=n_features)

    for _ in range(n_epochs):
        diff = X_i - X_j
        margin = diff @ w
        sigmoid = 1.0 / (1.0 + np.exp(-margin))
        grad = -((1.0 - sigmoid)[:, None] * diff).mean(axis=0) + l2 * w
        w = w - lr * grad
        w = np.clip(w, bounds[0], bounds[1])

    final_margin = (X_i - X_j) @ w
    train_pairwise_acc = float((final_margin > 0).mean())

    return BoundedLinearModel(w, intercept=0.0), train_pairwise_acc


W2_GAIN_MAX = max(CVSS_VERSION_GAIN.values())


def combined_label(doc, current_year=2026):
    severity = severity_gain(doc.severity) / SEVERITY_GAIN_MAX
    status = w2_gain(doc) / W2_GAIN_MAX
    recency = recency_weight(doc.pub_year, current_year)
    return round(severity + status + recency, 4)


def fit_bounded_ridge(X, y, alpha=1.0, bounds=COEF_BOUNDS):
    _, n_features = X.shape
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

    if X_i.shape[0] != 0:
        pairwise_model, train_acc = fit_pairwise_ranking(X_i, X_j)
        coefficients = dict(zip(FEATURE_NAMES, pairwise_model.coef_.tolist()))
        coefficients["intercept"] = float(pairwise_model.intercept_)
        print(f"Training pairwise accuracy: {train_acc:.4f}")
        for feat in FEATURE_NAMES:
            print(f"{feat:15s} : {coefficients[feat]:+.6f}")

        # ..-> REPLACE PICKLE SAVE WITH JSON SAVE <..-
        output_path = "domain/ltwr_cve_model.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_payload = {
            "coef": [float(w) for w in pairwise_model.coef_],
            "intercept": float(pairwise_model.intercept_),
            "features": FEATURE_NAMES
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_payload, f, indent=2)
        print(f"[+] Saved headline LTWR JSON weights to {output_path}")

    X_train, y_train = build_regression_training_set(pipeline, queries, TRAIN_PACKAGES)
    if X_train.shape[0] != 0:
        ridge_model = fit_bounded_ridge(X_train, y_train / COMBINED_LABEL_MAX, alpha=1.0, bounds=COEF_BOUNDS)
        ridge_coefficients = dict(zip(FEATURE_NAMES, ridge_model.coef_.tolist()))
        for feat in FEATURE_NAMES:
            print(f"{feat:15s} : {ridge_coefficients[feat]:+.6f}")

        # ..-> REPLACE RIDGE PICKLE SAVE WITH JSON SAVE <..-
        ablation_path = "domain/ltwr_cve_model_ablation_ridge.json"
        ablation_payload = {
            "coef": [float(w) for w in ridge_model.coef_],
            "intercept": float(ridge_model.intercept_),
            "features": FEATURE_NAMES
        }
        with open(ablation_path, "w", encoding="utf-8") as f:
            json.dump(ablation_payload, f, indent=2)
        print(f"[+] Saved ablation Ridge JSON weights to {ablation_path}")

if __name__ == "__main__":
    main()