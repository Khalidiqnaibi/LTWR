# TWR — Software Supply-Chain / CVE Domain

Third domain instantiation of the TWR paper's Equation 2, after clinical
(original paper) and SEC filings. This one closes a gap the earlier two
could not close without new annotation infrastructure: **real, external,
pre-existing top-k relevance judgments**, sourced from NVD's own CPE-match
linkage between a CVE and the software it affects.

## 0. Why this domain, and what it fixes

The clinical and SEC domains both had genuine, citable, externally-defined
`w1/w2/w3` — but neither had an independent signal for "which documents
are the true top-k for this query." Fitting LTWR's `beta/gamma/delta`
against anything derived only from `w1/w2/w3` and `RRF(d)` is circular:
there is no data outside those four inputs to check the fit against.

CVE/NVD data closes that gap for free. NVD analysts already link each CVE
to the specific package/product/version it affects (CPE matching), as
part of their normal review work, for a purpose that predates and is
unaware of this paper. A query like *"what vulnerabilities affect
`log4j-core`"* therefore has a real, pre-existing answer: the actual CVEs
NVD has CPE-matched to that package. `domain/corpus_gen.py` derives
`data_in/ground_truth.json` — `{package: [cve_id, ...]}`, most-severe
/ most-recent first — directly from that linkage, and
`domain/train_ltwr_cve.py` fits `beta/gamma/delta` with a **pairwise
ranking loss against that real ordering**, not a regression to a
self-declared scalar label.

## 1. Domain mapping (Eq. 2)

| Symbol | Clinical (original) | SEC (2nd domain) | CVE (this domain) |
|---|---|---|---|
| Document | Paper/article | Filing chunk | CVE/GHSA advisory record |
| `w1` | OCEBM evidence level | Filing type hierarchy | **Severity** (CVSS base-severity band, prefers v3.1 → v3.0 → v2) |
| `w2` | SJR journal quartile | ICFR attestation status | **Advisory review status** (NVD `vulnStatus`: Analyzed/Modified/Awaiting Analysis/Undergoing Analysis/Rejected) |
| `w3` | Recency decay | Recency decay | Recency decay (`λ=0.12`, faster than academic's `0.08`, slower than SEC's `0.15`) |
| Query | Clinical question | "What does filing X say about Y" | "What vulnerabilities affect package X" |
| Ground truth | *(none — static weights only)* | *(none — static weights only)* | **NVD's own CPE-matched CVE list per package** |

`w2` deliberately uses NVD's *review status*, not the *reporter's*
identity — the same design principle used when the SEC domain's `w2` was
switched from auditor identity to ICFR attestation status: grade the
finding, not the grader.

## 2. Files

``` bash
infra/cve_document.py        CveDocument dataclass
domain/gains.py           w1/w2/w3 weight + gain functions, is_authoritative()
domain/corpus_gen.py      NVD API pull + offline seed corpus + ground-truth derivation
pipeline/retrieval.py     3-arm pipeline (RRF / static TWR / LTWR), mirrors academic_retrieval.py
domain/query_gen.py       package-scoped query benchmark generator
domain/train_ltwr_cve.py      PRIMARY: pairwise ranking loss vs. real ground truth
                               ABLATION ONLY: bounded-ridge regression to combined_label
domain/run_experiment_cve.py  full stats battery; reports BOTH gain-based and real-ground-truth metrics
domain/model_utils.py         BoundedLinearModel (copied unchanged from the academic domain)
```

## 3. Run order

```bash
# 1. Build corpus + ground truth. Live NVD pull needs network access to
#    services.nvd.nist.gov (NOT reachable from every sandboxed
#    environment -- this was developed against the offline fallback,
#    see note below). For a real run:
python -c "
from domain.corpus_gen import generate_corpus, SEED_ECOSYSTEM_LOOKUP
packages = ['xz-utils','log4j-core','openssl','lodash','express','flask',
            'django','spring-framework','struts2','jackson-databind',
            'openssh','curl','requests']
generate_corpus(packages, SEED_ECOSYSTEM_LOOKUP)
"
# Offline fallback used for development/testing (no network needed):
python domain/corpus_gen.py    # -> data_in/corpus.json + ground_truth.json

# 2. Generate the query benchmark (package-scoped, 4 dimensions x N packages)
python domain/query_gen.py     # -> data_in/queries.json

# 3. Train LTWR -- fits beta/gamma/delta via pairwise loss against REAL
#    ground truth (primary), plus the combined_label ablation (reference
#    only, do not report as validated against relevance)
python domain/train_ltwr_cve.py    # -> domain/ltwr_model.pkl (+ ablation pkl)

# 4. Run the full 3-arm comparison + stats battery on package-disjoint
#    test queries
python domain/run_experiment_cve.py
# -> eval_results/metrics_per_query.csv
# -> eval_results/stats_report.csv
# -> eval_results/fusion_latency.csv
```

## 4. NETWORK NOTE (read before trusting any numbers)

`services.nvd.nist.gov` was not reachable from the sandboxed environment
this code was authored in. `domain/corpus_gen.py` was therefore
developed and tested against `build_seed_corpus()` — a small (17-record),
hand-verified set of real, publicly documented CVEs (Log4Shell,
Heartbleed, the xz-utils backdoor, Spring4Shell, etc., with accurate
severity/status/year metadata) — NOT a live NVD pull. The end-to-end
pipeline (pairwise training, real-ground-truth nDCG/MRR, the full stats
battery) was verified correct against this seed corpus, including a
directional check that reversing a known-good ranking order lowers
`real_ndcg3` as expected.

**Before reporting any numbers in the paper, re-run step 1 with
`generate_corpus()` against live NVD data, on a machine with real network
access, across a package list large enough to support a benchmark
comparable in size to the clinical paper's 170 queries** (the seed
corpus's 13-package, 52-query benchmark is a development-scale
placeholder, not a submission-scale one).

## 5. On the two training objectives in `train_ltwr_cve.py`

- **`fit_pairwise_ranking()` (primary, report this as "LTWR"):** fits
  `beta/gamma/delta` (as a bounded linear model over
  `[rrf_score, bm25_score, dense_score, w1, w2, w3]`) via a RankNet-style
  pairwise logistic loss against `ground_truth.json`. This is
  standard supervised Learning-to-Rank against real judgments — the fix
  this project's discussion converged on. Coefficients are randomly
  initialized then adjusted via projected gradient descent; this is the
  same "start somewhere, iteratively adjust" mechanism proposed earlier
  in the project's discussion, made valid here specifically because the
  adjustment direction comes from a real external loss (pairwise
  ground-truth ordering) rather than from a self-referential score built
  only out of `w1/w2/w3` and `RRF(d)`.

- **`fit_bounded_ridge()` (ablation only, label it as such if reported):**
  regresses to `combined_label()`, an equal-weighted sum of normalized
  `w1+w2+w3`. This measures "does LTWR reproduce its own declared
  training target better than static weights do" — a fact about
  optimization fidelity, not about retrieval quality. Kept only as a
  labeled comparison point; do not present its results as validated
  against real relevance.

## 6. Statistics reported

`run_experiment_cve.py` reports two metric families, computed on the same
package-disjoint test queries:

- `ndcg3_severity`, `ndcg3_vuln_status`, `mrr_authoritative` — gain-based,
  same family the clinical/academic/SEC papers report. Kept for
  cross-domain comparability in the paper's Section 6 abstraction table.
  **These measure agreement with this paper's own `w1/w2/w3` scheme, not
  independent relevance** — do not present them as validating LTWR
  against ground truth.

- `real_ndcg3`, `real_mrr` — computed directly against
  `ground_truth.json`. **These are the metrics that actually answer
  whether trust-weighting improves retrieval of the true top documents.
  Lead with these in the paper's headline results table.**

Both families go through the identical Shapiro-Wilk-gated paired
test → Holm-Bonferroni correction → Cliff's delta pipeline as the
clinical paper's Section 4.3, for every `(metric, arm-pair)` combination.
