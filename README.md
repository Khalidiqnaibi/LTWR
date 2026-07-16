# LTWR — Business/SEC-Filing Domain

Three-arm fusion study: **RRF vs. static TWR vs. LTWR** (Learned Fusion
Weights — a learned combination over the *same* closed scalar feature set
static TWR uses, not a text reranker), extending the original clinical TWR
paper into the SEC-filing domain.

This folder is meant to be dropped into the root of the
`Traceability-First-Retrieval` repo (it references `infra/` and `pipeline/`
paths from there).

Production build: real EDGAR data pull, real Sentence-BERT dense retriever.
No sandbox stand-ins remain in these files.

## 0. Before you run anything

- **Set a real SEC User-Agent.** Edit `USER_AGENT` at the top of
  `business_domain/corpus_gen.py` to `"Your Name (you@your-institution.edu)"`.
  SEC blocks generic/missing user agents and enforces a rate limit — the
  script already throttles to <=10 req/sec, don't remove that.
- **Audit tier is now a structured field, not a heuristic.** `corpus_gen.py`'s
  `resolve_audit_tier()` reads `dei:AuditorFirmId` (falling back to
  `dei:AuditorName`) directly from the company-facts API and exact-matches
  against a PCAOB Firm ID list for the Big 4 — no filing-text parsing. This
  field is only mandatory for fiscal years ending after Dec 15, 2021, so the
  corpus defaults to `MIN_FISCAL_YEAR = 2022` onward to avoid mixing in
  undisclosed years. `"Unknown"` should now be rare (a malformed/missing API
  response) rather than routine — if you see many, check the API response
  shape rather than assuming the field is genuinely absent.
- **`litigation` and `margin` facts are regex-extracted from filing text**
  (not standard XBRL tags) — illustrative patterns only, validate on a sample.
- **w1 (filing-type hierarchy) is 4 tiers: 10-K, 10-Q, 8-K, DEF14A.**
  PressRelease was dropped from the weight table entirely — EDGAR doesn't
  host company press releases (they're 8-K exhibits at best), and rather
  than leave a 5th tier defined but permanently empty in the corpus, `w1`
  in `pipeline/business_retrieval.py` and `FILING_RANK` in
  `business_domain/gains.py` now only define the 4 types the corpus
  actually contains. If a press-release tier is wanted later, it needs to
  be added back to both places plus a real data source (EDGAR 8-K exhibits
  or an external source) — don't just add the weight-table entry without
  populating the corpus, or you're back to the same empty-tier mismatch.

## 1. Setup

```bash
git clone https://github.com/Khalidiqnaibi/Trust-Weighted-Ranking.git
cd Trust-Weighted-Ranking
pip install -r requirements.txt --break-system-packages
```

First run of `pipeline/business_retrieval.py` will download the
`all-MiniLM-L6-v2` model (~80MB) from HuggingFace Hub — needs normal internet
access, cached locally after that.

## 2. Run order

Each step writes files the next step reads — run them in this order.


1. pull the real EDGAR corpus (rate-limited, several minutes for
 ~15 companies x ~4 filing types x ~7 years; SEC throttles at <=10 req/sec)
```bash
python business_domain/corpus_gen.py
```
-> data_in/business_corpus.json
 review the printed count and confirm "Unknown" audit_tier rows are rare
 (structured field now, not a text heuristic) -- see Section 0.

2. generate the query benchmark
```bash
python business_domain/query_gen.py
```
 --> data_in/business_queries.json  (200 queries, 4 ablation dimensions)

3. train the LTWR model (train tickers only)
```bash
python -m business_domain.train_ltwr
```
-> business_domain/ltwr_model.pkl
prints training-row count and feature importances -- sanity-check that
w1/w2/w3 dominate raw rrf/bm25/dense score, or the model isn't learning
what it's supposed to.

4. run all three arms on held-out test tickers + full stats
```bash
mkdir -p eval_results
python -m business_domain.run_experiment
```
-> eval_results/business_metrics_per_query.csv
-> eval_results/business_stats_report.csv
-> eval_results/business_fusion_latency.csv


Step 1 is the slow one (network-bound + rate limit). Steps 2-4 run in under a
minute on CPU for a corpus in the low thousands of chunks; the fusion logic
itself is unaffected by corpus size, only the BM25/dense index-build time
scales with it.

## 3. What to check in the output

- **`business_stats_report.csv`** — the headline table: mean delta, which
  test was used (Shapiro-Wilk decides t-test vs. Wilcoxon per metric),
  Holm-corrected p-value, Cliff's delta, and a significance flag, for all
  three pairwise arm comparisons across all three metrics (9 rows total).
- **`business_metrics_per_query.csv`** — per-query, per-arm nDCG@3 (filing),
  nDCG@3 (audit), and MRR — needed if you want to re-run stats stratified by
  `ablation_dimension` (filing_type / audit_tier / recency / combined).
- **`business_fusion_latency.csv`** — microseconds per query for each arm's
  fusion step only (retrieval stage excluded). This is the number that
  answers "does LTWR preserve TWR's zero-overhead property" — check it
  before claiming zero-overhead in the paper, don't assume it from the
  architecture alone. (On the earlier sandbox run this was ~50x slower than
  static TWR due to per-call model.predict() overhead, not architecture —
  worth re-checking on real data and considering a batched/compiled scorer
  if it still shows up.)

## 4. Re-running with a different train/test ticker split

Edit `TRAIN_TICKERS` / `TEST_TICKERS` in `business_domain/train_ltwr.py`
(also imported by `run_experiment.py`, so only needs to change in one place).
Keep the split company-disjoint — a random query-level split will leak
company-specific vocabulary between train and test and inflate LTWR's
apparent advantage. If you add/remove companies from `corpus_gen.py`'s
`companies` list, update both ticker lists to match.

## 5. Re-running statistics only (no re-training)

If you only changed the corpus or queries but not the model:
```bash
python -m business_domain.run_experiment
```
This reloads the existing `ltwr_model.pkl` — delete it first if you want a
clean retrain instead.

## 6. Adjusting the weight tables or λ

- `FILING_W1` / `AUDIT_W2` in `pipeline/business_retrieval.py` — static TWR's
  hand-set weights (Phase 0 spec). `AUDIT_W2` includes an explicit `"Unknown"`
  entry (weighted like `Unaudited`) for auditor-extraction misses — resolve
  those to a real tier once verified, don't leave them as Unknown at scale.
- `RECENCY_LAMBDA` in `business_domain/gains.py` — currently 0.15 (faster
  decay than the clinical paper's 0.05, reflecting faster financial-data
  staleness). Changing this changes both the static TWR score *and* the LTWR
  training labels, since both derive from the same `recency_weight()` call —
  re-run Steps 3-4 after any change here.