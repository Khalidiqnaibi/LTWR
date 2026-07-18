# LTWR — Academic-Publishing Domain

**Learned Trust-Weighted Ranking**: RRF vs. static TWR vs. LTWR, instantiating
the original TWR equation

```
TWR(d) = alpha * RRF(d) + beta * w1(d) + gamma * w2(d) + delta * w3(d)
```

in the academic-publishing domain, per the specification:

- **w1 — Peer-Review Status**: 1.0 for a journal article, 0.4 for a preprint
  (a `ProceedingsArticle` tier at 0.8 is also included, since Crossref
  actually returns that type and collapsing it into one of the other two
  would misrepresent it — see `academic_domain/gains.py`).
- **w2 — Retraction Penalty**: 1.0 for no correction, 0.0 for retracted.
- **w3 — Recency Decay**: `e^(-lambda * age)`.

This replaces the earlier SEC-filing domain implementation entirely. Data
source is the Crossref REST API (`api.crossref.org`) — fully open, no API
key required.

## 0. Why this domain is cleaner than the SEC one

All three signals come from structured Crossref fields, not text heuristics:

| Signal | Crossref field | Notes |
|---|---|---|
| w1 | `type` | `journal-article` / `posted-content` / `proceedings-article` — exact, no parsing |
| w2 | `relation.is-retracted-by` | populated via publisher CrossMark metadata |
| w3 | `published` / `published-print` / `published-online` | structured date |

The one caveat worth stating plainly: **w2's coverage depends on publishers
registering CrossMark relation metadata**, so it under-counts some real
retractions (a retraction that exists but wasn't CrossMark-registered won't
show up via this field). `corpus_gen.py` compensates for this by pulling
confirmed-retracted works directly from Crossref's retraction notices
(`filter=update-type:retraction`) rather than relying on retractions turning
up naturally in topic-based search results — **this oversampling step is
required for w2 to have real variance, not optional.** See the module
docstring in `academic_domain/corpus_gen.py` for the SEC-domain bug this is
specifically designed to avoid repeating (audit tier there ended up
perfectly collinear with filing type because of a similar oversampling gap).

If you need broader retraction coverage than the `relation` field gives you,
Crossref also hosts the full Retraction Watch database via a separate Labs
endpoint requiring free registration — see the corpus_gen.py docstring for
the link, and swap `fetch_confirmed_retracted_works()` to that source if
needed.

## 1. Setup

```bash
pip install -r requirements.txt --break-system-packages
```

Edit `CONTACT_EMAIL` at the top of `academic_domain/corpus_gen.py` before
running — Crossref's polite pool (faster, more reliable responses) is keyed
off a real contact email in the User-Agent, similar in spirit to SEC's
User-Agent requirement but not strictly enforced.

First run of the retrieval pipeline downloads `all-MiniLM-L6-v2` (~80MB)
from HuggingFace Hub — needs normal internet access, cached after that.

## 2. Run order

```bash
# Step 1 -- pull the real Crossref corpus (several minutes: 8 fields x
# ~3 work types x rate-limited requests, plus retraction-notice resolution)
python domain/corpus_gen.py
# -> data_in/academic_corpus.json
# check the field/pub_type/retracted distribution before proceeding --
# see Section 3 for the exact check to run.

# Step 2 -- generate the query benchmark
python domain/query_gen.py
# -> data_in/academic_queries.json  (200 queries, 4 ablation dimensions)

# Step 3 -- train the LTWR model (train fields only)
python -m domain.train_ltwr
# -> domain/ltwr_model.pkl
# prints learned coefficients -- sanity-check signs and magnitudes before
# proceeding (see Section 4).

# Step 4 -- run all three arms on held-out test fields + full stats
python -m domain.run_experiment
# -> eval_results/academic_metrics_per_query.csv
# -> eval_results/academic_stats_report.csv
# -> eval_results/academic_fusion_latency.csv
```

## 3. Data-quality check to run before trusting any result

Same class of check that caught the SEC-domain's w2-collapse bug:

```python
import json
from collections import Counter

corpus = json.load(open("data_in/academic_corpus.json"))
print("retracted distribution:", Counter(d["retracted"] for d in corpus))
print("pub_type distribution:", Counter(d["pub_type"] for d in corpus))
print("retracted by pub_type:")
by_type = {}
for d in corpus:
    by_type.setdefault(d["pub_type"], Counter())[d["retracted"]] += 1
for pt, c in by_type.items():
    print(f"  {pt}: {dict(c)}")
```

You want: a real (non-trivial, ideally >5%) fraction of `retracted=True`
documents, AND that fraction should NOT be perfectly determined by
`pub_type` alone (i.e. retracted documents should appear across journal
articles, not just concentrated in one type) — if `w2` collapses into a
restatement of `w1` the way SEC's audit tier collapsed into filing type,
LTWR training will show the same near-zero coefficient/importance on `w2`
regardless of whether retraction actually matters in this domain.

## 4. What to check in the learned coefficients

`train_ltwr.py` prints something like:
```
=== LEARNED COEFFICIENTS ===
rrf_score       : ...
bm25_score      : ...
dense_score     : ...
w1_peer_review  : ...
w2_retraction   : ...
w3_recency      : ...
Intercept       : ...
```
Sanity checks before moving to Step 4:
- `w2_retraction` should have a clearly positive, non-trivial coefficient
  (retracted=0.0/not-retracted=1.0, so a positive coefficient means the
  model correctly learned to reward non-retracted work). Near-zero here
  after confirming real variance in Section 3 would itself be a genuine,
  reportable finding — not a bug — the same way the SEC domain's null
  result on learned-vs-static was.
- If `rrf_score` and `dense_score` end up with unstable or sign-flipped
  coefficients, that's collinearity between `rrf_score` and its two inputs
  (`bm25_score`/`dense_score`) — consider dropping `rrf_score` from the
  feature set (it's derived from the other two anyway) or increasing
  `alpha` in the `Ridge` call.

## 5. Re-running with a different train/test field split

Edit `TRAIN_FIELDS` / `TEST_FIELDS` in `academic_domain/train_ltwr.py` (also
imported by `run_experiment.py`). Keep the split field-disjoint — a random
query-level split leaks field-specific vocabulary between train and test.

## 6. Adjusting weights, gamma, or lambda

- `PUB_TYPE_WEIGHT` / retraction weights in `academic_domain/gains.py` — w1/w2
  as specified in the equation.
- `RECENCY_LAMBDA` in `academic_domain/gains.py` — currently 0.08, slower
  decay than the SEC domain's 0.15, reflecting that academic relevance
  persists longer than financial data currency. Re-run Steps 3-4 after any
  change here, since both static TWR's score and LTWR's training labels
  derive from the same `recency_weight()` call.
- `gamma` (retraction penalty's fusion weight) defaults higher than `beta`/
  `delta` in `static_twr()` — stated explicitly in that function's docstring
  as a design choice (retraction should suppress hard, not nudge), not a
  derived constant. Change it there if your paper wants a different
  weighting philosophy.