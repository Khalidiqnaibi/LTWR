# LTWR: Learned Trust-Weighted Retrieval

> **Mitigating the Semantic Trap in High-Stakes Retrieval-Augmented Generation (RAG) at Zero Online Latency.**

---

##  Overview

Standard hybrid retrieval pipelines combine sparse lexical search (BM25) and dense vector search (FAISS, Sentence-BERT) using Reciprocal Rank Fusion (RRF). While highly effective for topical relevance, **similarity-based rankers are mathematically blind to institutional authority and evidence quality**.

This creates the **Semantic Trap**: a low-evidence, unverified source (e.g., a single case report or forum post) outranking an authoritative source (e.g., a Cochrane systematic review or official NVD advisory) simply because its phrasing sits closer to the query embedding.

**LTWR (Learned Trust-Weighted Retrieval)** solves this by folding domain-specific structural trust metadata—such as evidence hierarchies, journal quartiles, authority levels, and temporal decay—directly into the rank fusion arithmetic.

``` bash
[BM25 + Dense Retrieval] ──> [Top-N Candidates] ──> [LTWR Zero-Latency Fusion] ──> [Ranked Output]
                                                           │
                                            ┌──────────────┴──────────────┐
                                            │ Structural Trust Metadata   │
                                            │ - Evidence Hierarchy (OCEBM)│
                                            │ - Institutional Tier (SJR)  │
                                            │ - Exponential Recency Decay │
                                            └─────────────────────────────┘

```

###  Key Advantages

* **$O(1)$ Zero Runtime Latency:** Operates entirely at the fusion layer over retrieved candidates. Requires no neural re-ranking forward passes, no cross-encoders, and zero GPU overhead.
* **Offline LTR Compiler Paradigm:** Uses Learning-to-Rank (LTR) as an *offline compiler* to discover optimal fusion coefficients ($\alpha, \beta, \gamma, \delta$), compiling them into static parameters applied instantaneously at runtime.
* **Determinism & Auditability:** Fully auditable, rule-based weights ensure safety compliance in critical domains (Clinical Medicine, Cybersecurity/CVE, Legal, Finance).
* **Proven Multi-Domain Generalization:** Empirically validated across gold-standard datasets in Clinical Medicine and Cybersecurity/CVE.

---

##  Mathematical Formulation

### Additive Trust Score

For a document $d$ retrieved across rankers $r \in R$, LTWR extends the standard Reciprocal Rank Fusion (RRF) score by adding weighted structural trust terms:

$$TWR(d) = \alpha \cdot RRF(d) + \sum_{i=1}^{n} c_i \cdot W_i(d)$$

Where:

* $RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}$ with standard smoothing $k=60$.
* $W_i(d) \in [0, 1]$ represents normalized structural trust factors (e.g., evidence level, journal standing, authority level).
* $c_i$ represents the fusion coefficients learned offline via LTWR or set statically.

### Structural Trust Weights (Clinical Instantiation)

$$\text{Score}(d) = \alpha \cdot RRF(d) + \beta \cdot w_1(d) + \gamma \cdot w_2(d) + \delta \cdot w_3(d)$$

1. **Evidence-Level Weight $w_1(d)$:** Derived from the Oxford Centre for Evidence-Based Medicine (OCEBM) hierarchy (Level 1 Systematic Review $= 1.00$ down to Level 5 Case Report $= 0.20$).
2. **Journal-Tier Weight $w_2(d)$:** Derived from SCImago Journal Rank (SJR) quartiles ($\text{Q1} = 1.00, \text{Q2} = 0.85, \text{Q3} = 0.70, \text{Q4} = 0.55$).
3. **Recency Weight $w_3(d)$:** Exponential decay over publication age:

$$w_3(d) = e^{-\lambda (Y_{\text{current}} - Y_{\text{pub}}(d))}$$



---

## Empirical Evaluation & Benchmark Results

LTWR was evaluated against standard Reciprocal Rank Fusion (RRF) across **170 paired gold-standard clinical queries** and **cybersecurity CVE threat databases**.

### Headline Results (Clinical Benchmark, $n=170$)

| Metric | Baseline (RRF) | TWR / LTWR | Improvement ($\Delta$) | Statistical Test | p-value (Holm-adj.) | Cliff's Delta ($\delta$) |
| --- | --- | --- | --- | --- | --- | --- |
| **nDCG@3 (Evidence Level)** | 0.781 | **0.957** | $+0.1761$ | Paired t-test | $1.25 \times 10^{-22}$ | **$+0.641$ (Large)** |
| **nDCG@3 (Journal Tier)** | 0.580 | **0.915** | $+0.3348$ | Wilcoxon | $2.47 \times 10^{-24}$ | **$+0.705$ (Large)** |
| **MRR (Authoritative Docs)** | 0.306 | **0.848** | $+0.5415$ | Wilcoxon | $1.52 \times 10^{-21}$ | **$+0.633$ (Large)** |

* **Top-1 Document Evidence Level:** Shifted from an average of **3.45** (Case-control/Cohort) under RRF to **1.44** (RCT/Systematic Review) under LTWR.
* **Top-1 Journal Standing:** Shifted from **Q3/Q4** under RRF to **Q1** on average under LTWR.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Khalidiqnaibi/LTWR.git
cd LTWR
pip install -r requirements.txt

```

### 2. Basic Usage

```python
from ltwr import TrustWeightedRanker, MetadataConfig

# Define structural metadata mapping rules
config = MetadataConfig(
    alpha=1.0,  # RRF weight
    beta=0.5,   # Evidence level weight
    gamma=0.5,  # Journal tier weight
    delta=0.3,  # Recency decay weight
    lambda_decay=0.05
)

# Initialize LTWR Ranker
ranker = TrustWeightedRanker(config=config)

# Sample retrieved candidate pool (BM25 + FAISS Dense outputs)
candidates = [
    {
        "id": "doc_101",
        "title": "Single-patient case report on treatment X",
        "bm25_rank": 0,
        "dense_rank": 1,
        "metadata": {"ocebm_level": 5, "sjr_quartile": "Q4", "year": 2015}
    },
    {
        "id": "doc_202",
        "title": "Systematic review and meta-analysis on treatment X",
        "bm25_rank": 2,
        "dense_rank": 0,
        "metadata": {"ocebm_level": 1, "sjr_quartile": "Q1", "year": 2023}
    }
]

# Run zero-latency fusion ranking
fused_results = ranker.rank(candidates)

for doc in fused_results:
    print(f"Rank: {doc['final_rank']} | Score: {doc['twr_score']:.4f} | Title: {doc['title']}")

```

### 3. Offline Parameter Optimization (LTWR Compiler)

```python
from ltwr.compiler import LTWRCompiler

# Fit optimal static coefficients offline using Learning-to-Rank on query logs
compiler = LTWRCompiler(loss_function="pairwise_hinge")
optimal_params = compiler.fit(query_logs=train_query_logs)

print("Learned Static Parameters:", optimal_params)
# Output: {'alpha': 0.85, 'beta': 0.62, 'gamma': 0.48, 'delta': 0.25}

```
---

##  Reproducibility

To execute the complete evaluation pipeline and recreate all figures and statistical tests reported in the paper:

```bash
python experiments/run_eval.py --dataset clinical_cte --output_dir results/

```

Every execution logs the raw sparse rank, dense rank, fused rank, and structured trust provenance metadata to an audit log file (`audit_trail.jsonl`).

---

##  Citation

If you use **LTWR** or **TWR** in your research or production RAG systems, please cite:

```bibtex
@article{iqnaibi2026ltwr,
  title={Trust-Weighted Retrieval (TWR): Mitigating the Semantic Trap in Clinical and High-Stakes Retrieval-Augmented Generation},
  author={Iqnaibi, Khalid},
  journal={Working Draft / IR Journal Submission},
  year={2026}
}

```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.