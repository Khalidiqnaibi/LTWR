# LTWR: Learned Trust-Weighted Ranking

> **Mitigating the Semantic Trap in High-Stakes Retrieval-Augmented Generation (RAG) at Zero Online Latency.**

---

## Dataset: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21727536.svg)](https://doi.org/10.5281/zenodo.21727536)

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

### 1. Environment Setup

Make sure that you have a `.env` file with the shape of `.env.example` in the root directory of the project.

### 2. Clone the Repository

```
git clone https://github.com/Khalidiqnaibi/LTWR.git
cd LTWR
```

### 3. Dataset Preparation

Download the LTWR CVE dataset from Zenodo then extract it into the `data_in/` directory:

```bash
cd data_in
wget https://zenodo.org/api/records/21727536/files-archive
```

### 4. Installation & Execution

```bash
cd ..
pip install -r requirements.txt
python main.py
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
