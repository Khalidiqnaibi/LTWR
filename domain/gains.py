"""
gains.py -- CVE/supply-chain-security analogue of business_domain/gains.py
and domain/gains.py (academic). Same pattern: ordinal gain functions for
nDCG, plus an "authoritative document" definition for MRR, plus the w1/w2/w3
weight functions Eq. 2 of the TWR paper calls for.

w1 -- Severity tier. Derived from CVSS base severity (prefers v3.1, falls
      back to v3.0, then v2's HIGH/MEDIUM/LOW-only scale). This is NVD's
      own analyst-assigned severity band -- not self-authored -- mirroring
      how OCEBM evidence level is an external hierarchy tagged onto each
      clinical document, not something TWR invents.

w2 -- Advisory review status. NVD's own vulnStatus field: whether an NVD
      analyst has completed analysis (Analyzed/Modified -- reviewed,
      confirmed, CVSS-scored) vs. still pending (Awaiting Analysis /
      Undergoing Analysis -- exists, not yet reviewed) vs. Rejected
      (withdrawn/disputed -- the CVE ID was assigned but the record was
      later invalidated). This is the direct supply-chain analogue of SJR
      journal quartile: a third-party review-status hierarchy over the
      record itself, not a credential of who reported it.

w3 -- Recency decay, same exponential form as both other domains. Faster
      decay than academic literature (a 2019 CVE for a since-patched,
      since-deprecated package version matters far less to a "what
      vulnerabilities affect X today" query than a 2019 RCT still matters
      to a clinical question), but slower than SEC filings (the
      vulnerability's existence and severity don't change quarter to
      quarter the way financial figures do).
"""
import numpy as np

# w1: CVSS base-severity band. NVD's own baseSeverity field uses these
# four bands (plus an implicit "no CVSS score available" case for very
# old or reserved-but-unscored CVEs, gained/weighted as Unrated).
SEVERITY_GAIN = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unrated": 0}
SEVERITY_WEIGHT = {"Critical": 1.00, "High": 0.80, "Medium": 0.55, "Low": 0.30, "Unrated": 0.10}

# w2: NVD vulnStatus. "Analyzed" and "Modified" both indicate an NVD
# analyst has completed review and assigned CVSS/CPE data (Modified means
# a previously Analyzed record was later updated, not that it's less
# reviewed -- both are gained identically as fully reviewed). "Awaiting
# Analysis" and "Undergoing Analysis" mean the CVE ID exists but NVD
# hasn't finished review, analogous to "Unranked" in SJR: real, citable,
# but of lower certified trust than a completed record. "Rejected" means
# the CVE was withdrawn/disputed by its CNA -- the closest analogue to a
# retraction, and is gained/weighted at the floor, same treatment as
# retraction_weight() in the academic domain's gains.py.
#
# CONFIRMED (see corpus.json coverage check): for the current 17-package,
# mature-open-source-project corpus, vuln_status is ~74% Modified / ~26%
# Analyzed with ZERO Undergoing Analysis / Awaiting Analysis / Rejected
# records, across the full 2018-2026 publication-year range -- not a
# recency artifact. This means w2 is effectively a near-constant column
# for this corpus: both present values map to the same gain (4) and
# weight (1.00). This is a real property of well-tracked, actively-
# maintained packages (NVD finishes review on essentially everything for
# them), not a bug in the gain/weight tables below -- the tables are
# ready to differentiate the other three statuses correctly if a future
# corpus pull includes less-mature or very-recently-disclosed packages
# where those statuses actually occur. Report this as a stated corpus
# characteristic/limitation, not something to force a fix around: it
# explains both why static_twr's gamma*w2 term is a uniform additive
# shift with no discriminative effect, and why LTWR's learned w2
# coefficient has no gradient signal to move away from its random
# initialization (w2(doc_i) - w2(doc_j) = 0 for every pairwise training
# example, since Analyzed and Modified are gained identically).
VULN_STATUS_GAIN = {
    "Analyzed": 4,
    "Modified": 4,
    "Undergoing Analysis": 2,
    "Awaiting Analysis": 1,
    "Rejected": 0,
}
VULN_STATUS_WEIGHT = {
    "Analyzed": 1.00,
    "Modified": 1.00,
    "Undergoing Analysis": 0.55,
    "Awaiting Analysis": 0.35,
    "Rejected": 0.0,
}

# w2 REPLACEMENT: CVSS scoring version. vuln_status (above) was confirmed
# near-constant (~100% Analyzed/Modified) across every package in this
# corpus -- no discriminative signal. cvss_version, by contrast, was
# verified against the actual pulled corpus to have real spread: 90.1%
# v3.1 / 9.9% v3.0 overall, with meaningful per-package variance in 10 of
# 16 packages (including large ones -- curl 18% v3.0, django 12%, redis
# 25%, flask 44%) and non-degenerate spread on BOTH sides of the
# train/test split (train 13.5% v3.0, test 6.2% v3.0). No v2 or unscored
# records appear given MIN_PUB_YEAR=2018 (CVSS v2 was long phased out by
# then). This is a real, NVD-structural, non-self-authored signal --
# which CVSS methodology a CVE was scored under -- and is adopted here as
# the paper's actual w2, per the TWR paper's own stated principle that any
# structural weight with real supporting data can serve as a Wi (Section
# 6.1: "each Wi can represent any domain-specific hierarchical truth
# structure"). v3.1 is NVD's current, most rigorous methodology
# (finalized 2019); a record still carrying only a v3.0 score signals an
# older record never rescored under the current standard. No v2/unscored
# values appear in the current corpus, but both are included below for
# completeness/robustness if a future pull (e.g. lowering MIN_PUB_YEAR,
# or adding less-mature packages) surfaces them.
CVSS_VERSION_GAIN = {"v3.1": 3, "v3.0": 2, "v2": 1, "unscored": 0}
CVSS_VERSION_WEIGHT = {"v3.1": 1.00, "v3.0": 0.70, "v2": 0.40, "unscored": 0.10}

# Single source of truth for which signal retrieval.py's w2 slot uses.
# vuln_status_gain()/vuln_status_weight() are kept fully intact above
# (not deleted) since they were evaluated, are correctly implemented, and
# the paper should report both: cvss_version as the adopted w2, and
# vuln_status as a signal that was tried and found non-discriminative for
# this corpus -- see the module docstring's w2 section.
W2_SIGNAL = "cvss_version"  # "cvss_version" (adopted) or "vuln_status" (original, low-variance)

RECENCY_LAMBDA = 0.12  # faster decay than academic (0.08), slower than SEC filings (0.15)


def severity_gain(severity: str) -> float:
    return float(SEVERITY_GAIN.get(severity, 0))


def severity_weight(severity: str) -> float:
    """w1(d): CVSS-derived severity weight."""
    return float(SEVERITY_WEIGHT.get(severity, 0.10))


def vuln_status_gain(vuln_status: str) -> float:
    return float(VULN_STATUS_GAIN.get(vuln_status, 0))


def vuln_status_weight(vuln_status: str) -> float:
    """w2(d) ORIGINAL: NVD review-status weight. Kept intact -- evaluated,
    correctly implemented, confirmed non-discriminative for this corpus.
    Not called by retrieval.py's w2 slot by default; see W2_SIGNAL."""
    return float(VULN_STATUS_WEIGHT.get(vuln_status, 0.0))


def cvss_version_gain(cvss_version: str) -> float:
    return float(CVSS_VERSION_GAIN.get(cvss_version, 0))


def cvss_version_weight(cvss_version: str) -> float:
    """w2(d) ADOPTED: CVSS-scoring-version weight. See CVSS_VERSION_WEIGHT
    comment above for why this replaced vuln_status as w2."""
    return float(CVSS_VERSION_WEIGHT.get(cvss_version, 0.10))


def w2_gain(doc) -> float:
    """Dispatches to the active w2 signal per W2_SIGNAL. Accepts either a
    CveDocument-like object (attribute access) or a plain dict (e.g. from
    CveTWRPipeline.provenance()'s output) -- both expose vuln_status and
    cvss_version."""
    vuln_status = doc["vuln_status"] if isinstance(doc, dict) else doc.vuln_status
    cvss_version = doc["cvss_version"] if isinstance(doc, dict) else doc.cvss_version
    if W2_SIGNAL == "cvss_version":
        return cvss_version_gain(cvss_version)
    return vuln_status_gain(vuln_status)


def w2_weight(doc) -> float:
    """Dispatches to the active w2 signal per W2_SIGNAL. Accepts either a
    CveDocument-like object or a plain dict -- see w2_gain()."""
    vuln_status = doc["vuln_status"] if isinstance(doc, dict) else doc.vuln_status
    cvss_version = doc["cvss_version"] if isinstance(doc, dict) else doc.cvss_version
    if W2_SIGNAL == "cvss_version":
        return cvss_version_weight(cvss_version)
    return vuln_status_weight(vuln_status)


def recency_weight(pub_year: int, current_year: int = 2026, lam: float = RECENCY_LAMBDA) -> float:
    """w3(d): exponential recency decay, same form as Eq. 3 in the TWR paper."""
    age = max(0, current_year - pub_year)
    return float(np.exp(-lam * age))


def is_authoritative(severity: str, vuln_status: str) -> bool:
    """MRR criterion, mirroring the clinical paper's 'evidence<=2 and tier
    in Q1/Q2': a CVE record counts as authoritative if it's High-or-worse
    severity AND has completed NVD analyst review (Analyzed/Modified).

    NOTE: deliberately still keyed on vuln_status, not cvss_version, even
    though cvss_version replaced vuln_status as w2 in the fusion score.
    'Authoritative' here means 'review is complete' (a lifecycle-status
    question), which is conceptually distinct from 'scored under the
    current CVSS methodology' (a scoring-vintage question) -- conflating
    them would silently change what mrr_authoritative measures. Since
    vuln_status is near-constant in this corpus, mrr_authoritative is
    correspondingly close to a pure severity-based criterion here; this
    is the same corpus-characteristic caveat noted in the w2 discussion
    above, not a separate bug."""
    return severity in ("Critical", "High") and vuln_status in ("Analyzed", "Modified")