"""
gains.py .. CVE/supply-chain-security analogue of business_domain/gains.py
and domain/gains.py (academic). Same pattern: ordinal gain functions for
nDCG, plus an "authoritative document" definition for MRR, plus the w1/w2/w3
weight functions Eq. 2 of the TWR paper calls for.

w1 .. Severity tier. Derived from CVSS base severity (prefers v3.1, falls
      back to v3.0, then v2's HIGH/MEDIUM/LOW-only scale). This is NVD's
      own analyst-assigned severity band .. not self-authored .. mirroring
      how OCEBM evidence level is an external hierarchy tagged onto each
      clinical document, not something TWR invents.

w2 .. Advisory review status. NVD's own vulnStatus field: whether an NVD
      analyst has completed analysis (Analyzed/Modified .. reviewed,
      confirmed, CVSS-scored) vs. still pending (Awaiting Analysis /
      Undergoing Analysis .. exists, not yet reviewed) vs. Rejected
      (withdrawn/disputed .. the CVE ID was assigned but the record was
      later invalidated). This is the direct supply-chain analogue of SJR
      journal quartile: a third-party review-status hierarchy over the
      record itself, not a credential of who reported it.

w3 .. Recency decay, same exponential form as both other domains. Faster
      decay than academic literature (a 2019 CVE for a since-patched,
      since-deprecated package version matters far less to a "what
      vulnerabilities affect X today" query than a 2019 RCT still matters
      to a clinical question), but slower than SEC filings (the
      vulnerability's existence and severity don't change quarter to
      quarter the way financial figures do).
"""
import numpy as np

SEVERITY_GAIN = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unrated": 0}
SEVERITY_WEIGHT = {"Critical": 1.00, "High": 0.80, "Medium": 0.55, "Low": 0.30, "Unrated": 0.10}

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


CVSS_VERSION_GAIN = {"v3.1": 3, "v3.0": 2, "v2": 1, "unscored": 0}
CVSS_VERSION_WEIGHT = {"v3.1": 1.00, "v3.0": 0.70, "v2": 0.40, "unscored": 0.10}

W2_SIGNAL = "cvss_version" 

RECENCY_LAMBDA = 0.12  # faster decay than academic (0.08), slower than SEC filings (0.15)


def severity_gain(severity: str) -> float:
    return float(SEVERITY_GAIN.get(severity, 0))


def severity_weight(severity: str) -> float:
    """w1(d): CVSS-derived severity weight."""
    return float(SEVERITY_WEIGHT.get(severity, 0.10))


def vuln_status_gain(vuln_status: str) -> float:
    return float(VULN_STATUS_GAIN.get(vuln_status, 0))


def vuln_status_weight(vuln_status: str) -> float:
    """w2(d) ORIGINAL: NVD review-status weight. Kept intact .. evaluated,
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
    CveTWRPipeline.provenance()'s output) .. both expose vuln_status and
    cvss_version."""
    vuln_status = doc["vuln_status"] if isinstance(doc, dict) else doc.vuln_status
    cvss_version = doc["cvss_version"] if isinstance(doc, dict) else doc.cvss_version
    if W2_SIGNAL == "cvss_version":
        return cvss_version_gain(cvss_version)
    return vuln_status_gain(vuln_status)


def w2_weight(doc) -> float:
    """Dispatches to the active w2 signal per W2_SIGNAL. Accepts either a
    CveDocument-like object or a plain dict .. see w2_gain()."""
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
    severity AND has completed NVD analyst review (Analyzed/Modified)."""
    return severity in ("Critical", "High") and vuln_status in ("Analyzed", "Modified")