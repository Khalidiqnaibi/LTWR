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

RECENCY_LAMBDA = 0.12  # faster decay than academic (0.08), slower than SEC filings (0.15)


def severity_gain(severity: str) -> float:
    return float(SEVERITY_GAIN.get(severity, 0))


def severity_weight(severity: str) -> float:
    """w1(d): CVSS-derived severity weight."""
    return float(SEVERITY_WEIGHT.get(severity, 0.10))


def vuln_status_gain(vuln_status: str) -> float:
    return float(VULN_STATUS_GAIN.get(vuln_status, 0))


def vuln_status_weight(vuln_status: str) -> float:
    """w2(d): NVD review-status weight."""
    return float(VULN_STATUS_WEIGHT.get(vuln_status, 0.0))


def recency_weight(pub_year: int, current_year: int = 2026, lam: float = RECENCY_LAMBDA) -> float:
    """w3(d): exponential recency decay, same form as Eq. 3 in the TWR paper."""
    age = max(0, current_year - pub_year)
    return float(np.exp(-lam * age))


def is_authoritative(severity: str, vuln_status: str) -> bool:
    """MRR criterion, mirroring the clinical paper's 'evidence<=2 and tier
    in Q1/Q2': a CVE record counts as authoritative if it's High-or-worse
    severity AND has completed NVD analyst review (Analyzed/Modified)."""
    return severity in ("Critical", "High") and vuln_status in ("Analyzed", "Modified")