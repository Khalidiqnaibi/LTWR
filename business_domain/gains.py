"""
gains.py -- business-domain analogue of evidence_gain()/tier_gain() in
eval/ablation_eval.py. Same integer 0-4 ordinal scale, same "authoritative
document" pattern used for MRR.
"""
import numpy as np

# Ordinal rank of filing type by informational authority (1 = highest)
FILING_RANK = {"10-K": 1, "10-Q": 2, "DEF14A": 3, "8-K": 4}
# "Unknown" = corpus_gen.py's resolve_audit_tier() couldn't confidently
# identify the auditor from filing text -- gained like Unaudited until
# manually verified.
AUDIT_GAIN = {"Big4": 4, "OtherAudited": 2, "Unaudited": 0, "Unknown": 0}
RECENCY_LAMBDA = 0.15  # re-tuned for financial-data staleness (faster decay than clinical 0.05)


def filing_gain(filing_type: str) -> float:
    rank = FILING_RANK.get(filing_type, 5)
    return max(0.0, 6.0 - float(rank))


def audit_gain(audit_tier: str) -> float:
    return float(AUDIT_GAIN.get(str(audit_tier), 0))


def recency_weight(filing_year: int, current_year: int = 2026, lam: float = RECENCY_LAMBDA) -> float:
    age = max(0, current_year - filing_year)
    return float(np.exp(-lam * age))


def is_authoritative(filing_type: str, audit_tier: str) -> bool:
    """Mirrors the clinical MRR criterion (evidence<=2 and tier in Q1/Q2):
    a filing counts as authoritative if it's a primary audited filing
    (10-K or 10-Q) from a Big4 or other-audited firm."""
    return filing_type in ("10-K", "10-Q") and audit_tier in ("Big4", "OtherAudited")