"""
gains.py -- academic-publishing analogue of business_domain/gains.py.
Same pattern: ordinal gain functions for nDCG, plus an "authoritative
document" definition for MRR.
"""
import numpy as np

# w1: peer-review status. Journal articles and peer-reviewed conference
# proceedings are gained highest; preprints (posted-content on Crossref,
# e.g. arXiv/bioRxiv/SSRN) lowest of the two tiers named in the equation.
# ProceedingsArticle is included as a third real-world tier Crossref
# actually returns; treated as peer-reviewed but slightly below a full
# journal article to reflect typically lighter review at many venues --
# adjust if your sub-field disagrees (e.g. top-tier CS conferences would
# arguably deserve 1.0, not 0.8).
PUB_TYPE_GAIN = {"JournalArticle": 4, "ProceedingsArticle": 3, "Preprint": 1}
PUB_TYPE_WEIGHT = {"JournalArticle": 1.0, "ProceedingsArticle": 0.8, "Preprint": 0.4}

RECENCY_LAMBDA = 0.08  # slower decay than SEC filings -- academic relevance persists longer


def peer_review_gain(pub_type: str) -> float:
    return float(PUB_TYPE_GAIN.get(pub_type, 0))


def peer_review_weight(pub_type: str) -> float:
    """w1(d) as specified: 1.0 journal article, 0.4 preprint."""
    return float(PUB_TYPE_WEIGHT.get(pub_type, 0.4))


def retraction_gain(retracted: bool) -> float:
    """0/1 gain for nDCG, mirrors retraction_weight's 0.0/1.0 scale directly."""
    return 0.0 if retracted else 1.0


def retraction_weight(retracted: bool) -> float:
    """w2(d) as specified: 1.0 for no correction, 0.0 for retracted."""
    return 0.0 if retracted else 1.0


def recency_weight(pub_year: int, current_year: int = 2026, lam: float = RECENCY_LAMBDA) -> float:
    """w3(d) as specified: exponential recency decay."""
    age = max(0, current_year - pub_year)
    return float(np.exp(-lam * age))


def is_authoritative(pub_type: str, retracted: bool) -> bool:
    """MRR criterion: a peer-reviewed, non-retracted work."""
    return pub_type in ("JournalArticle", "ProceedingsArticle") and not retracted