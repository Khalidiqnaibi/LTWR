from dataclasses import dataclass


@dataclass
class BusinessDocument:
    """SEC-filing analogue of ClinicalDocument.

    filing_type   -- one of 10-K, 10-Q, 8-K, DEF14A                 (w1 signal)
    audit_tier    -- one of Big4, OtherAudited, Unaudited, Unknown  (w2 signal)
    filing_year   -- calendar year of filing                        (w3 signal)
    """
    chunk_id: str
    text: str
    source: str          # e.g. "EDGAR:AAPL:10-K:2025"
    filing_type: str
    audit_tier: str
    filing_year: int
    ticker: str
    domain: str           # sector, e.g. "technology", "retail"