from dataclasses import dataclass


@dataclass
class AcademicDocument:
    """Academic-publishing analogue of BusinessDocument.

    pub_type   -- one of JournalArticle, Preprint, ProceedingsArticle    (w1 signal)
    retracted  -- True if this work has an active retraction notice      (w2 signal)
    pub_year   -- year of publication                                   (w3 signal)
    """
    chunk_id: str
    text: str
    source: str          # e.g. "Crossref:10.1234/abcd.2023.001"
    doi: str
    pub_type: str
    retracted: bool
    pub_year: int
    venue: str            # journal / repository name
    field: str            # research field, e.g. "machine_learning", "oncology"
