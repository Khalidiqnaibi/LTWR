from dataclasses import dataclass


@dataclass
class CveDocument:
    """Software-vulnerability analogue of AcademicDocument / BusinessDocument.

    A CveDocument corresponds to one CVE (or GHSA-mirrored CVE) record, with
    metadata drawn from NVD's public CVE API 2.0 -- no annotation required,
    since NVD's own analysts have already assigned severity and review
    status as part of the CVE lifecycle.

    severity   -- one of Critical, High, Medium, Low, Unrated  (w1 signal)
                  Derived from the highest available CVSS base-severity
                  band (prefers CVSS v3.1, falls back to v3.0, then v2).
    vuln_status -- one of Analyzed, Modified, Awaiting Analysis,
                  Undergoing Analysis, Rejected               (w2 signal,
                  ORIGINAL definition)
                  NVD's own vulnStatus field: whether an NVD analyst has
                  reviewed and confirmed the record, is it still pending,
                  or was it withdrawn/disputed. CONFIRMED via corpus
                  coverage report to have ~zero variance (~100%
                  Analyzed/Modified) across every package in the current
                  17-package corpus of mature, actively-maintained
                  projects -- real corpus characteristic, not a bug.
    cvss_version -- one of "v3.1", "v3.0", "v2", "unscored"     (w2 signal,
                  CANDIDATE ALTERNATIVE)
                  Which CVSS scoring methodology NVD scored this CVE
                  under. Real, NVD-structural, non-self-authored --
                  v3.1 is NVD's current methodology (finalized 2019); a
                  CVE still carrying only a v3.0 or v2 score signals an
                  older record never rescored under the current standard.
                  Being evaluated as a replacement for vuln_status as w2,
                  since vuln_status showed no discriminative variance in
                  the current corpus -- see resolve_cvss_version() in
                  corpus_gen.py for the full rationale. Defaults to
                  "unscored" so CveDocument(**r) still works on any
                  corpus.json written before this field was added,
                  without needing to regenerate the corpus just to load
                  it.
    pub_year   -- calendar year the CVE was first published    (w3 signal)
    """
    chunk_id: str
    text: str
    source: str            # e.g. "NVD:CVE-2024-3094"
    cve_id: str
    severity: str
    vuln_status: str
    pub_year: int
    package: str            # affected package/product, e.g. "openssl", "xz-utils"
    ecosystem: str           # dependency ecosystem, e.g. "npm", "pypi", "maven", "system"
    cvss_version: str = "unscored"