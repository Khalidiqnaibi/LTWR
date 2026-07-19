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
                  Undergoing Analysis, Rejected               (w2 signal)
                  NVD's own vulnStatus field: whether an NVD analyst has
                  reviewed and confirmed the record, is it still pending,
                  or was it withdrawn/disputed.
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
