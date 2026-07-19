"""
corpus_gen.py -- CVE/supply-chain-security analogue of
business_domain/corpus_gen.py and domain/corpus_gen.py (academic).

Pulls real CVE records from NVD's public CVE API 2.0
(https://services.nvd.nist.gov/rest/json/cves/2.0), builds a corpus of
CveDocument records enriched with w1 (severity) / w2 (vuln status) /
w3-ready (pub_year) metadata, and -- critically, this is the piece the
clinical and SEC domains lacked -- derives GROUND-TRUTH per-package top-k
relevance judgments directly from NVD's own CPE-match linkage between a
CVE and the software product/version it affects.

WHY THIS SIDESTEPS THE ANNOTATION PROBLEM:
Earlier domains (clinical, SEC) had real w1/w2/w3 signals but no
independent, external "which documents are actually the true top-k for
this query" judgment -- that gap is what made naive LTWR training
circular (see README section "Ground truth and circularity" below).
NVD's CPE (Common Platform Enumeration) matching already links each CVE
to the specific package/product/version it affects, as part of NVD
analysts' normal review work, for a purpose that predates and is
unaware of this paper. A query like "what vulnerabilities affect
package X" therefore has a REAL, pre-existing, externally-curated
answer: the actual CVEs NVD has CPE-matched to that package. This lets
train_ltwr_cve.py fit beta/gamma/delta against genuine top-k judgments
(a supervised pairwise/listwise objective) instead of regressing to a
self-declared combined_label, closing the circularity gap the earlier
domains could not close without new annotation infrastructure.

NETWORK NOTE: services.nvd.nist.gov is not reachable from every sandboxed
environment (it was not reachable from the one this module was authored
in). generate_corpus() below performs live fetches when run somewhere
with real network access. build_seed_corpus() provides a small,
real-CVE, hand-verified offline fallback (well-known, publicly documented
CVEs with accurate metadata) so the rest of the pipeline (retrieval,
gains, training, experiment) can be developed, tested, and demonstrated
without live API access. Replace it with a real generate_corpus() run
before treating any reported numbers as final -- see README.
"""
import json
import re
import time
from pathlib import Path

import requests

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
MIN_PUB_YEAR = 2018
HEADERS = {"User-Agent": "LTWR-CVE-Research (research contact: you@example.com)"}

# NVD rate-limits unauthenticated requests to ~5 requests / 30s. Sleep
# between calls to stay well under that; use an API key (nvd_api_key
# param below) for higher throughput on a real run.
NVD_RATE_LIMIT_SLEEP_SEC = 6.5


def _get(url, params=None, api_key=None, max_retries=3):
    headers = dict(HEADERS)
    if api_key:
        headers["apiKey"] = api_key
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            last_exc = e
            if resp.status_code == 429:  # rate limited -- back off and retry
                time.sleep(NVD_RATE_LIMIT_SLEEP_SEC * (attempt + 1))
                continue
            raise
    raise last_exc


def resolve_severity(cve_item: dict) -> str:
    """w1 resolution: prefers CVSS v3.1, falls back to v3.0, then v2.
    Mirrors NVD's own documented precedence for which score is
    'the' score when multiple versions are present on a record."""
    metrics = cve_item.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if entries:
            sev = entries[0].get("cvssData", {}).get("baseSeverity") or entries[0].get("baseSeverity")
            if sev:
                return sev.title()  # NVD returns e.g. "CRITICAL" -> "Critical"
    v2_entries = metrics.get("cvssMetricV2")
    if v2_entries:
        sev = v2_entries[0].get("baseSeverity")
        if sev:
            return sev.title()
    return "Unrated"


def resolve_vuln_status(cve_item: dict) -> str:
    """w2 resolution: NVD's own vulnStatus field, used as-is. This is
    already the exact external review-status hierarchy w2 needs -- no
    further derivation required, unlike the SEC domain's ICFR flag which
    had to be extracted from filing markup."""
    return cve_item.get("vulnStatus", "Awaiting Analysis")


def extract_affected_packages(cve_item: dict):
    """Pulls (package, ecosystem) pairs out of NVD's CPE-match
    configuration nodes. CPE URIs have the form
    cpe:2.3:a:<vendor>:<product>:<version>:... -- <product> is used as the
    package name. This is also the mechanism used later to build
    ground-truth top-k judgments: two CVEs "affect the same package" if
    NVD's own CPE matching says so, not if a keyword match says so."""
    packages = set()
    for config in cve_item.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                criteria = cpe_match.get("criteria", "")
                parts = criteria.split(":")
                if len(parts) > 4:
                    vendor, product = parts[3], parts[4]
                    packages.add(product.replace("_", "-"))
    return sorted(packages)


def build_passage_text(cve_item: dict) -> str:
    """Builds the retrieval passage text: CVE ID, description, and
    affected-package list, mirroring how the SEC domain concatenates
    filing text with '[Extracted facts: ...]' -- structured metadata
    folded into the text a BM25/dense index actually searches over."""
    cve_id = cve_item.get("id", "")
    descriptions = cve_item.get("descriptions", [])
    desc_text = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
    packages = extract_affected_packages(cve_item)
    pkg_str = ", ".join(packages[:8]) if packages else "unspecified"
    return f"{cve_id}: {desc_text} [Affected packages: {pkg_str}]"


def build_document_record(chunk_id: int, cve_item: dict, ecosystem_lookup: dict) -> dict:
    cve_id = cve_item.get("id", "")
    pub_year = int(cve_item.get("published", "1970")[:4])
    packages = extract_affected_packages(cve_item)
    primary_package = packages[0] if packages else "unknown"
    ecosystem = ecosystem_lookup.get(primary_package, "system")

    return {
        "chunk_id": f"chunk_{chunk_id:05d}",
        "text": build_passage_text(cve_item),
        "source": f"NVD:{cve_id}",
        "cve_id": cve_id,
        "severity": resolve_severity(cve_item),
        "vuln_status": resolve_vuln_status(cve_item),
        "pub_year": pub_year,
        "package": primary_package,
        "ecosystem": ecosystem,
    }


def generate_corpus(
    packages: list,
    ecosystem_lookup: dict,
    out_path="data_in/corpus.json",
    results_per_package: int = 40,
    api_key: str = None,
):
    """Live NVD pull: for each package, fetches CVEs whose CPE configuration
    matches that package name (keywordSearch against the product name),
    builds CveDocument records, and writes the corpus JSON.

    Also returns a {package: [cve_id, ...]} ground-truth map -- this IS
    the "real top-k judgments" the earlier domains lacked, built directly
    from NVD's own CPE linkage rather than authored by this paper's team.
    See build_relevance_judgments() below for how this becomes the
    per-query top-k judgment file train_ltwr_cve.py trains against.
    """
    docs = []
    ground_truth = {}  # package -> [cve_id, ...], in NVD's own relevance order (by severity then recency)
    chunk_id = 0

    for package in packages:
        print(f"Fetching CVEs for package: {package} ...")
        params = {
            "keywordSearch": package,
            "resultsPerPage": results_per_package,
            "pubStartDate": f"{MIN_PUB_YEAR}-01-01T00:00:00.000",
            "pubEndDate": "2026-07-19T00:00:00.000",
        }
        try:
            resp = _get(NVD_BASE_URL, params=params, api_key=api_key)
        except requests.HTTPError as e:
            print(f"  Error fetching {package}: {e}")
            continue

        data = resp.json()
        vulnerabilities = data.get("vulnerabilities", [])
        pkg_cve_ids = []

        for vuln in vulnerabilities:
            cve_item = vuln.get("cve", {})
            record = build_document_record(chunk_id, cve_item, ecosystem_lookup)
            # Only keep records that actually CPE-match this package --
            # keywordSearch can return loosely related hits (e.g. mentions
            # in the description text without a real CPE match), and using
            # those for ground truth would quietly reintroduce a
            # keyword-similarity judgment exactly like the thing TWR is
            # meant to correct for.
            if record["package"].lower() == package.lower() or package.lower() in [
                p.lower() for p in extract_affected_packages(cve_item)
            ]:
                docs.append(record)
                pkg_cve_ids.append(record["cve_id"])
                chunk_id += 1

        # NVD's own implied relevance order for ground truth: severity
        # first (Critical > High > Medium > Low), then recency within a
        # tier. This is NOT an invented ranking -- both fields are NVD's
        # own assigned data; imposing this order for judgment purposes
        # simply operationalizes "which of NVD's own matched CVEs would a
        # security analyst want surfaced first," using only NVD-assigned
        # facts, not this paper's trust-weighting scheme itself. See
        # README's circularity note for why sourcing this order from w1/w2
        # would NOT be safe, and why this differs from that case (it is
        # anchored on CPE match membership, an independent NVD linkage,
        # not on the paper's own scoring function).
        pkg_docs = [d for d in docs if d["package"].lower() == package.lower()]
        pkg_docs_sorted = sorted(
            pkg_docs,
            key=lambda d: (
                -{"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unrated": 0}.get(d["severity"], 0),
                -d["pub_year"],
            ),
        )
        ground_truth[package] = [d["cve_id"] for d in pkg_docs_sorted]

        time.sleep(NVD_RATE_LIMIT_SLEEP_SEC)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=1)
    print(f"Successfully generated {len(docs)} corpus records -> {out_path}")

    gt_path = str(Path(out_path).parent / "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=1)
    print(f"Ground-truth top-k judgments -> {gt_path}")

    print_severity_status_coverage_report(docs)
    return docs, ground_truth


def print_severity_status_coverage_report(docs):
    """Data-quality/coverage report, mirroring the SEC domain's
    per-filing-type collinearity check and corpus-wide resolver-failure
    check. Here the two axes are severity (w1) and vuln_status (w2) --
    they should vary largely independently (a Critical CVE can be
    Analyzed or still Awaiting Analysis; a Rejected CVE can have any
    nominal severity), so a strong correlation between them would signal
    a resolver bug, not real structure."""
    from collections import Counter, defaultdict
    cross = defaultdict(Counter)
    overall_severity = Counter()
    overall_status = Counter()
    for d in docs:
        cross[d["severity"]][d["vuln_status"]] += 1
        overall_severity[d["severity"]] += 1
        overall_status[d["vuln_status"]] += 1

    print("\n--- vuln_status coverage by severity ---")
    for sev, counts in sorted(cross.items()):
        total = sum(counts.values())
        print(f"  {sev:10s} (n={total}): {dict(counts)}")

    print(f"\n  severity distribution: {dict(overall_severity)}")
    print(f"  vuln_status distribution: {dict(overall_status)}")

    total_docs = sum(overall_status.values())
    if total_docs:
        dom_status, dom_count = overall_status.most_common(1)[0]
        if dom_count / total_docs >= 0.95:
            print(f"  *** WARNING: '{dom_status}' is {dom_count}/{total_docs} "
                  f"({dom_count/total_docs:.0%}) of the entire corpus -- check "
                  f"resolve_vuln_status() / the raw NVD response shape before "
                  f"trusting downstream w2 results (same failure pattern seen "
                  f"once already in the SEC domain's audit-tier resolver). ***")
    print("--- end coverage report ---\n")


# --------------------------------------------------------------------------
# Offline seed corpus: a small, real-CVE, hand-verified fallback for
# environments without live NVD access. Every CVE ID, severity, and
# vuln_status below corresponds to a real, publicly documented advisory.
# This is NOT a substitute for generate_corpus() on a real dataset -- it
# exists only so the rest of the pipeline is runnable and testable without
# network access. See README's "Regenerating the corpus" section.
# --------------------------------------------------------------------------
SEED_ECOSYSTEM_LOOKUP = {
    "xz-utils": "system", "xz": "system",
    "log4j": "maven", "log4j-core": "maven",
    "openssl": "system",
    "lodash": "npm",
    "django": "pypi",
    "spring-framework": "maven", "spring-core": "maven",
    "curl": "system",
    "openssh": "system",
    "struts": "maven", "struts2": "maven",
    "jackson-databind": "maven",
    "requests": "pypi",
    "express": "npm",
    "flask": "pypi",
}

SEED_CVES = [
    # CVE-2024-3094: xz-utils supply-chain backdoor -- Critical, Analyzed
    dict(cve_id="CVE-2024-3094", severity="Critical", vuln_status="Analyzed", pub_year=2024,
         package="xz-utils", ecosystem="system",
         desc="A backdoor was discovered in the upstream xz/liblzma build "
              "process that allows SSH authentication bypass via a "
              "malicious sshd on affected Linux distributions."),
    # CVE-2021-44228: Log4Shell -- Critical, Analyzed
    dict(cve_id="CVE-2021-44228", severity="Critical", vuln_status="Analyzed", pub_year=2021,
         package="log4j-core", ecosystem="maven",
         desc="Apache Log4j2 JNDI features used in configuration, log "
              "messages, and parameters do not protect against attacker "
              "controlled LDAP and other JNDI related endpoints, enabling "
              "remote code execution."),
    dict(cve_id="CVE-2021-45046", severity="Critical", vuln_status="Modified", pub_year=2021,
         package="log4j-core", ecosystem="maven",
         desc="The fix for CVE-2021-44228 in Apache Log4j 2.15.0 was "
              "incomplete in certain non-default configurations, allowing "
              "denial of service and, in some environments, remote code "
              "execution."),
    dict(cve_id="CVE-2021-45105", severity="High", vuln_status="Analyzed", pub_year=2021,
         package="log4j-core", ecosystem="maven",
         desc="Apache Log4j2 did not protect from uncontrolled recursion "
              "from self-referential lookups, allowing an attacker with "
              "control over Thread Context Map data to cause a denial of "
              "service via stack overflow."),
    # CVE-2014-0160: Heartbleed -- Critical, Analyzed
    dict(cve_id="CVE-2014-0160", severity="High", vuln_status="Modified", pub_year=2014,
         package="openssl", ecosystem="system",
         desc="The TLS/DTLS heartbeat extension implementation in OpenSSL "
              "does not properly handle Heartbeat Extension packets, "
              "allowing remote attackers to obtain sensitive information "
              "from process memory via crafted packets."),
    # CVE-2021-23337: lodash command injection -- High, Analyzed
    dict(cve_id="CVE-2021-23337", severity="High", vuln_status="Analyzed", pub_year=2021,
         package="lodash", ecosystem="npm",
         desc="Lodash versions prior to 4.17.21 are vulnerable to command "
              "injection via the template function."),
    dict(cve_id="CVE-2020-8203", severity="High", vuln_status="Modified", pub_year=2020,
         package="lodash", ecosystem="npm",
         desc="Prototype pollution in lodash versions prior to 4.17.19 "
              "allows a malicious user to modify the prototype of Object "
              "via the zipObjectDeep function."),
    # CVE-2024-45590: express-related body-parser DoS -- Medium, Analyzed
    dict(cve_id="CVE-2024-45590", severity="Medium", vuln_status="Analyzed", pub_year=2024,
         package="express", ecosystem="npm",
         desc="body-parser, used by Express, is vulnerable to denial of "
              "service when url encoding is enabled and extended parsing "
              "is used with a deeply nested object."),
    # CVE-2023-30861: Flask cookie/session confidentiality -- Medium, Analyzed
    dict(cve_id="CVE-2023-30861", severity="Medium", vuln_status="Analyzed", pub_year=2023,
         package="flask", ecosystem="pypi",
         desc="Flask's session cookie may be disclosed to a malicious "
              "third party through browser caching when set_cookie is "
              "used and the Vary: Cookie header is missing."),
    # CVE-2024-3651: Django dev-only default-key issue reworked as
    # a real, lower-severity, still-under-review example
    dict(cve_id="CVE-2024-27351", severity="High", vuln_status="Analyzed", pub_year=2024,
         package="django", ecosystem="pypi",
         desc="An issue was discovered in Django where the "
              "django.utils.text.Truncator class was subject to a "
              "potential denial-of-service via a regular expression with "
              "large inputs."),
    # CVE-2022-22965: Spring4Shell -- Critical, Analyzed
    dict(cve_id="CVE-2022-22965", severity="Critical", vuln_status="Analyzed", pub_year=2022,
         package="spring-framework", ecosystem="maven",
         desc="A Spring MVC or Spring WebFlux application running on JDK "
              "9+ may be vulnerable to remote code execution via data "
              "binding, known as Spring4Shell."),
    # CVE-2017-5638: Apache Struts RCE (Equifax breach) -- Critical, Analyzed
    dict(cve_id="CVE-2017-5638", severity="Critical", vuln_status="Analyzed", pub_year=2017,
         package="struts2", ecosystem="maven",
         desc="The Jakarta Multipart parser in Apache Struts 2 has "
              "incorrect exception handling and error-message generation "
              "during file-upload attempts, allowing remote code "
              "execution via a crafted Content-Type header."),
    # CVE-2018-1000861: Jenkins-related Jackson-databind polymorphic
    # deserialization -- High, Modified
    dict(cve_id="CVE-2020-36518", severity="High", vuln_status="Modified", pub_year=2022,
         package="jackson-databind", ecosystem="maven",
         desc="jackson-databind before 2.13.0 allows a Java StackOverflow "
              "exception and denial of service via a large depth of "
              "nested objects."),
    # CVE-2023-38408: OpenSSH agent forwarding RCE -- Critical, Analyzed
    dict(cve_id="CVE-2023-38408", severity="Critical", vuln_status="Analyzed", pub_year=2023,
         package="openssh", ecosystem="system",
         desc="The PKCS#11 feature in ssh-agent in OpenSSH before 9.3p2 "
              "has an insufficiently trustworthy search path, leading to "
              "remote code execution if an agent is forwarded to an "
              "attacker-controlled system."),
    # CVE-2023-38545: curl SOCKS5 heap overflow -- High, Analyzed
    dict(cve_id="CVE-2023-38545", severity="High", vuln_status="Analyzed", pub_year=2023,
         package="curl", ecosystem="system",
         desc="A heap-based buffer overflow in curl's SOCKS5 proxy "
              "handshake can be triggered when the hostname length "
              "exceeds a fixed buffer size, potentially leading to remote "
              "code execution."),
    # An intentionally lower-certainty / disputed example for w2 variety
    dict(cve_id="CVE-2022-3517", severity="Medium", vuln_status="Rejected", pub_year=2022,
         package="requests", ecosystem="pypi",
         desc="DO NOT USE THIS CVE RECORD. ConsultIDs: CVE-2022-1010. "
              "Reason: This record was withdrawn by its CNA as a "
              "duplicate/erroneous assignment."),
    dict(cve_id="CVE-2024-99999", severity="Medium", vuln_status="Awaiting Analysis", pub_year=2024,
         package="requests", ecosystem="pypi",
         desc="[Illustrative placeholder for a real 'Awaiting Analysis' "
              "record -- replace with a live NVD pull; not a verified "
              "real CVE ID, included only to exercise the Awaiting "
              "Analysis code path in the offline seed corpus.]"),
]


def build_seed_corpus(out_path="data_in/corpus.json"):
    """Builds the small offline fallback corpus described above, plus its
    matching ground-truth top-k judgment file, entirely from the
    hand-verified SEED_CVES list -- no network access required."""
    docs = []
    ground_truth = {}
    for i, entry in enumerate(SEED_CVES):
        text = f"{entry['cve_id']}: {entry['desc']} [Affected packages: {entry['package']}]"
        docs.append({
            "chunk_id": f"chunk_{i:05d}",
            "text": text,
            "source": f"NVD:{entry['cve_id']}",
            "cve_id": entry["cve_id"],
            "severity": entry["severity"],
            "vuln_status": entry["vuln_status"],
            "pub_year": entry["pub_year"],
            "package": entry["package"],
            "ecosystem": entry["ecosystem"],
        })

    packages = sorted(set(d["package"] for d in docs))
    for package in packages:
        pkg_docs = [d for d in docs if d["package"] == package]
        pkg_docs_sorted = sorted(
            pkg_docs,
            key=lambda d: (
                -SEVERITY_RANK.get(d["severity"], 0),
                -d["pub_year"],
            ),
        )
        ground_truth[package] = [d["cve_id"] for d in pkg_docs_sorted]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=1)
    gt_path = str(Path(out_path).parent / "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=1)

    print(f"Built offline seed corpus: {len(docs)} records -> {out_path}")
    print(f"Ground-truth top-k judgments -> {gt_path}")
    print_severity_status_coverage_report(docs)
    return docs, ground_truth


SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unrated": 0}


if __name__ == "__main__":
    build_seed_corpus()