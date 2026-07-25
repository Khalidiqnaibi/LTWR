import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_CPE_URL = "https://services.nvd.nist.gov/rest/json/cpes/2.0"
MIN_PUB_YEAR = 2018

PACKAGE_CPE_HINTS = {
    "xz-utils": "xz",
    "log4j-core": "log4j",
    "spring-framework": "spring framework",
    "struts2": "struts2",
    "express": "expressjs",
    "requests": "python requests",
}

MAX_PAGES_PER_CPE = 10
RESULTS_PER_PAGE = 100

CONTACT_EMAIL = "241154@ppu.edu.ps"
HEADERS = {"User-Agent": f"LTWR-CVE-Research (research contact: {CONTACT_EMAIL})"}

NVD_RATE_LIMIT_SLEEP_SEC = 6.5


def _get(url, params=None, api_key=None, max_retries=3):
    if CONTACT_EMAIL == "your_email@example.com": # Using a generic placeholder for demonstration
        print("WARNING: CONTACT_EMAIL is still the placeholder value. Please replace it with your real contact info.")
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
    """Pulls all (product) names out of NVD's CPE-match configuration
    nodes, purely for DISPLAY in the passage text (build_passage_text
    below) -- e.g. "this CVE also affects these bundling products." This
    is NOT used for relevance decisions anymore (see module docstring):
    which CVEs belong to which package is now determined upstream, by
    which resolved CPE pattern a CVE was fetched under, not by inspecting
    this list after the fact."""
    packages = set()
    for config in cve_item.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                criteria = cpe_match.get("criteria", "")
                parts = criteria.split(":")
                if len(parts) > 4:
                    packages.add(parts[4].replace("_", "-"))
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


def build_document_record(chunk_id: int, cve_item: dict, ecosystem_lookup: dict, package: str) -> dict:
    """package: the canonical package name this CVE was fetched for --
    known with CERTAINTY here, not guessed. Retrieval is now by a resolved
    CPE vendor:product pair specific to this package (see
    resolve_cpe_matches / fetch_cves_for_virtual_match)."""
    cve_id = cve_item.get("id", "")
    pub_year = int(cve_item.get("published", "1970")[:4])
    ecosystem = ecosystem_lookup.get(package, "system")

    return {
        "chunk_id": f"chunk_{chunk_id:05d}",
        "text": build_passage_text(cve_item),
        "source": f"NVD:{cve_id}",
        "cve_id": cve_id,
        "severity": resolve_severity(cve_item),
        "vuln_status": resolve_vuln_status(cve_item),
        "pub_year": pub_year,
        "package": package,
        "ecosystem": ecosystem,
    }


def _split_cpe(cpe_name: str):
    """cpe:2.3:{part}:{vendor}:{product}:{version}:... -> (part, vendor,
    product). Returns None if the string doesn't look like a well-formed
    CPE 2.3 URI."""
    parts = cpe_name.split(":")
    if len(parts) < 5:
        return None
    return parts[2], parts[3], parts[4]


def resolve_cpe_matches(package: str, hint: str = None, api_key: str = None,
                         max_results: int = 100) -> list:
    """Queries NVD's CPE DICTIONARY (services.nvd.nist.gov/rest/json/cpes/2.0)
    -- a curated space of official product identifiers -- for the product
    this package refers to, and returns a list of virtualMatchString
    patterns (wildcarded CPE 2.3 URIs, one per distinct vendor:product pair
    found, covering ALL versions of each) to feed into
    fetch_cves_for_virtual_match().
    """
    query_phrase = hint or package
    # Changed 'keywordSearchPhrase' to 'keywordSearch' as per NVD API 2.0 documentation for CPE search
    params = {"keywordSearch": query_phrase, "resultsPerPage": max_results}
    resp = _get(NVD_CPE_URL, params=params, api_key=api_key)
    data = resp.json()
    products = data.get("products", [])

    pairs = set()
    for entry in products:
        cpe_name = entry.get("cpe", {}).get("cpeName", "")
        parsed = _split_cpe(cpe_name)
        if not parsed:
            continue
        part, vendor, product = parsed
        if part != "a":  # Applications only; skip Operating Systems ("o") / Hardware ("h")
            continue
        pairs.add((vendor, product))

    canonical_norms = {
        package.lower(), package.lower().replace("-", "_"),
        query_phrase.lower(), query_phrase.lower().replace("-", "_"),
    }
    relevant_pairs = []
    for vendor, product in sorted(pairs):
        product_hyphen = product.replace("_", "-").lower()
        product_underscore = product.replace("-", "_").lower()
        if (product_hyphen in canonical_norms or product_underscore in canonical_norms
                or any(len(c) > 3 and c in product_hyphen for c in canonical_norms)):
            relevant_pairs.append((vendor, product))

    print(f"  CPE dictionary lookup for '{package}' (query: '{query_phrase}'): "
          f"{len(pairs)} distinct vendor:product pair(s) found, "
          f"{len(relevant_pairs)} kept as relevant: {relevant_pairs}")
    if not relevant_pairs and pairs:
        print(f"  *** '{package}': none of the {len(pairs)} pairs found were kept as "
              f"relevant -- inspect this full list and add an entry to "
              f"PACKAGE_CPE_HINTS if the right one is in here under a name the "
              f"filter didn't recognize: {sorted(pairs)[:25]} ***")
    elif not pairs:
        print(f"  *** '{package}': CPE dictionary lookup returned ZERO products for "
              f"query '{query_phrase}' -- try a different PACKAGE_CPE_HINTS phrase. ***")

    return [f"cpe:2.3:a:{vendor}:{product}:*:*:*:*:*:*:*:*" for vendor, product in relevant_pairs]


def fetch_cves_for_virtual_match(virtual_match_string: str, api_key: str = None,
                                  results_per_page: int = RESULTS_PER_PAGE,
                                  max_pages: int = MAX_PAGES_PER_CPE) -> list:
    """Fetches every CVE NVD says structurally matches this CPE pattern --
    a server-side CPE match, not a free-text guess. Every record returned
    here is relevant BY CONSTRUCTION."""
    all_vulns = []
    for page_num in range(max_pages):
        params = {
            "virtualMatchString": virtual_match_string,
            "resultsPerPage": results_per_page,
            "startIndex": page_num * results_per_page,
        }
        resp = _get(NVD_CVE_URL, params=params, api_key=api_key)
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        all_vulns.extend(vulns)
        if len(vulns) < results_per_page:
            break  # got fewer than a full page -- no more results for this CPE pattern
        time.sleep(NVD_RATE_LIMIT_SLEEP_SEC)
    return all_vulns


def generate_corpus(
    packages: list,
    ecosystem_lookup: dict,
    out_path="data_in/corpus.json",
    api_key: str = None,
):
    """Live NVD pull: for each package, resolves its real CPE vendor:product
    identifier(s) via the CPE dictionary, fetches every CVE NVD has
    structurally CPE-matched against those identifiers, builds CveDocument
    records (package assignment is now CERTAIN, not guessed), and writes
    the corpus JSON.

    Also returns a {package: [cve_id, ...]} ground-truth map -- built
    directly from NVD's own CPE linkage and severity/status fields, not
    authored by this paper's team. See build_relevance_judgments() usage
    below for how this becomes the per-query top-k judgment file
    train_ltwr_cve.py trains against.
    """
    docs = []
    ground_truth = {}  # package -> [cve_id, ...], in NVD's own relevance order (severity then recency)
    chunk_id = 0

    for package in packages:
        hint = PACKAGE_CPE_HINTS.get(package)
        print(f"Resolving CPE identifier for package: {package} ...")
        virtual_match_strings = resolve_cpe_matches(package, hint=hint, api_key=api_key)
        time.sleep(NVD_RATE_LIMIT_SLEEP_SEC)

        if not virtual_match_strings:
            print(f"  Skipping '{package}': no relevant CPE product resolved (see message above).")
            ground_truth[package] = []
            continue

        seen_cve_ids = set()
        n_raw_total = 0
        n_dropped_old = 0

        for vm_string in virtual_match_strings:
            print(f"  Fetching CVEs for {vm_string} ...")
            try:
                vulns = fetch_cves_for_virtual_match(vm_string, api_key=api_key)
            except requests.HTTPError as e:
                print(f"  Error fetching {vm_string}: {e}")
                continue
            n_raw_total += len(vulns)

            for vuln in vulns:
                cve_item = vuln.get("cve", {})
                cve_id = cve_item.get("id", "")
                if not cve_id or cve_id in seen_cve_ids:
                    continue  # same CVE reachable via >1 resolved vendor:product pair
                seen_cve_ids.add(cve_id)

                record = build_document_record(chunk_id, cve_item, ecosystem_lookup, package)
                if record["pub_year"] < MIN_PUB_YEAR:
                    n_dropped_old += 1
                    continue

                docs.append(record)
                chunk_id += 1

            time.sleep(NVD_RATE_LIMIT_SLEEP_SEC)

        n_kept = len([d for d in docs if d["package"] == package])
        print(f"  {package}: {n_raw_total} raw CPE-matched CVEs across "
              f"{len(virtual_match_strings)} vendor:product pair(s) -> {n_kept} kept "
              f"({n_dropped_old} dropped as pre-{MIN_PUB_YEAR})")

        # NVD's own implied relevance order for ground truth: severity
        # first (Critical > High > Medium > Low), then recency within a
        # tier. This is NOT an invented ranking -- both fields are NVD's
        # own assigned data; imposing this order for judgment purposes
        # simply operationalizes "which of NVD's own matched CVEs would a
        # security analyst want surfaced first," using only NVD-assigned
        # facts, not this paper's trust-weighting scheme itself.
        pkg_docs = [d for d in docs if d["package"] == package]
        pkg_docs_sorted = sorted(
            pkg_docs,
            key=lambda d: (-SEVERITY_RANK.get(d["severity"], 0), -d["pub_year"]),
        )
        ground_truth[package] = [d["cve_id"] for d in pkg_docs_sorted]

    if not docs:
        raise RuntimeError(
            "generate_corpus() produced ZERO documents across all "
            f"{len(packages)} packages -- refusing to write an empty "
            "corpus.json/ground_truth.json. Check network access to "
            "services.nvd.nist.gov and the per-package messages printed above."
        )

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

    packages_seen = sorted(set(d["package"] for d in docs))
    print(f"\n  packages with >=1 kept record ({len(packages_seen)}): {packages_seen}")
    print("--- end coverage report ---\n")


SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unrated": 0}

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


if __name__ == "__main__":
    DEFAULT_PACKAGES = [
        "xz-utils", "log4j-core", "openssl", "lodash", "express", "flask",
        "django", "spring-framework", "struts2", "jackson-databind",
        "openssh", "curl", "requests",
    ]

    print("Attempting live NVD pull (services.nvd.nist.gov)...")
    generate_corpus(DEFAULT_PACKAGES, SEED_ECOSYSTEM_LOOKUP)
    print("Live NVD pull completed successfully.")