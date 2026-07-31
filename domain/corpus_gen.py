"""
corpus_gen.py .. CVE/supply-chain-security corpus + ground-truth generator.
See module docstring history: this file replaced free-text keywordSearch-
against-CVE-prose with a CPE-Dictionary-first architecture (resolve the
real vendor:product CPE identifier for a package, then fetch CVEs that
NVD has structurally CPE-matched against it via virtualMatchString).
"""
import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

import requests

NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_CPE_URL = "https://services.nvd.nist.gov/rest/json/cpes/2.0"
MIN_PUB_YEAR = 2018

PACKAGE_CPE_HINTS = {
    # xz-utils: bare "xz" matched unrelated same-word products (GNU gzip's
    # zgrep, and an unrelated Go-language library also named "xz").
    # "tukaani" is the actual upstream project/vendor name for the C
    # xz-utils this package means .. cpe:2.3:a:tukaani:xz.
    "xz-utils": "tukaani xz",
    "log4j-core": "log4j",
    "spring-framework": "spring",
    "struts2": "struts2",
    "express": "express",
    "requests": "requests",
    # apache-http-server: bare "httpd" matched unrelated same-word products
    # (ACME Ultra Mini HTTPd, Jasper httpdx). Vendor-scoping targets
    # Apache's actual httpd .. cpe:2.3:a:apache:http_server.
    "apache-http-server": "apache httpd",
    # guava: bare "guava" pulled in an unrelated hit (Terracotta Quartz
    # Scheduler) via the fuzzy word-overlap matcher. Vendor-scoping to
    # cpe:2.3:a:google:guava avoids that.
    "guava": "google guava",
}

MAX_PAGES_PER_CPE = 10
RESULTS_PER_PAGE = 100

# Training-readiness thresholds
MIN_CVES_PER_PACKAGE = 2

# A SEPARATE, higher bar specifically for packages landing in the TEST split
MIN_CVES_PER_TEST_PACKAGE = 15

# Used only to size the min-power warning in generate_corpus()
QUERY_DIMENSIONS_PER_PACKAGE = 4

PLACEHOLDER_EMAIL = "you@example.com"
CONTACT_EMAIL = os.getenv("EMAIL",PLACEHOLDER_EMAIL)
HEADERS = {"User-Agent": f"LTWR-CVE-Research (research contact: {CONTACT_EMAIL})"}

# NVD rate-limits unauthenticated requests to ~5 requests / 30s 
NVD_RATE_LIMIT_SLEEP_SEC = 6.5


def _get(url, params=None, api_key=None, max_retries=3):
    if CONTACT_EMAIL == PLACEHOLDER_EMAIL:
        raise RuntimeError(
            "CONTACT_EMAIL is still the placeholder value edit it in .env to a real email address and re-run"
        )
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
            if resp.status_code == 429:  # rate limited .. back off and retry
                time.sleep(NVD_RATE_LIMIT_SLEEP_SEC * (attempt + 1))
                continue
            raise
    raise last_exc


def resolve_severity(cve_item: dict) -> str:
    """w1 resolution: prefers CVSS v3.1, falls back to v3.0, then v2."""
    metrics = cve_item.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if entries:
            sev = entries[0].get("cvssData", {}).get("baseSeverity") or entries[0].get("baseSeverity")
            if sev:
                return sev.title()
    v2_entries = metrics.get("cvssMetricV2")
    if v2_entries:
        sev = v2_entries[0].get("baseSeverity")
        if sev:
            return sev.title()
    return "Unrated"


def resolve_vuln_status(cve_item: dict) -> str:
    """w2 resolution: NVD's own vulnStatus field, used as-is."""
    return cve_item.get("vulnStatus", "Awaiting Analysis")


def extract_affected_packages(cve_item: dict):
    """Pulls all product names out of NVD's CPE-match configuration nodes,
    purely for DISPLAY in the passage text .. not used for relevance
    decisions """
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
    cve_id = cve_item.get("id", "")
    descriptions = cve_item.get("descriptions", [])
    desc_text = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
    packages = extract_affected_packages(cve_item)
    pkg_str = ", ".join(packages[:8]) if packages else "unspecified"
    return f"{cve_id}: {desc_text} [Affected packages: {pkg_str}]"


def build_document_record(chunk_id: int, cve_item: dict, ecosystem_lookup: dict, package: str) -> dict:
    """package: the canonical package name this CVE was fetched for ..
    known with CERTAINTY here (retrieval is by a resolved CPE vendor:
    product pair specific to this package)."""
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
    parts = cpe_name.split(":")
    if len(parts) < 5:
        return None
    return parts[2], parts[3], parts[4]


def _normalize(s: str) -> str:
    return s.replace("_", " ").replace("-", " ").lower().strip()


def _is_relevant(product: str, package: str, query_phrase: str) -> bool:
    product_norm = _normalize(product)
    product_words = set(product_norm.split())

    for candidate in (package, query_phrase):
        cand_norm = _normalize(candidate)
        if cand_norm == product_norm:
            return True
        cand_words = set(cand_norm.split())
        shared = {w for w in (product_words & cand_words) if len(w) > 3}
        if shared:
            return True
        if len(cand_norm) > 3 and (cand_norm.replace(" ", "") in product_norm.replace(" ", "")
                                    or product_norm.replace(" ", "") in cand_norm.replace(" ", "")):
            return True
    return False


# Vendor allowlist for packages
PACKAGE_VENDOR_ALLOWLIST = {
    "xz-utils": {"tukaani"},
    "apache-http-server": {"apache"},
    "guava": {"google"},
}


def resolve_cpe_matches(package: str, hint: str = None, api_key: str = None,
                         max_results: int = 100) -> list:
    """Queries NVD's CPE DICTIONARY for the product this package refers to,
    and returns a list of virtualMatchString patterns (one per distinct
    vendor:product pair found) to feed into fetch_cves_for_virtual_match().
    """
    query_phrase = hint or package
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

    allowed_vendors = PACKAGE_VENDOR_ALLOWLIST.get(package)
    relevant_pairs = [
        (vendor, product) for vendor, product in sorted(pairs)
        if _is_relevant(product, package, query_phrase)
        and (allowed_vendors is None or vendor.lower() in allowed_vendors)
    ]

    print(f"  CPE dictionary lookup for '{package}' (query: '{query_phrase}'): "
          f"{len(pairs)} distinct vendor:product pair(s) found, "
          f"{len(relevant_pairs)} kept as relevant: {relevant_pairs}")
    if not relevant_pairs and pairs:
        print(f"'{package}': none of the {len(pairs)} pairs found were kept as "
              f"relevant .. inspect this full list and add/adjust a PACKAGE_CPE_HINTS "
              f"entry if the right one is in here under a name the filter still "
              f"didn't recognize: {sorted(pairs)[:25]}")
    elif not pairs:
        print(f"'{package}': CPE dictionary lookup returned ZERO products for "
              f"query '{query_phrase}' .. try a different PACKAGE_CPE_HINTS phrase")

    return [f"cpe:2.3:a:{vendor}:{product}:*:*:*:*:*:*:*:*" for vendor, product in relevant_pairs]


def fetch_cves_for_virtual_match(virtual_match_string: str, api_key: str = None,
                                  results_per_page: int = RESULTS_PER_PAGE,
                                  max_pages: int = MAX_PAGES_PER_CPE) -> list:
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
            break
        time.sleep(NVD_RATE_LIMIT_SLEEP_SEC)
    return all_vulns


# Manual, per-(package, cve_id) exclusions .. for cases where NVD's own
# CPE configuration for a CVE genuinely includes this package's
# vendor:product (so PACKAGE_VENDOR_ALLOWLIST correctly lets it through),
# but manual review of the CVE's own description confirmed it is about a
# different tool entirely. This is NOT a correction to our matching logic
# .. the match is real on NVD's side .. it's an explicit, documented
# override of NVD's CPE tagging for this one record, kept narrow and
# auditable rather than silently dropped.
#
# CVE-2022-1271 / xz-utils: NVD's CPE configuration for this CVE lists
# "debian-linux, gzip, jboss-data-grid, xz" as affected products (see the
# CVE's own [Affected packages: ...] text), so a tukaani/xz vendor match
# is real. But the CVE description is entirely about GNU gzip's zgrep
# utility .. xz-utils itself is not the vulnerable component. Likely
# NVD over-tagged this CPE configuration to cover a packaging bundle
# (zgrep ships alongside xz in some distributions) rather than the
# vulnerable code itself. Excluded from xz-utils's ground truth so a
# query for "vulnerabilities affecting xz-utils" doesn't surface a
# gzip bug as a top-k result.
MANUAL_CVE_EXCLUSIONS = {
    ("xz-utils", "CVE-2022-1271"),
}


def generate_corpus(
    packages: list,
    ecosystem_lookup: dict,
    out_path="data_in/corpus.json",
    api_key: str = None,
):
    """Live NVD pull: for each package, resolves its real CPE vendor:product
    identifier(s), fetches every CVE NVD has structurally CPE-matched
    against it, builds records, and writes the corpus + ground truth +
    train/test split."""
    docs = []
    ground_truth = {}
    chunk_id = 0
    n_manually_excluded = 0

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
                    continue
                seen_cve_ids.add(cve_id)

                if (package, cve_id) in MANUAL_CVE_EXCLUSIONS:
                    n_manually_excluded += 1
                    print(f"    Manually excluding {cve_id} for '{package}' "
                          f"(see MANUAL_CVE_EXCLUSIONS comment for reasoning).")
                    continue

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

        pkg_docs = [d for d in docs if d["package"] == package]
        pkg_docs_sorted = sorted(
            pkg_docs,
            key=lambda d: (-SEVERITY_RANK.get(d["severity"], 0), -d["pub_year"]),
        )
        ground_truth[package] = [d["cve_id"] for d in pkg_docs_sorted]

    if not docs:
        raise RuntimeError(
            "generate_corpus() produced ZERO documents across all "
            f"{len(packages)} packages .. refusing to write an empty "
            "corpus.json/ground_truth.json. Check network access to "
            "services.nvd.nist.gov and the per-package messages printed above."
        )

    usable_packages = {pkg for pkg, cves in ground_truth.items() if len(cves) >= MIN_CVES_PER_PACKAGE}
    dropped_packages = {pkg: len(cves) for pkg, cves in ground_truth.items() if pkg not in usable_packages}

    if dropped_packages:
        print(f"\n Dropping {len(dropped_packages)} package(s) with < "
              f"{MIN_CVES_PER_PACKAGE} ground-truth CVEs from the training-ready "
              f"output: {dropped_packages}")
        print("    Most likely cause is resolve_cpe_matches() not finding a "
              "matching CPE vendor:product pair. Check the 'CPE dictionary "
              "lookup' messages above and consider adjusting PACKAGE_CPE_HINTS "
              "before treating the drop as final.\n")

    docs = [d for d in docs if d["package"] in usable_packages]
    ground_truth = {pkg: cves for pkg, cves in ground_truth.items() if pkg in usable_packages}

    if not docs:
        raise RuntimeError(
            f"After filtering to packages with >= {MIN_CVES_PER_PACKAGE} "
            "ground-truth CVEs, ZERO packages remain .. refusing to write "
            "an empty corpus.json/ground_truth.json."
        )

    # Auto-generated, size-balanced, package-disjoint train/test split.
    ranked = sorted(ground_truth.items(), key=lambda kv: -len(kv[1]))
    train_packages, test_packages = [], []
    train_n, test_n = 0, 0
    for pkg, cves in ranked:
        if train_n <= test_n:
            train_packages.append(pkg)
            train_n += len(cves)
        else:
            test_packages.append(pkg)
            test_n += len(cves)

    train_test_split = {
        "train_packages": train_packages,
        "test_packages": test_packages,
        "train_cve_count": train_n,
        "test_cve_count": test_n,
        "min_cves_per_package_threshold": MIN_CVES_PER_PACKAGE,
        "dropped_packages": dropped_packages,
    }

    # Aggregate min-power warning (unchanged from prior version).
    n_test_queries = len(test_packages) * QUERY_DIMENSIONS_PER_PACKAGE
    MIN_RECOMMENDED_TEST_QUERIES = 30
    if n_test_queries < MIN_RECOMMENDED_TEST_QUERIES:
        print(f"\n*** WARNING: this split yields only {n_test_queries} test queries "
              f"({len(test_packages)} packages x {QUERY_DIMENSIONS_PER_PACKAGE} "
              f"dimensions), below the ~{MIN_RECOMMENDED_TEST_QUERIES} recommended "
              f"for adequate power at Holm-Bonferroni-corrected significance. "
              f"Add more packages to `packages` and re-run before treating "
              f"results from this split as conclusive. ***\n")

    thin_test_packages = {pkg: len(ground_truth[pkg]) for pkg in test_packages
                           if len(ground_truth[pkg]) < MIN_CVES_PER_TEST_PACKAGE}
    if thin_test_packages:
        print(f"\n*** WARNING: {len(thin_test_packages)} TEST package(s) have fewer "
              f"than {MIN_CVES_PER_TEST_PACKAGE} ground-truth CVEs: {thin_test_packages}. "
              f"These will produce a near-meaningless per-query metric distribution "
              f"individually, even though they still count toward the aggregate "
              f"test-query total above. Consider moving them to TRAIN (where a low "
              f"bar is fine) or dropping them, rather than leaving them in TEST as-is. ***\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=1)
    print(f"Successfully generated {len(docs)} corpus records -> {out_path}")
    if n_manually_excluded:
        print(f"  ({n_manually_excluded} record(s) manually excluded via "
              f"MANUAL_CVE_EXCLUSIONS .. see that dict for reasoning)")

    gt_path = str(Path(out_path).parent / "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=1)
    print(f"Ground-truth top-k judgments -> {gt_path}")

    split_path = str(Path(out_path).parent / "train_test_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(train_test_split, f, indent=1)
    print(f"Train/test package split -> {split_path}")
    print(f"  TRAIN_PACKAGES ({len(train_packages)} packages, {train_n} CVEs): {train_packages}")
    print(f"  TEST_PACKAGES  ({len(test_packages)} packages, {test_n} CVEs): {test_packages}")

    print_severity_status_coverage_report(docs)
    return docs, ground_truth, train_test_split


def print_severity_status_coverage_report(docs):
    from collections import Counter, defaultdict
    cross = defaultdict(Counter)
    overall_severity = Counter()
    overall_status = Counter()
    for d in docs:
        cross[d["severity"]][d["vuln_status"]] += 1
        overall_severity[d["severity"]] += 1
        overall_status[d["vuln_status"]] += 1

    print("\n..- vuln_status coverage by severity ..-")
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
                  f"({dom_count/total_docs:.0%}) of the entire corpus .. check "
                  f"resolve_vuln_status() before trusting downstream w2 results. ***")

    packages_seen = sorted(set(d["package"] for d in docs))
    print(f"\n  packages with >=1 kept record ({len(packages_seen)}): {packages_seen}")
    print("..- end coverage report ..-\n")


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
    "pyyaml": "pypi",
    "jinja2": "pypi",
    "axios": "npm",
    "guava": "maven",
    "netty": "maven",
    "tomcat": "system",
    "nginx": "system",
    "apache-http-server": "system",
    "postgresql": "system",
    "redis": "system",
}


if __name__ == "__main__":
    DEFAULT_PACKAGES = [
        "xz-utils", "log4j-core", "openssl", "lodash", "express", "flask",
        "django", "spring-framework", "struts2", "jackson-databind",
        "openssh", "curl", "requests",
        "pyyaml", "jinja2", "axios", "guava", "netty", "tomcat",
        "nginx", "apache-http-server", "postgresql", "redis",
    ]

    print("Attempting live NVD pull (services.nvd.nist.gov)...")
    generate_corpus(DEFAULT_PACKAGES, SEED_ECOSYSTEM_LOOKUP)
    print("Live NVD pull completed successfully.")