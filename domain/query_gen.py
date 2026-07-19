"""
query_gen.py -- generates the stratified query benchmark for the
CVE/supply-chain-security LTWR study, mirroring business_domain/query_gen.py
and domain/query_gen.py's 4-dimension design.

KEY DIFFERENCE FROM THE EARLIER DOMAINS: every query here is scoped to a
single package (e.g. "log4j-core"), because that is exactly the unit
corpus_gen.py's ground-truth judgment file (ground_truth.json) is
keyed on. This is deliberate -- it's what makes real top-k evaluation
possible at all: "what vulnerabilities affect log4j-core" has a genuine,
externally-curated answer (NVD's own CPE-matched CVE list for that
package); "what are the most severe vulnerabilities in general" would
not, and generating queries at that granularity would silently drop back
into needing self-authored judgments.
"""
import json
from pathlib import Path

QUERY_TEMPLATES = {
    # severity dimension: phrasing emphasizes criticality/impact framing
    "severity": "What are the most severe known vulnerabilities in {package}?",
    # vuln_status dimension: phrasing emphasizes confirmed/reviewed framing
    "vuln_status": "What are the confirmed, analyst-reviewed vulnerabilities in {package}?",
    # recency dimension: phrasing emphasizes current/latest framing
    "recency": "What are the most recently disclosed vulnerabilities in {package}?",
    # combined: neutral phrasing, all three signals matter roughly equally
    "combined": "What vulnerabilities affect {package}?",
}


def generate_queries(packages, out_path="data_in/queries.json"):
    """One query per (package, dimension) combination -- exhaustive rather
    than sampled, since the package list here is intentionally small
    (unlike the academic/SEC domains' larger topic pools) and every
    package needs at least one query per dimension for a clean per-
    dimension breakdown (Table 4 analogue)."""
    queries = []
    qid = 1
    for package in packages:
        for dim, template in QUERY_TEMPLATES.items():
            queries.append({
                "id": qid,
                "query": template.format(package=package),
                "package": package,
                "ablation_dimension": dim,
            })
            qid += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(queries, f, indent=1)
    print(f"Generated {len(queries)} queries -> {out_path}")
    return queries


if __name__ == "__main__":
    # Default package list matches corpus_gen.py's SEED_CVES packages.
    packages = [
        "xz-utils", "log4j-core", "openssl", "lodash", "express",
        "flask", "django", "spring-framework", "struts2",
        "jackson-databind", "openssh", "curl", "requests",
    ]
    generate_queries(packages)