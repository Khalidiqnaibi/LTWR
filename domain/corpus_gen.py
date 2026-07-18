"""
corpus_gen.py -- pulls a real academic-publishing corpus for the LTWR study
from the Crossref REST API (api.crossref.org), fully open, no API key.

Three Crossref-native, structured fields map directly onto Eq. 2:

  w1 (peer-review status) <- Crossref "type" field
        journal-article       -> JournalArticle  (weight 1.0)
        proceedings-article   -> ProceedingsArticle (weight 0.8)
        posted-content        -> Preprint         (weight 0.4)

  w2 (retraction penalty) <- Crossref "relation.is-retracted-by" field,
        populated via publisher CrossMark metadata when a work has an
        active retraction notice. This is the SAME mechanism Crossref
        itself uses to surface retraction status -- no scraping, no text
        heuristics, exact-match on a structured relation type.

  w3 (recency decay)      <- Crossref "published"/"published-print"/
        "published-online" date field.

IMPORTANT -- avoiding the SEC-domain's w2-collapse bug: pulling works by
topic alone would likely yield near-zero retracted documents (retractions
are a small fraction of the literature), reproducing the same "feature has
no real variance" problem found in the SEC audit-tier pull. This script
therefore DELIBERATELY oversamples confirmed-retracted works: it first
queries Crossref's retraction notices directly (filter=update-type:retraction),
resolves each notice's "update-to" DOI back to the original work, and pulls
those originals explicitly -- rather than hoping retracted work is a large
enough fraction of general search results. This is a design REQUIREMENT for
this domain to be evaluable, not an optional embellishment; skipping it will
reproduce the exact w2-collapse issue found in the SEC/business domain.

For a broader, dedicated retraction dataset beyond what the works API's
`relation` field surfaces (which depends on publishers registering CrossMark
metadata, and so under-counts some retractions), Crossref also hosts the
full Retraction Watch database via a separate Labs endpoint that requires a
free registration token: https://www.crossref.org/blog/news-crossref-and-
retraction-watch/ -- swap fetch_retracted_dois() to that source if the
relation-field approach yields too few confirmed retractions for a given
field.
"""
import json
import re
import time
from pathlib import Path

import requests

# --- Crossref polite-pool etiquette ----------------------------------------
# Crossref has no hard auth requirement, but including a mailto param moves
# requests into the faster, more reliable "polite pool". Replace with your
# real contact info.
CONTACT_EMAIL = "your_name@your_institution.edu"
HEADERS = {"User-Agent": f"LTWR Research (mailto:{CONTACT_EMAIL})"}
REQUEST_DELAY_SEC = 0.15

WORKS_URL = "https://api.crossref.org/works"

# Research fields used to diversify the corpus (mirrors the SEC domain's
# multi-sector company list) -- each is a bibliographic search term.
FIELDS = {
    "machine_learning": "machine learning neural networks",
    "oncology": "cancer treatment oncology clinical",
    "climate_science": "climate change atmospheric warming",
    "psychology": "psychology cognitive behavioral study",
    "genomics": "genomics CRISPR gene editing",
    "materials_science": "materials science nanomaterials synthesis",
    "epidemiology": "epidemiology infectious disease outbreak",
    "economics": "economics macroeconomic policy analysis",
}

PUB_TYPE_MAP = {
    "journal-article": "JournalArticle",
    "proceedings-article": "ProceedingsArticle",
    "posted-content": "Preprint",
}

MIN_YEAR = 2015
MAX_YEAR = 2026


def _get(url, params=None):
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return resp


def _clean_abstract(raw_abstract: str) -> str:
    """Crossref abstracts are JATS-XML-tagged; strip tags to plain text."""
    if not raw_abstract:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_abstract)
    return re.sub(r"\s+", " ", text).strip()


def _extract_year(work: dict) -> int:
    for date_field in ("published", "published-print", "published-online", "issued"):
        entry = work.get(date_field)
        if entry and entry.get("date-parts") and entry["date-parts"][0]:
            year = entry["date-parts"][0][0]
            if year:
                return int(year)
    return MIN_YEAR  # fallback, should be rare


def _extract_pub_type(work: dict) -> str:
    return PUB_TYPE_MAP.get(work.get("type"), "JournalArticle")


def _is_retracted(work: dict) -> bool:
    """Structured retraction check: presence of relation.is-retracted-by,
    populated by publisher CrossMark metadata."""
    relations = work.get("relation", {})
    return bool(relations.get("is-retracted-by"))


def build_text(work: dict) -> str:
    title = " ".join(work.get("title", []) or [])
    abstract = _clean_abstract(work.get("abstract", ""))
    container = " ".join(work.get("container-title", []) or [])
    parts = [p for p in [title, abstract] if p]
    text = ". ".join(parts) if parts else title
    if container:
        text += f" [Published in: {container}]"
    return text.strip()


def work_to_record(work: dict, chunk_id: int, field: str, retracted_override: bool = None) -> dict:
    doi = work.get("DOI", "")
    pub_type = _extract_pub_type(work)
    year = _extract_year(work)
    retracted = retracted_override if retracted_override is not None else _is_retracted(work)
    venue = (work.get("container-title") or work.get("publisher", [""]))[0] if work.get("container-title") else work.get("publisher", "")

    return {
        "chunk_id": f"chunk_{chunk_id:05d}",
        "text": build_text(work),
        "source": f"Crossref:{doi}",
        "doi": doi,
        "pub_type": pub_type,
        "retracted": retracted,
        "pub_year": year,
        "venue": venue,
        "field": field,
    }


def fetch_field_works(field_key: str, query: str, rows: int = 40) -> list:
    """Pulls a mix of journal articles, preprints, and proceedings papers
    for one research field."""
    works = []
    for crossref_type, target_rows in [("journal-article", rows), ("posted-content", rows // 2),
                                        ("proceedings-article", rows // 3)]:
        params = {
            "query.bibliographic": query,
            "filter": f"type:{crossref_type},from-pub-date:{MIN_YEAR}-01-01,until-pub-date:{MAX_YEAR}-12-31",
            "rows": target_rows,
            "select": "DOI,title,abstract,type,published,published-print,published-online,"
                      "issued,container-title,publisher,relation",
        }
        resp = _get(WORKS_URL, params=params)
        items = resp.json().get("message", {}).get("items", [])
        works.extend(items)
    return works


def fetch_confirmed_retracted_works(query: str, n_notices: int = 15) -> list:
    """Queries Crossref retraction NOTICES directly (update-type:retraction),
    resolves each one's update-to DOI back to the original retracted work,
    and fetches that original work. This is what guarantees w2 has real
    variance -- see module docstring."""
    params = {
        "query.bibliographic": query,
        "filter": "update-type:retraction",
        "rows": n_notices,
        "select": "DOI,update-to",
    }
    resp = _get(WORKS_URL, params=params)
    notices = resp.json().get("message", {}).get("items", [])

    original_works = []
    for notice in notices:
        for update in notice.get("update-to", []):
            if update.get("type") == "original-form" or update.get("label", "").lower() == "retraction":
                original_doi = update.get("DOI")
                if not original_doi:
                    continue
                try:
                    orig_resp = _get(f"{WORKS_URL}/{original_doi}")
                    original_works.append(orig_resp.json().get("message", {}))
                except requests.HTTPError:
                    continue
    return original_works


def generate_corpus(out_path="data_in/academic_corpus.json", n_per_field=40,
                     n_retracted_per_field=6):
    docs = []
    chunk_id = 0

    for field_key, query in FIELDS.items():
        print(f"Pulling field: {field_key} ...")

        general_works = fetch_field_works(field_key, query, rows=n_per_field)
        for work in general_works:
            docs.append(work_to_record(work, chunk_id, field_key))
            chunk_id += 1

        retracted_works = fetch_confirmed_retracted_works(query, n_notices=n_retracted_per_field)
        for work in retracted_works:
            if not work.get("DOI"):
                continue
            docs.append(work_to_record(work, chunk_id, field_key, retracted_override=True))
            chunk_id += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(docs, f, indent=1)
    print(f"Pulled {len(docs)} real Crossref works -> {out_path}")
    return docs


if __name__ == "__main__":
    generate_corpus()