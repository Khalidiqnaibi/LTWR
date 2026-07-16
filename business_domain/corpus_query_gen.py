"""
corpus_gen.py

An end-to-end pipeline that pulls real SEC filings from EDGAR, resolves structural 
auditor-of-record tiers to prevent LTR model feature variance drops, compiles a 
unified business-domain corpus, and generates a stratified evaluation query benchmark.
"""

import json
import re
import time
import random
import argparse
from pathlib import Path
import requests

# --- SEC Fair-Access Requirements ------------------------------------------
USER_AGENT = "LTWR Research (241154@ppu.edu.ps)"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY_SEC = 0.11  # Keeps requests comfortably below SEC's 10 req/sec limit

# --- SEC API Endpoints -----------------------------------------------------
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"
TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"

# --- Configurations & Mappings ---------------------------------------------
FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A"]
FILING_TYPE_NORMALIZED = {"DEF 14A": "DEF14A"}  # Normalizes variant names

MIN_FISCAL_YEAR = 2022

# Only 8-K is genuinely unaudited. 10-Q filings receive an auditor's
# limited review (SAS 100 / AS 4105), and DEF14A proxy statements
# reference audited financials -- both need per-filing resolution, not a
# blanket "Unaudited" tag, or audit_tier collapses into a restatement of
# filing_type (see business_domain/README.md's note on this).
AUDIT_TIER_BY_TYPE = {
    "10-K": None,      # Resolved dynamically per filing
    "10-Q": None,      # Resolved dynamically per filing
    "DEF14A": None,    # Resolved dynamically per filing
    "8-K": "Unaudited",
}

PCAOB_BIG4_FIRM_IDS = {
    "34": "Deloitte & Touche LLP",
    "42": "Ernst & Young LLP",
    "185": "KPMG LLP",
    "238": "PricewaterhouseCoopers LLP",
}

FACT_XBRL_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "headcount": ["NumberOfEmployees"],
}

# --- Request Wrapper --------------------------------------------------------
def _get(url, params=None):
    """Executes a rate-limited HTTP GET request compliant with SEC rules."""
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)
    return resp

# --- Normalization Helpers --------------------------------------------------
def _normalize_accn(accn: str) -> str:
    """Standardizes accession keys to prevent formatting discrepancies during checks."""
    return accn.replace("-", "").strip().lower()

# --- Metadata Pulling ------------------------------------------------------
def get_cik_for_ticker(ticker: str) -> str:
    """Looks up a ticker to find its unique Central Index Key (CIK)."""
    resp = _get(TICKER_LOOKUP_URL)
    table = resp.json()
    for entry in table.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"])
    raise ValueError(f"No CIK found for ticker {ticker}")


def get_submissions(cik: str) -> dict:
    """Retrieves a metadata list of all filings for a given CIK."""
    return _get(SUBMISSIONS_URL.format(cik=int(cik))).json()


def get_company_facts(cik: str) -> dict:
    """Retrieves historical XBRL facts. Falls back cleanly for smaller filers."""
    try:
        return _get(COMPANY_FACTS_URL.format(cik=int(cik))).json()
    except requests.HTTPError:
        return {}


def list_filings(submissions: dict, filing_types=FILING_TYPES, years=range(MIN_FISCAL_YEAR, 2027)):
    """Generates structured metadata dicts from a company's recent filings list."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    for form, date, accession, doc in zip(forms, dates, accessions, docs):
        if form not in filing_types:
            continue
        year = int(date[:4])
        if year not in years:
            continue
        yield {
            "form": form,
            "filing_date": date,
            "filing_year": year,
            "accession": accession,
            "primary_document": doc,
        }


def fetch_filing_text(cik: str, accession: str, doc: str) -> str:
    """Downloads filing document and strips XML/HTML markup into clean text."""
    accession_nodash = accession.replace("-", "")
    url = FILING_DOC_URL.format(cik=int(cik), accession_nodash=accession_nodash, doc=doc)
    resp = _get(url)
    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Advanced Auditor Resolution --------------------------------------------
def resolve_audit_tier(company_facts: dict, year: int, accession: str, filing_text: str = None) -> str:
    """
    Resolves the auditor tier dynamically. Utilizes sequential fallback:
    1. Check PCAOB Firm ID (accounting for leading-zero variations)
    2. Check Auditor Name
    3. Run text-scanning signature fallback (handles missing facts/string API blocks)
    """
    if year < MIN_FISCAL_YEAR:
        return "PreDisclosureRule"

    target_accn = _normalize_accn(accession)
    dei = company_facts.get("facts", {}).get("dei", {})

    # 1. Resolve via PCAOB Auditor Firm ID
    firm_id_entry = dei.get("AuditorFirmId") or dei.get("AuditorFirmID")
    if firm_id_entry:
        for unit_vals in firm_id_entry.get("units", {}).values():
            for v in unit_vals:
                if _normalize_accn(v.get("accn", "")) == target_accn:
                    val = str(v.get("val", "")).strip().lstrip("0")
                    return "Big4" if val in PCAOB_BIG4_FIRM_IDS else "OtherAudited"

    # 2. Resolve via Auditor Legal Name
    name_entry = dei.get("AuditorName")
    if name_entry:
        for unit_vals in name_entry.get("units", {}).values():
            for v in unit_vals:
                if _normalize_accn(v.get("accn", "")) == target_accn:
                    reported_name = str(v.get("val", "")).strip().lower()
                    is_big4 = any(b4.lower() in reported_name for b4 in PCAOB_BIG4_FIRM_IDS.values())
                    return "Big4" if is_big4 else "OtherAudited"

    # 3. Robust Text-Scanning Fallback (essential since String items are omitted from SEC facts)
    if filing_text:
        text_lower = filing_text.lower()
        if "deloitte" in text_lower:
            return "Big4"
        if "ernst & young" in text_lower or "ey llp" in text_lower or "ernst and young" in text_lower:
            return "Big4"
        if "kpmg" in text_lower:
            return "Big4"
        if "pricewaterhousecoopers" in text_lower or "pwc" in text_lower:
            return "Big4"

        # Check if there is any registered audit signature or opinion reference
        if "independent registered public accounting firm" in text_lower or "report of independent" in text_lower:
            return "OtherAudited"

    return "Unknown"

# --- Fact Extraction Functions ----------------------------------------------
def extract_text_facts(filing_text: str) -> dict:
    """Regex parsing fallback to pull margin and litigation records directly from filing text."""
    facts = {}
    m = re.search(r"operating margin[^%]{0,40}?(\d{1,2}(?:\.\d+)?)\s*%", filing_text, re.IGNORECASE)
    if m:
        facts["margin"] = float(m.group(1))

    m = re.search(r"(\d+)\s+(?:material\s+)?(?:pending\s+)?legal proceedings", filing_text, re.IGNORECASE)
    if m:
        facts["litigation"] = int(m.group(1))

    return facts


def extract_xbrl_fact(company_facts: dict, fact_key: str, accession: str):
    """Retrieves numeric metrics corresponding to the target accession from the XBRL tree."""
    tags = FACT_XBRL_TAGS.get(fact_key, [])
    facts_dict = company_facts.get("facts", {})
    target_accn = _normalize_accn(accession)

    best_val = None
    max_fy = -1

    for namespace in ["us-gaap", "dei"]:
        ns_data = facts_dict.get(namespace, {})
        for tag in tags:
            entry = ns_data.get(tag)
            if not entry:
                continue
            for unit_vals in entry.get("units", {}).values():
                for v in unit_vals:
                    if _normalize_accn(v.get("accn", "")) == target_accn:
                        fy = v.get("fy")
                        if fy is not None:
                            if fy > max_fy:
                                max_fy = fy
                                best_val = v.get("val")
                        elif best_val is None:
                            best_val = v.get("val")
    return best_val

# --- Document Building ------------------------------------------------------
def build_document_record(chunk_id, ticker, sector, filing, cik, filing_text, company_facts):
    """Compiles single-passage records enriched with metadata and extracted numeric features."""
    form = filing["form"]
    filing_type = FILING_TYPE_NORMALIZED.get(form, form)
    year = filing["filing_year"]
    accession = filing["accession"]

    # Retrieve audited state: handle statically where possible, or invoke resolver
    audit_tier = AUDIT_TIER_BY_TYPE.get(filing_type)
    if audit_tier is None:
        audit_tier = resolve_audit_tier(company_facts, year, accession, filing_text)

    text_facts = extract_text_facts(filing_text)
    xbrl_facts = {k: extract_xbrl_fact(company_facts, k, accession) for k in ("revenue", "debt", "headcount")}

    # Standardize output summary details
    fact_summary_parts = []
    for k, v in {**xbrl_facts, **text_facts}.items():
        if v is not None:
            fact_summary_parts.append(f"{k}={v}")
    fact_summary = "; ".join(fact_summary_parts)

    passage_text = (filing_text[:1500] + (f" [Extracted facts: {fact_summary}]" if fact_summary else "")).strip()

    return {
        "chunk_id": f"chunk_{chunk_id:05d}",
        "text": passage_text,
        "source": f"EDGAR:{ticker}:{filing_type}:{year}:{accession}",
        "filing_type": filing_type,
        "audit_tier": audit_tier,
        "filing_year": year,
        "ticker": ticker,
        "domain": sector,
    }

# --- Pipeline Routines ------------------------------------------------------
def generate_corpus(tickers_with_sectors, out_path="data_in/business_corpus.json",
                     years=range(MIN_FISCAL_YEAR, 2027), n_docs_target=900):
    """Pulls and builds the business-domain evaluation corpus."""
    docs = []
    chunk_id = 0

    for ticker, sector in tickers_with_sectors:
        print(f"Pulling filings for {ticker} ({sector})...")
        try:
            cik = get_cik_for_ticker(ticker)
            submissions = get_submissions(cik)
            company_facts = get_company_facts(cik)
        except Exception as e:
            print(f"Error fetching metadata for {ticker}: {e}")
            continue

        for filing in list_filings(submissions, years=years):
            try:
                filing_text = fetch_filing_text(cik, filing["accession"], filing["primary_document"])
            except requests.HTTPError:
                continue  # Skip unretrievable, paper, or restricted filings

            record = build_document_record(chunk_id, ticker, sector, filing, cik, filing_text, company_facts)
            docs.append(record)
            chunk_id += 1

            if len(docs) >= n_docs_target:
                break
        if len(docs) >= n_docs_target:
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=1)
    print(f"Successfully generated {len(docs)} corpus records -> {out_path}")
    return docs


def generate_queries(corpus_path="data_in/business_corpus.json",
                      out_path="data_in/business_queries.json",
                      n_per_dimension=50):
    """Generates the stratified query evaluation benchmark based on corpus tickers."""
    if not Path(corpus_path).exists():
        print(f"Error: Corpus file {corpus_path} does not exist. Cannot generate queries.")
        return []

    random.seed(7)
    corpus = json.load(open(corpus_path, encoding="utf-8"))
    tickers = sorted(set(d["ticker"] for d in corpus))

    facts = ["revenue", "debt", "litigation", "headcount", "margin"]

    query_templates = {
        "filing_type": "What does {ticker}'s official annual filing state about {fact}?",
        "audit_tier": "What is the audited, verified figure for {ticker}'s {fact}?",
        "recency": "What is {ticker}'s current, most recent {fact}?",
        "combined": "What is {ticker}'s {fact}?",
    }

    fact_names = {
        "revenue": "total revenue", 
        "debt": "long-term debt", 
        "litigation": "legal proceedings",
        "headcount": "employee headcount", 
        "margin": "operating margin",
    }

    queries = []
    qid = 1
    dims = list(query_templates.keys())
    
    for dim in dims:
        combos = [(t, f) for t in tickers for f in facts]
        random.shuffle(combos)
        for ticker, fact in combos[:n_per_dimension]:
            text = query_templates[dim].format(ticker=ticker, fact=fact_names[fact])
            queries.append({
                "id": qid,
                "query": text,
                "ticker": ticker,
                "fact": fact,
                "ablation_dimension": dim,
            })
            qid += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=1)
    print(f"Successfully generated {len(queries)} benchmark queries -> {out_path}")
    return queries


# --- CLI Entrance Point -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC Edgar Corpus & Stratified Query Generator")
    parser.add_argument("--corpus_out", type=str, default="data_in/business_corpus.json", help="Path to write the corpus JSON")
    parser.add_argument("--queries_out", type=str, default="data_in/business_queries.json", help="Path to write the benchmark queries")
    parser.add_argument("--target_docs", type=int, default=900, help="Target number of documents to collect")
    args = parser.parse_args([])

    # Domain companies for evaluation
    companies = [
        ("AAPL", "technology"), ("MSFT", "technology"), ("TGT", "retail"),
        ("JNJ", "healthcare"), ("PFE", "healthcare"), ("XOM", "energy"),
        ("CVX", "energy"), ("CAT", "industrials"), ("HON", "industrials"),
        ("JPM", "financials"), ("BAC", "financials"), ("WMT", "retail"),
        ("CRM", "technology"), ("NUE", "industrials"), ("UNH", "healthcare"),
    ]

    print("Step 1: Commencing Real SEC EDGAR Filing Retrieval...")
    generate_corpus(
        tickers_with_sectors=companies,
        out_path=args.corpus_out,
        years=range(MIN_FISCAL_YEAR, 2027),
        n_docs_target=args.target_docs
    )

    print("\nStep 2: Commencing Stratified Query Benchmark Generation...")
    generate_queries(
        corpus_path=args.corpus_out,
        out_path=args.queries_out,
        n_per_dimension=50
    )

    print("\nProcessing finished. Pipeline outputs are completely ready for LTWR training.")