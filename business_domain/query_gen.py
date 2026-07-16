"""
generates the stratified query benchmark for LTWR
business-domain evaluation, mirroring the CTE-170 clinical benchmark's
4-dimension design (data_in/queries.json).
"""
import json
import random
from pathlib import Path

random.seed(7)

FACTS = ["revenue", "debt", "litigation", "headcount", "margin"]

QUERY_TEMPLATES = {
    # filing_type dimension: phrasing emphasizes "official/annual" framing,
    # designed so filing-type hierarchy should dominate the ranking
    "filing_type": "What does {ticker}'s official annual filing state about {fact}?",
    # audit_tier dimension: phrasing emphasizes verified/audited framing
    "audit_tier": "What is the audited, verified figure for {ticker}'s {fact}?",
    # recency dimension: phrasing emphasizes "current/latest" framing
    "recency": "What is {ticker}'s current, most recent {fact}?",
    # combined: neutral phrasing, all three signals matter roughly equally
    "combined": "What is {ticker}'s {fact}?",
}

FACT_NAME = {
    "revenue": "total revenue", "debt": "long-term debt", "litigation": "legal proceedings",
    "headcount": "employee headcount", "margin": "operating margin",
}


def generate_queries(corpus_path="data_in/business_corpus.json",
                      out_path="data_in/business_queries.json",
                      n_per_dimension=50):
    corpus = json.load(open(corpus_path))
    tickers = sorted(set(d["ticker"] for d in corpus))

    queries = []
    qid = 1
    dims = list(QUERY_TEMPLATES.keys())
    for dim in dims:
        combos = [(t, f) for t in tickers for f in FACTS]
        random.shuffle(combos)
        for ticker, fact in combos[:n_per_dimension]:
            text = QUERY_TEMPLATES[dim].format(ticker=ticker, fact=FACT_NAME[fact])
            queries.append({
                "id": qid,
                "query": text,
                "ticker": ticker,
                "fact": fact,
                "ablation_dimension": dim,
            })
            qid += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(queries, f, indent=1)
    print(f"Generated {len(queries)} queries -> {out_path}")
    return queries


if __name__ == "__main__":
    generate_queries()