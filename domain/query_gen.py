"""
query_gen.py -- generates the stratified query benchmark for the academic-
publishing LTWR study, mirroring business_domain/query_gen.py's 4-dimension
design.
"""
import json
import random
from pathlib import Path

random.seed(11)

FIELD_TOPICS = {
    "machine_learning": ["neural network generalization", "transformer architectures", "reinforcement learning"],
    "oncology": ["immunotherapy response", "tumor biomarkers", "chemotherapy resistance"],
    "climate_science": ["sea level rise projections", "carbon capture methods", "extreme weather attribution"],
    "psychology": ["cognitive bias replication", "behavioral intervention outcomes", "memory formation"],
    "genomics": ["CRISPR off-target effects", "gene expression regulation", "genome-wide association"],
    "materials_science": ["nanomaterial synthesis", "battery electrode materials", "superconductor properties"],
    "epidemiology": ["disease transmission modeling", "vaccine efficacy", "outbreak surveillance methods"],
    "economics": ["monetary policy effects", "labor market outcomes", "inflation forecasting"],
}

QUERY_TEMPLATES = {
    # peer_review dimension: phrasing emphasizes peer-reviewed/published framing
    "peer_review": "What does peer-reviewed research say about {topic}?",
    # retraction dimension: phrasing emphasizes reliability/validity framing
    "retraction": "What is the scientifically validated finding on {topic}?",
    # recency dimension: phrasing emphasizes current/latest framing
    "recency": "What is the most recent research on {topic}?",
    # combined: neutral phrasing, all three signals matter roughly equally
    "combined": "What does research show about {topic}?",
}


def generate_queries(out_path="data_in/academic_queries.json", n_per_dimension=50):
    all_topics = [(field, topic) for field, topics in FIELD_TOPICS.items() for topic in topics]

    queries = []
    qid = 1
    for dim, template in QUERY_TEMPLATES.items():
        combos = list(all_topics)
        random.shuffle(combos)
        # cycle through topics if n_per_dimension exceeds the topic pool
        picks = (combos * ((n_per_dimension // len(combos)) + 1))[:n_per_dimension]
        for field, topic in picks:
            queries.append({
                "id": qid,
                "query": template.format(topic=topic),
                "field": field,
                "topic": topic,
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
