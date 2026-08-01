# Dockerfile for LTWR (Learned Trust-Weighted Retrieval)
# Place this file at the ROOT of the LTWR repo, next to requirements.txt.
# Run fetch_data.sh first — it writes data_in/* AND this repo's .env
# before you build, and both need to be present in the build context.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
# libgomp1 is required at runtime by lightgbm's compiled extension

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model BEFORE copying repo code, since it doesn't depend
# on any of it. This way editing a .py file or regenerating .env no longer
# invalidates this layer and forces a ~5min re-download every rebuild.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

VOLUME ["/app/eval_results"]

# main.py is the repo's documented entrypoint (see README Quickstart) and
# runs from /app, so "from infra..." resolves correctly without needing a
# PYTHONPATH workaround.
ENTRYPOINT ["python", "main.py"]