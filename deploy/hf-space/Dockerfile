# Hugging Face Space for the shared EcoRoute API (Docker SDK, free CPU).
# Builds the gateway from GitHub and runs it in hosted mode: every caller pays with
# their own provider key, and only holders of an access key get in.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ARG ECOROUTE_REF=main
RUN git clone --depth 1 --branch "$ECOROUTE_REF" https://github.com/waliddoesthis/ecoroute.git /src

RUN useradd --create-home --uid 1000 ecoroute
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface \
    ECOROUTE_HOST=0.0.0.0 PORT=8080 ECOROUTE_ARGS=--caller-keys
ARG ROUTER_URL=https://github.com/waliddoesthis/ecoroute/releases/latest/download/router.pt
ENV ECOROUTE_ROUTER_URL=$ROUTER_URL

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN cp -r /src/pyproject.toml /src/README.md /src/LICENSE /src/src /src/configs /src/scripts \
    /src/integrations . && pip install . && cp /src/docker/entrypoint.sh /entrypoint.sh \
    && chmod +x /entrypoint.sh && mkdir -p artifacts .cache && chown ecoroute artifacts .cache
USER ecoroute

# Models and router baked in, so a Space waking from sleep starts without downloads.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
from ecoroute.confidentiality.ner import hf_tagger; hf_tagger()"
RUN python -c "import os, urllib.request; \
urllib.request.urlretrieve(os.environ['ECOROUTE_ROUTER_URL'], 'artifacts/router.pt')"

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
