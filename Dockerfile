# EcoRoute gateway: docker build -t ecoroute . && docker run -p 8080:8080 ecoroute
# The router file (artifacts/router.pt) is mounted, baked in with --build-arg ROUTER_URL,
# or downloaded at start from $ECOROUTE_ROUTER_URL. See docs/deploy.md.
FROM python:3.11-slim

RUN useradd --create-home ecoroute
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface \
    ECOROUTE_HOST=0.0.0.0 PORT=8080

# CPU build of PyTorch: the gateway runs one small encoder and one PII model per request.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .
COPY configs ./configs
COPY scripts ./scripts
COPY integrations ./integrations
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p artifacts .cache && chown ecoroute artifacts .cache
USER ecoroute

# Fetch the prompt encoder and the PII model now, so the first request is not slow.
ARG PRELOAD=1
RUN if [ "$PRELOAD" = "1" ]; then python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
from ecoroute.confidentiality.ner import hf_tagger; hf_tagger()"; fi

ARG ROUTER_URL=
RUN if [ -n "$ROUTER_URL" ]; then \
    python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], 'artifacts/router.pt')" "$ROUTER_URL"; fi

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
ENTRYPOINT ["/entrypoint.sh"]
