# EcoRoute

An explainable, eco-aware LLM router. For every prompt it picks the cheapest, lowest-energy model that can handle it, treats confidentiality as a hard rule, and explains each decision.

Design docs: [`docs/brainstorm.md`](docs/brainstorm.md), [`docs/model-research.md`](docs/model-research.md), [`docs/zero-cost-plan.md`](docs/zero-cost-plan.md).

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Run everything on Lightning AI from your machine

`scripts/lightning_run.py` does the two steps below in one go. It ships the current commit to a
Studio (no GitHub token needed), builds the data, trains the baselines on a GPU, downloads the
reports to `reports/lightning/` and always stops the Studio at the end.

```bash
pip install lightning-sdk
export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...   # Lightning account settings, Keys
python scripts/lightning_run.py --teamspace <your-teamspace> --machine T4
```

It refuses to start below `--min-credits` (default 3) and stops the Studio after
`--max-hours` (default 2). A full run on a T4 takes about 11 minutes and about 0.2 credits.
Add `--skip-build` to reuse the data already in the Studio, `--pytest` to run the tests first,
and `--train-args "--seeds 3"` to pass options through to `train_baselines.py`.

## Build the training data (Lightning AI Studio)

1. Create a Studio (CPU is enough for this step; no GPU credits needed).
2. In its terminal:
   ```bash
   git clone https://github.com/waliddoesthis/ecoroute && cd ecoroute
   git checkout design/brainstorm
   pip install -e ".[dev]"
   python scripts/build_dataset.py --out data/processed
   ```
3. It downloads SPROUT, RouterBench and RouterArena (~2 GB) and writes
   `data/processed/outcomes.parquet` and `prompts.parquet`, then prints a summary.

Since the repo is private, step 2 needs a GitHub token or the Studio's GitHub integration to clone.

## Train the baseline predictors (Lightning AI, GPU)

After the data build above, on an L4 or T4 Studio:

```bash
python scripts/train_baselines.py --source routerbench
python scripts/train_baselines.py --source sprout
```

It embeds every prompt once (cached next to the data), trains the model-mean, kNN,
matrix-factorization and IRT predictors, and writes `reports/baselines_<source>.json` with
calibration, AUC and the cost-vs-accuracy routing curve. SPROUT has token counts but no
cost, so its cost is estimated from the price table its authors published
(`src/ecoroute/data/prices.py`).

## Confidentiality check

Every prompt is classified before routing as public, internal, confidential or restricted.
The strictest layer wins, and only models whose `clearance` in `configs/models.yaml` covers
the level are allowed.

```python
from ecoroute.confidentiality import Detector, allowed_models

result = Detector().classify(prompt, {"sensitivity_label": "Confidential"})
result.level      # e.g. Level.RESTRICTED
result.reasons()  # e.g. ["rules: aws_access_key (restricted)"]; matched values are never echoed
```

Layers so far: caller policy (labels such as Purview/DLP, or a minimum level) and
deterministic rules (secrets, cards with Luhn, IBANs with mod-97, SSNs, emails, phones, IPs,
high-entropy strings, and a company dictionary). `redact()` / `restore()` mask sensitive
values so a cheaper external model can be used when the task doesn't need them.
`tests/test_confidentiality.py` holds the prompts that must never reach an uncleared model.

## Train and try the deployable router

```bash
python scripts/train_router.py --data data/processed --out artifacts/router.pt
python scripts/route_demo.py --router artifacts/router.pt
```

The router is trained on SPROUT and scores the catalog models in `configs/models.yaml`
through their `anchor` (a SPROUT model of similar tier plus a logit shift). Shifts marked
`source: prior` only encode tier order; fit real ones with `CatalogPredictor.fit_shift()`
on a few hundred graded answers per model.
