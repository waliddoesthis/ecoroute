# EcoRoute

An explainable, eco-aware LLM router. For every prompt it picks the cheapest, lowest-energy model that can handle it, treats confidentiality as a hard rule, and explains each decision.

Design docs: [`docs/brainstorm.md`](docs/brainstorm.md), [`docs/model-research.md`](docs/model-research.md), [`docs/zero-cost-plan.md`](docs/zero-cost-plan.md).

## Setup

```bash
pip install -e ".[dev]"
pytest
```

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
prices yet, so its report has quality metrics only.
