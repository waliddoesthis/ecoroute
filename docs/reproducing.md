# Reproducing the results

Every number in the README and the [technical report](report.md) comes from a script in
`scripts/`. This page lists the commands, what each one measures, and how to run them on
a free GPU.

## 1. Data

All data is open. No paid grading is used.

| Dataset | Use |
|---|---|
| [SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT) | About 44k prompts graded on 13 models: training and the main test set |
| [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench) | 1,140 Python tasks with pass/fail for 11 of the same models |
| [RouterArena](https://huggingface.co/datasets/RouteWorks/RouterArena) | Difficulty labels, and a held-out difficulty check |
| [RouterBench](https://huggingface.co/datasets/withmartian/routerbench) | Baseline comparison |
| [ai4privacy pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | Privacy-filter recall |

```bash
python scripts/build_dataset.py --out data/processed     # about 2 GB download
```

Splits are fixed by a hash of each prompt (train 80%, validation 10%, test 10%; see
`assign_split` in `src/ecoroute/data/schema.py`). Every model's answer to a prompt is in
the same split.

## 2. Train and evaluate the router

```bash
python scripts/train_router.py --data data/processed --out artifacts/router.pt
python scripts/eval_router.py --router artifacts/router.pt --data data/processed
```

`train_router.py` takes four steps:

1. It fits the decision graph on the train split.
2. It calibrates it on the second half of validation.
3. On the first half, it chooses each profile's threshold and fallback margin.
4. It prints test results.

`eval_router.py` evaluates the stored thresholds once on the SPROUT test split. It writes
`reports/eval_<router>_sprout.json`, which records:

- the test-set size;
- the git commit;
- the reference model;
- the pricing source;
- the bootstrap method.

The README table is that report for router v14.

The router file stores the predictor, encoder name, thresholds and margins.
`--predictor ensemble` trains the kNN, MF, IRT and MLP ensemble instead, for comparison.

## 3. All measurement scripts

| Script | What it measures |
|---|---|
| `scripts/eval_router.py` | The results table: accuracy change, 90% CI and cost saved per profile |
| `scripts/train_router.py` | Training, threshold choice, and coding-prompt results |
| `scripts/check_routing.py` | Decisions on held-out RouterArena prompts by difficulty, with cost and energy per request on the deployed catalog |
| `scripts/eval_confidentiality.py` | Privacy filter detection rate, false alarms and speed |
| `scripts/bench_route.py` | Routing time per stage |
| `scripts/e2e_check.py` | The gateway end to end: privacy, refusals, profiles, streaming |
| `scripts/train_baselines.py` | Baseline predictors on SPROUT and RouterBench |
| `scripts/sweep_graph.py` | The graph's settings, chosen on validation only |
| `scripts/impact.py` | Scales the measured savings to company volumes ([impact.md](impact.md)) |

## 4. Hardware

All published runs used one NVIDIA T4 (16 GB) on Lightning AI's free tier. Training and
evaluation fit in its memory. The data needs about 2 GB of disk, plus a cached embedding
file for each encoder.

```bash
pip install lightning-sdk
export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...
python scripts/lightning_run.py --teamspace <your-teamspace> --machine T4 --pytest
```

`lightning_run.py` takes these steps:

1. It ships the current commit to a Studio.
2. It builds the data and trains.
3. It downloads the reports to `reports/lightning/`.
4. It always stops the Studio at the end.

It refuses to start below `--min-credits` (default 3) and stops after `--max-hours`
(default 2). Inside a Studio you can also run the commands above directly.

## 5. Adding a deployed model

`configs/models.yaml` holds each model's price, energy, clearance and `anchor`. The anchor
is a graded model of a similar tier plus a logit shift, so a new model needs no
retraining. Fit the shift with `CatalogPredictor.fit_shift()`, using a few hundred graded
answers per model.
