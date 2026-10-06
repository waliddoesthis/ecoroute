# EcoRoute

An explainable, eco-aware LLM router behind an OpenAI-compatible gateway. For every prompt
it picks the cheapest, lowest-energy model that is predicted to answer it correctly. It
never sends confidential data to a model that isn't cleared for it, and it explains each
decision.

Apps keep their OpenAI client and change only `base_url` and `model="ecoroute/auto"`.

## Results

Router v14, measured on held-out SPROUT test prompts against always using GPT-4o (accuracy
0.845). Thresholds are chosen on validation prompts the router never trained or calibrated
on.

| Profile  | Accuracy target       | Points lost vs GPT-4o (90% CI) | Cost saved |
|----------|-----------------------|--------------------------------|------------|
| quality  | no loss               | -1.6 (-2.2 to -0.9), i.e. better | 34%      |
| balanced | at most 1 point lost  | 0.3 (-0.4 to 1.0)              | 70%        |
| eco      | at most 3 points lost | 1.6 (0.8 to 2.3)               | 75%        |

- **Privacy filter:** keeps 99% of texts with restricted data (keys, account and ID
  numbers, cards) and 97% of texts with personal data off external models, with 7.6% false
  alarms on ordinary prompts.
- **Coding prompts (BigCodeBench):** 48% to 60% cheaper than GPT-4o for 1.8 points lower
  accuracy. Telling hard coding tasks from easy ones is still the weakest part (AUC 0.65).
- **Speed:** a full routing decision (privacy check, embedding, prediction, explanation)
  takes about 72 ms on a T4 GPU, small next to the LLM call itself.

The method, every number and how it was measured are in
[`docs/how-it-decides.md`](docs/how-it-decides.md).

## How it decides

1. **Privacy filter (hard rule).** The prompt and the whole conversation are classified as
   public, internal, confidential or restricted. The check combines the caller's label,
   deterministic rules (secrets, cards with Luhn, IBANs, SSNs, labelled and long ID
   numbers, emails, phones) and a PII model
   ([piiranha](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)).
   Models whose `clearance` is below the level are removed. Nothing can override this.
2. **Decision graph.** The prompt is linked to skills (code, math, chat, ...), to a
   difficulty level, and to the most similar past prompts. Each path ends at a model with
   the share of such prompts that model answered correctly. The result is a calibrated
   P(correct) for every model, plus the heaviest paths that explain it.
3. **Cost and energy.** Among the allowed models with P(correct) at or above the profile's
   threshold, take the one with the lowest `cost + lambda * energy`. If none reaches the
   threshold, take the most likely model, or the cheapest one within the profile's small
   fallback margin of it.

Every response carries the decision in headers: `X-EcoRoute-Model`, `-Level`,
`-Difficulty`, `-Reason` and `-Time-Ms`.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

You need a trained router file (`router.pt`); see [Train a router](#train-a-router).
Encoder and PII models are downloaded from Hugging Face on first use.

**See decisions without calling any model:**

```bash
python scripts/route_demo.py --router artifacts/router.pt
```

**Run the gateway as a dry run** (no API keys, no cost; each answer names the model that
would have been used):

```bash
python scripts/serve.py --router artifacts/router.pt --dry-run
python scripts/e2e_check.py --router artifacts/router.pt   # end-to-end checks, PASS/FAIL
```

**Run it for real:**

```bash
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...   # only those you use
python scripts/serve.py --router artifacts/router.pt --port 8080 --pii-device 0
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
r = client.chat.completions.with_raw_response.create(
    model="ecoroute/auto",
    messages=[{"role": "user", "content": "Summarise this ..."}],
    extra_headers={"X-EcoRoute-Policy": "balanced"},  # eco / balanced / quality
)
r.headers["X-EcoRoute-Model"], r.headers["X-EcoRoute-Reason"]
```

- **Routable models:** only models with a confirmed `api_id` in `configs/models.yaml` and
  their provider's API key set. Local models go through Ollama or vLLM; see
  `configs/providers.yaml`.
- **Request headers:** `X-EcoRoute-Sensitivity` passes your own data label (e.g.
  `Confidential`), and `X-EcoRoute-Min-Level` sets a minimum level.
- **Naming a model:** asking for a specific model instead of `ecoroute/auto` skips the
  ranking but never the privacy check. An uncleared model gets a 403.
- **Explanation only:** `POST /route` returns the decision and explanation without
  calling any model.
- **Provider failures:** if the chosen provider fails, the gateway retries on the most
  likely other cleared model.
- **CPU only:** add `--no-pii-model` to keep only the rules.

## Train a router

The training data is free and open, and no paid grading is needed.

- **[SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT):** about 44k
  prompts graded on 13 models.
- **[RouterArena](https://huggingface.co/datasets/RouteWorks/RouterArena):** difficulty
  labels.
- **[BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench):** 1,140 Python
  tasks graded on 11 of the same models.
- **[RouterBench](https://huggingface.co/datasets/withmartian/routerbench):** used for the
  baseline comparisons.

```bash
python scripts/build_dataset.py --out data/processed          # about 2 GB download
python scripts/train_router.py --data data/processed --out artifacts/router.pt
python scripts/check_routing.py --router artifacts/router.pt --data data/processed
```

`train_router.py` does four things:

- It fits the decision graph on the train split.
- It calibrates it on half of validation.
- On the other half, it chooses each profile's threshold and fallback margin.
- It prints test results with 90% intervals and the coding-prompt breakdown.

The router file stores the predictor, the encoder name, the thresholds and the margins.
`--predictor ensemble` trains the kNN, matrix factorization, IRT and MLP ensemble instead,
for comparison.

The catalog in `configs/models.yaml` holds prices, energy, clearance and an `anchor` for
each model. The anchor is a graded model of a similar tier plus a logit shift, so you can
add a new model without retraining. Fit the shift from a few hundred graded answers with
`CatalogPredictor.fit_shift()`.

### On Lightning AI (free GPU tier)

```bash
pip install lightning-sdk
export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...
python scripts/lightning_run.py --teamspace <your-teamspace> --machine T4 --pytest
```

This ships the current commit to a Studio, builds the data, trains, downloads the reports
to `reports/lightning/` and always stops the Studio at the end. It refuses to start below
`--min-credits` (default 3) and stops after `--max-hours` (default 2). Inside a Studio you
can also run the commands above directly.

## Measuring

| Script | What it measures |
|---|---|
| `scripts/train_router.py` | Routing accuracy and cost per profile, with intervals |
| `scripts/check_routing.py` | Decisions on sampled prompts, by difficulty |
| `scripts/eval_confidentiality.py` | Privacy filter recall, false alarms and speed |
| `scripts/bench_route.py` | Routing time per stage |
| `scripts/e2e_check.py` | The gateway end to end (privacy, routing, profiles, streaming) |
| `scripts/train_baselines.py` | Baseline predictors on SPROUT and RouterBench |
| `scripts/sweep_graph.py` | The graph's settings, chosen on validation only |

## Repository layout

```
configs/      models.yaml (catalog: price, energy, clearance), providers.yaml (endpoints)
src/ecoroute/
  confidentiality/  levels, rules, PII model layer, caller policy, redaction
  graph/            the decision graph (skills, difficulty, similar prompts)
  predictors/       baselines, calibration, catalog anchoring
  routing/          profiles and the decision with its explanation
  gateway/          OpenAI-compatible FastAPI app and provider backends
  data/, eval/      dataset loaders and routing metrics
  service.py        EcoRoute: embed, predict, decide (save/load the router file)
scripts/      build, train, evaluate, serve
docs/         how-it-decides.md (method and results), design notes and research
```

Design background: [`docs/brainstorm.md`](docs/brainstorm.md),
[`docs/model-research.md`](docs/model-research.md),
[`docs/zero-cost-plan.md`](docs/zero-cost-plan.md).
