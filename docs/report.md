# EcoRoute: privacy-constrained, explainable cost routing for LLM traffic

*Technical report, October 2026.*

## Abstract

Companies send most LLM requests to their most capable model, although a large share of
those requests would be answered just as well by a model that costs a fraction as much.
Many also cannot send confidential data to external providers at all. EcoRoute is an
OpenAI-compatible gateway that handles both problems for each request:

1. A layered privacy filter removes every model not cleared for the data in the
   conversation.
2. A weighted decision graph predicts which of the remaining models will answer
   correctly.
3. The cheapest model predicted to succeed is chosen.

Every decision comes with a plain-language explanation. Thresholds are chosen on held-out
data with a statistical margin, so each operating profile carries an accuracy guarantee
checked on a separate test set.

Results on SPROUT test prompts, against always using GPT-4o:

| Profile | Cost saved | Accuracy |
|---|---|---|
| default (balanced) | 69.7% | 0.3 points lower (90% CI -0.4 to 1.0) |
| eco | 75.0% | 1.6 points lower |
| quality | 34.2% | 1.6 points higher |

The privacy filter keeps 99.0% of texts containing restricted data and 96.7% of texts
containing personal data off external models, with 7.6% false alarms. A routing decision
adds about 74 ms on one T4 GPU.

## 1. Problem

For a stream of prompts *x*, a set of models *M*, and for each model *m* a price *c(m, x)*,
an energy use *e(m, x)* and a clearance level, choose for each prompt a model that:

1. is cleared for the confidentiality level of *x* (hard constraint);
2. answers correctly with high probability; and
3. minimises *c + λ·e* among the models that satisfy 1 and 2.

The accuracy of the whole policy should stay within a stated tolerance of the best single
model, and that tolerance should be verified on data the policy never saw.

## 2. Method

### 2.1 Privacy filter

Layers run cheapest first, and the strictest level wins:

| Layer | Detects |
|---|---|
| Caller policy | Labels from existing DLP schemes (an unknown label counts as confidential) |
| Rules | Keys and tokens, private keys, connection strings, cards (Luhn), IBANs (mod-97), SSNs, identifiers introduced by a label, unlabelled numbers of 9 or more digits, emails, phones, IPs |
| PII model | piiranha (DeBERTa-v3 fine-tuned on ai4privacy), run in batched windows over the whole conversation |

Each detected kind has its own confidence floor. Weak kinds such as first names, cities
and building numbers need a score of 0.85, and passwords 0.9. Models with lower clearance
are removed before any scoring.

### 2.2 Decision graph

The probability that model *m* answers prompt *x* correctly is the sum of weighted paths:

```
prompt --P(skill|x)--> skill --P(level|x)--> (skill, level) --solve rate--> model
prompt --similarity--> k nearest past prompts --solved?--> model   (weight beta)
```

- **Skills** are the task families in the outcome data.
- **Difficulty levels** (easy, medium, hard) are learned from how many models solved each
  training prompt, plus RouterArena's labelled prompts.
- **Edge weights** are smoothed solve rates (shrinkage 20).
- **Prompt representation:** BAAI/bge-small-en-v1.5 embeddings.
- **Calibration:** the scores are calibrated on held-out data.
- **Deployed models:** they are linked to the benchmark models through an anchor (a model
  of similar tier) plus a logit shift. The shift can be fitted from a few hundred graded
  answers.

The heaviest paths become the explanation.

### 2.3 Decision rule

Among the cleared models with *P ≥ τ*, the router takes the lowest *c + λ·e*.

When no model reaches *τ*, it takes the cleared model with the highest *P*. Within the
profile's fallback margin *δ* of that maximum, it takes the cheapest model instead.

Each profile (quality, balanced, eco) has a target: no loss, at most 1 point and at most
3 points against the most accurate single model. *λ* prices energy at 0, about
$0.25/kWh and about $2.5/kWh respectively.

### 2.4 Choosing thresholds without leakage

The validation split is cut in two:

- The calibrator is fitted on the second half.
- *τ* and *δ* are chosen on the first half.

*τ* is the cheapest value on a 0.01 grid whose accuracy gap to the reference, plus 1.645
paired standard errors, stays within the profile's target. The margin *δ* is then chosen
the same way, but it may only use the slack that *τ* leaves: the gap the margin adds,
plus its own 1.645 standard errors, must fit within the target minus *τ*'s bound. Test
results are reported once, with 90% bootstrap intervals.

## 3. Data

| Dataset | Use | Size |
|---|---|---|
| SPROUT (CARROT-LLM-Routing) | Correctness of 13 models on varied prompts (judge-graded) | ~44k prompts |
| BigCodeBench (instruct, v0.1.4) | Python tasks with pass/fail for 11 of the same models | 1,140 tasks |
| RouterArena | Difficulty labels; held-out difficulty check | 8,400 prompts |
| RouterBench | Baseline comparison | ~36k prompts |
| ai4privacy pii-masking-400k | Privacy recall (English validation texts) | 2,000 texts |

Splits are assigned from a hash of the prompt, so every model's answer to a prompt
lands in the same split (train 80%, validation 10%, test 10%). All data is open, and no
paid grading was used.

## 4. Results

### 4.1 Routing (SPROUT test split, router v14)

Reference: always GPT-4o, accuracy 0.845. Points lost are negative when the router is
more accurate.

| Profile | Target | τ | δ | Points lost (90% CI) | Cost saved |
|---|---|---|---|---|---|
| quality | 0 | 0.98 | 0 | -1.55 (-2.22 to -0.85) | 34.2% |
| balanced | ≤ 1 | 0.91 | 0.02 | 0.27 (-0.43 to 0.99) | 69.7% |
| eco | ≤ 3 | 0.86 | 0 | 1.57 (0.79 to 2.34) | 75.0% |

All three intervals lie within their targets. On the 108 BigCodeBench test tasks
(GPT-4o solves 54.6%), the router is 1.8 points below GPT-4o at 48% lower cost, and 60%
lower with the balanced profile.

### 4.2 Ablations

**Predictor** (SPROUT test; AUC is per model, Brier is per prompt and model):

| Variant | AUC | Brier | Hard vs easy AUC on RouterArena |
|---|---|---|---|
| Graph, skills and solve rates only (run 8) | 0.784 | 0.136 | 0.615 |
| + labelled difficulty edge (run 9) | 0.782 | 0.137 | 0.796 |
| + similar-prompt edge (run 12) | 0.802 | 0.132 | 0.763 |
| + coding data (run 34, final graph) | 0.803 | 0.134 | 0.759 |
| Black-box ensemble: kNN, MF, IRT, MLP (run 31) | 0.811 | 0.129 | n/a |

**Decision protocol** (balanced profile):

| Variant | Points lost (90% CI) | Saved | Meets target? |
|---|---|---|---|
| τ chosen on calibration data (runs 15-29) | overstated | - | not verifiable (leakage) |
| Leak-free τ (run 30) | -0.31 (-1.00 to 0.40) | 64.9% | yes |
| + coding data (run 34) | -0.05 (-0.73 to 0.70) | 67.3% | yes |
| + fallback margin, overall bound only (run 35) | 0.48 (-0.24 to 1.25) | 72.6% | no, CI exceeds 1 |
| + margin limited to τ's slack (run 36, final) | 0.27 (-0.43 to 0.99) | 69.7% | yes |

Under the same protocol (run 31), the ensemble saves 66.5% and the graph 64.9%. The graph
is kept because it is better calibrated (ECE 0.017 against 0.019) and explains each
decision. A validation sweep of the graph's settings (C, k, shrinkage; run 32) was flat
(Brier 0.130 to 0.132), so the defaults were kept.

### 4.3 Privacy filter

Recall is measured on 2,000 ai4privacy texts. False alarms are measured on 2,000 ordinary
benchmark prompts.

| Version | Restricted kept local | Personal data caught | False alarms |
|---|---|---|---|
| Rules + PII model, single threshold (run 6c) | 89.6% | 97.5% | 12.0% |
| + per-kind floors, labelled identifiers, long numbers (run 27) | 99.0% | 96.7% | 7.6% |

### 4.4 Latency (one NVIDIA T4)

| Stage | Median |
|---|---|
| Privacy check (rules + PII model) | 26 ms (p95 81 ms) |
| Embedding + prediction | 27 ms |
| Full routing decision | 74 ms (p95 121 ms) |
| Rules only, no GPU | 33 ms |

Replacing a per-query nearest-neighbour search with a cached matrix product brought a
decision down from 360 ms to about 72 ms (runs 18 to 20).

### 4.5 End to end

`scripts/e2e_check.py` sends real requests through the HTTP gateway with router v14 and a
backend that calls no model. It passes all 16 checks (run 37):

- secrets, emails, bare account numbers, caller labels, and names with addresses stay on
  the local model;
- matched secrets are never echoed;
- uncleared and unknown models are refused;
- the profiles order cost as expected;
- streaming works.

## 5. Limitations

- **Benchmark prompts, not company traffic.** The savings are measured on public benchmark
  prompts and pricing. Absolute savings depend on a company's prompt mix and contracts.
  Fitting the deployed models' shifts on internal graded answers is the intended next
  step.
- **Coding prompts.** Prediction is weak on coding prompts (AUC 0.65). More graded coding
  data is the clearest route to improvement.
- **Energy.** Energy for API models uses tier priors, because providers do not publish
  per-model figures. Self-hosted energy is computed from GPU power and throughput.
  Energy results are therefore estimates (see [impact.md](impact.md)).
- **The reference model.** The reference is GPT-4o, the strongest model with outcomes in
  SPROUT. Deployed catalogs use newer models through anchors.
- **Privacy recall.** Recall is high but not perfect. The filter is a technical control
  next to existing DLP, not a guarantee.

## 6. Reproducing

```bash
python scripts/build_dataset.py --out data/processed
python scripts/train_router.py --data data/processed --out artifacts/router.pt
python scripts/check_routing.py --router artifacts/router.pt --data data/processed
python scripts/eval_confidentiality.py --data data/processed --device 0
python scripts/e2e_check.py --router artifacts/router.pt --pii-device 0
```

All runs used Lightning AI's free GPU tier (T4). Each run's console output is kept with
the project's run records.
