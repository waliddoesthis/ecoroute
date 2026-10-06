# Impact at company scale

This page turns EcoRoute's measured per-request savings into yearly figures at typical
enterprise volumes. Only the saving rates are measured. Volumes, request sizes and
prices are assumptions, stated next to each table, so every figure here is an
**estimate**. To recompute them with your own numbers, run `python scripts/impact.py`.

## 1. What is measured

| Comparison | Profile | Cost saved | Energy saved | Accuracy |
|---|---|---|---|---|
| **A. vs always GPT-4o**, SPROUT test prompts (run 36) | quality | 34.2% | n/a | 1.6 points better |
| | balanced | 69.7% | n/a | 0.3 points lower (90% CI -0.4 to 1.0) |
| | eco | 75.0% | n/a | 1.6 points lower |
| **B. vs always the top deployed model** (claude-opus-5-5), 3,000 RouterArena prompts (run 38) | quality | 30.3% | 29.6% | not graded |
| | balanced | 53.2% | 53.4% | not graded |
| | eco | 55.4% | 54.8% | not graded |

- **A** is the headline result. It uses real correctness labels, so its accuracy is
  verified.
- **B** runs the deployed catalog through the same router. Its model choices come from
  benchmark anchors, its prices are list prices, and its API energy uses tier priors. It
  shows the order of magnitude a company would see with today's models.
- **B saves less than A** for two reasons. The deployed catalog's cheap models are
  anchored conservatively, and RouterArena prompts are harder on average (about half of
  its hard prompts still go to the top model).

## 2. Cost

**Assumptions:**

- Each request has 1,000 input tokens and 400 output tokens.
- GPT-4o list price is $2.50 and $10.00 per million tokens, so $0.0065 per request.
- The saving rates come from comparison A.

| Requests / month | Always GPT-4o / year | quality saves / year | balanced saves / year | eco saves / year |
|---|---|---|---|---|
| 1M | $78,000 | $26,676 | $54,366 | $58,500 |
| 10M | $780,000 | $266,760 | $543,660 | $585,000 |
| 100M | $7,800,000 | $2,667,600 | $5,436,600 | $5,850,000 |

For scale, an assistant used by 10,000 employees making about 50 requests per working day
generates about 10M requests a month.

Spend grows linearly with price, so routing matters more as reference models get more
expensive. Against the top deployed model (comparison B, $0.0107 per request at 500
output tokens), the balanced profile saves 53%. At 10M requests a month, that is about
$680k a year.

## 3. Energy and CO2

**Assumptions:**

- Energy per request comes from comparison B: 0.500 Wh always using the top model, and
  0.233 Wh with the balanced profile, at 500 output tokens.
- API models use tier priors of 0.05 to 1.0 Wh per 1,000 output tokens. Providers don't
  publish per-model figures, so these are order-of-magnitude estimates in line with
  Google's published median of about 0.24 Wh per Gemini prompt.
- Grid intensity is 0.4 kg CO2 per kWh.

| Requests / month | Energy saved / year | CO2 avoided / year |
|---|---|---|
| 1M | 3,204 kWh | 1.3 t |
| 10M | 32,040 kWh | 12.8 t |
| 100M | 320,400 kWh | 128.2 t |

Roughly, 32 MWh a year is the electricity use of about 9 average EU homes, or of 3
average US homes. Per request the absolute numbers are small, but they scale linearly
with volume. The relative saving (about half the energy for the same task) holds wherever
cheaper models are also smaller.

## 4. Risk reduction

These benefits are not priced above, but they matter as much to a company:

- **Leakage prevented.** 99.0% of texts with restricted data and 96.7% with personal data
  are kept off external models (run 27). This happens without asking employees to judge
  each prompt.
- **Explainability.** Each decision states the level found, the models cleared and the
  reason for the choice, which gives audits and incident reviews a concrete record.
- **Vendor flexibility.** Models are configuration. A company can add a new provider or a
  self-hosted model without retraining, and spread traffic across vendors.

## 5. Caveats

- The saving rates are measured on public benchmarks. A company's own rate depends on how
  much of its traffic is easy. Shadow mode
  ([integration guide, section 6](enterprise-integration.md#6-rollout)) measures it on
  real traffic before any change.
- The figures exclude EcoRoute's own running cost: one T4-class GPU for about 74 ms per
  request, or CPU only for the rules layer.
- Energy figures for API models are priors. Replace them with measured or
  provider-reported values in `configs/models.yaml` when they become available.
