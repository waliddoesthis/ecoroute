# How EcoRoute decides, per prompt

Every request goes through the same three steps. The model list (`configs/models.yaml`) is
configuration; the algorithm does not depend on which models are in it.

## 1. Privacy filter (hard rule)

The prompt, and the whole conversation for chats, is classified as public, internal,
confidential or restricted. Models whose `clearance` is below that level are removed;
nothing else can bring them back. Layers run cheapest first and the strictest wins:

| Layer | Catches | Cost |
|---|---|---|
| Caller policy | sensitivity labels the calling app passes (`X-EcoRoute-Sensitivity`) | 0 ms |
| Rules | keys and tokens, cards (Luhn), IBANs (mod-97), SSNs, emails, phones, IPs, company terms | < 1 ms |
| PII model | names, addresses, IDs, passwords in free text (DeBERTa-v3 on ai4privacy) | ~120-160 ms CPU |

Once a prompt is restricted the slower layers are skipped. Measured on 2,000 labelled
texts and 2,000 benchmark prompts (run 6c): restricted recall 0.90, personal-data recall
0.975, ordinary prompts flagged 12%. Recall comes first: a false alarm only keeps a
prompt on a cleared model, a miss leaks it.

## 2. The decision graph: who is likely to answer correctly

```
prompt --P(skill|prompt)--> skill --P(level|prompt)--> (skill, level) --solve rate--> model
prompt --similarity--> similar past prompts --solved?--> model
```

- Skills are the task families of the outcome data (math, knowledge, reasoning, ...).
- Levels are easy / medium / hard, learned from how many models solved each training
  prompt and from RouterArena's labelled prompts.
- Each (skill, level) -> model edge is the share of such prompts that model solved.
- Similar past prompts add a second path, weighted by beta (chosen on validation).

A model's predicted success is the sum of its paths, and the heaviest paths are printed in
the explanation. Deployed models are scored through an anchor model of the training data
plus a shift that can be fitted on a few hundred graded answers.

## 3. Cost and energy: the cheapest model that is good enough

Among the allowed models whose predicted success is at least tau, take the lowest
`cost + lambda_energy * energy`. If none reaches tau, take the allowed model most likely
to succeed, or the cheapest one within the profile's fallback margin of it.

- API models cost their token prices. Self-hosted models cost their GPU's electricity
  while generating (plus an optional hourly hardware cost); nothing is free.
- Profiles are accuracy targets: `quality` gives up nothing against the most accurate
  model, `balanced` at most 1 point, `eco` at most 3. Training measures on validation
  the cheapest tau that meets each target and stores it in the router file, so tau is
  read on the predictor's own scale. Energy is ignored by `quality`, priced at about
  $0.25/kWh (electricity plus its carbon) by `balanced`, and ten times that by `eco`.
  Untrained defaults are 0.85, 0.7 and 0.6.
- The fallback margin covers prompts where no model reaches tau. The predictor rarely
  separates models by a few points there (a coding prompt sits at P 0.69 to 0.74 for
  every model), so paying the top price for that edge mostly buys nothing. Each margin is
  chosen after its tau, on the same validation prompts, and may only spend the slack
  that tau's bound leaves: what the margin itself costs, plus 1.645 standard errors, must
  fit within the target minus tau's bound (run 36). Held only to the overall bound
  (run 35), it saved more but pushed balanced's test interval to 1.25 points.

## What it achieves (SPROUT test set, held-out threshold)

Graph router v14 (run 36, trained on SPROUT plus BigCodeBench coding tasks), against
always using GPT-4o (test accuracy 0.845 on SPROUT prompts). Each profile's tau and
fallback margin are chosen on validation prompts the calibrator never saw, and qualify
only if the gap plus 1.645 standard errors stays within the profile's target.

Points lost against GPT-4o on test (negative means more accurate), with a 90% interval:

| Profile  | Target           | tau  | Margin | Points lost (90% CI)  | Cost saved |
|----------|------------------|------|--------|-----------------------|------------|
| quality  | no loss          | 0.98 | 0      | -1.6 (-2.2 to -0.9)   | 34.2%      |
| balanced | at most 1 point  | 0.91 | 0.02   | 0.3 (-0.4 to 1.0)     | 69.7%      |
| eco      | at most 3 points | 0.86 | 0      | 1.6 (0.8 to 2.3)      | 75.0%      |

On the 108 coding test prompts (BigCodeBench, hard: GPT-4o solves 54.6%), the router is
1.8 points below GPT-4o at 48% lower cost (60% for balanced, whose fallback margin takes
a cheaper model where none is clearly better); predicting which model solves a coding
task is still weak there (AUC 0.65). Earlier runs reported larger savings with taus
picked partly on the calibrator's own data, which made the predictions look better than
they were.

- Against a black-box ensemble (kNN, matrix factorization, IRT and MLP) under the same
  protocol (run 31), both reach GPT-4o's accuracy on test (0.848 vs 0.845); the
  ensemble saves 66.5% and the graph 64.9%. The graph is better calibrated (ECE 0.017
  vs 0.019) and explains every decision, so it stays the default.
- Telling hard from easy on unseen RouterArena prompts: AUC 0.76-0.80 across versions.
- The privacy check (rules + PII model) keeps 99% of texts with restricted data and 97%
  of texts with any personal data off external models, with 7.6% false alarms on
  ordinary prompts (run 27). Long bare numbers (9+ digits, not part of a calculation)
  count as confidential: most missed account and ID numbers carry no label. First names,
  cities, building numbers and similar weak kinds need PII-model confidence 0.85,
  passwords 0.9. About 26 ms per request on a T4 GPU.
- A whole routing decision (privacy check, embedding, prediction, explanation) takes
  about 72 ms per request on a T4 GPU; each response carries the breakdown in the
  X-EcoRoute-Time-Ms header.
