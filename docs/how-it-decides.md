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
to succeed.

- API models cost their token prices. Self-hosted models cost their GPU's electricity
  while generating (plus an optional hourly hardware cost); nothing is free.
- Profiles: `quality` (tau 0.85, energy ignored), `balanced` (tau 0.7, energy at about
  $0.25/kWh, roughly electricity plus its carbon), `eco` (tau 0.6, energy x10).

## What it achieves (SPROUT test set, held-out threshold)

- Graph router (run 9): 60% cheaper than always using GPT-4o, accuracy within 1 point.
- Telling hard from easy on unseen RouterArena prompts: AUC 0.80.
