# EcoRoute: brainstorm

_Draft 2, 2026-10-05. Goal: for every prompt, pick the model that will answer it correctly at the lowest cost and energy, never sending confidential data where it is not allowed, and explain why._

---

## 1. What the industry actually does (research summary)

| System | How it decides | Takeaway for us |
|---|---|---|
| **RouteLLM** (LMSYS, open source) | Binary strong/weak routing. Four routers: similarity-weighted ranking, **matrix factorization**, BERT classifier, causal-LLM classifier. Trained on Chatbot Arena preference data. Reports 30-85% cost cuts at ~95% of GPT-4 quality. | Matrix factorization is the strongest of its four and is cheap. Good baseline. |
| **AWS Bedrock Intelligent Prompt Routing** | Predicts each model's response quality per prompt, switches from the fallback model only if the other is better by a configured "response quality difference". Exactly two models, same family, English only, no learning from your data. | The "quality margin threshold" is a clean, explainable knob. Their limits (2 models, no custom data) are exactly where we can do better. |
| **Azure AI Foundry Model Router, OpenAI GPT-5 router, NotDiamond, Martian** | Closed. Learned quality predictors over a fixed pool. RouterArena found commercial routers do not beat open ones (GPT-5 ranked #7). | No secret sauce to copy; the open literature is the state of the art. |
| **GraphRouter** (ICLR 2025) | Heterogeneous graph of task, query and LLM nodes; a GNN predicts the query→LLM edge (performance and cost). Generalises to new LLMs without retraining. | Direct match for your "graph with weights" idea. |
| **IRT-Router** (ACL 2025) | Item Response Theory: each prompt has a difficulty, each model an ability; P(correct) = σ(discrimination·(ability − difficulty)). | Gives us an **interpretable difficulty score** for free. |
| **CARROT** (2025) | Predicts per-model accuracy and cost separately, then picks the best trade-off. Released the SPROUT dataset. | Separate predictors per objective is the right decomposition. |
| **"Routing Plateau"** (arXiv 2606.07587, 2026) | 21 routers on 5 benchmarks land within 0.23 pts of each other. **kNN is consistently top-tier.** Hard prompts are 11-35% of data but 70-91% of the gap to oracle. | Don't over-engineer the predictor; spend effort on hard prompts, data scale, and a good encoder. Always benchmark against kNN. |

Nobody public routes on **confidentiality** or **energy**. That is our differentiator.

---

## 2. Core formulation

For prompt `q` and candidate model `m`:

```
maximise   U(q,m) = P̂(correct | q,m)  −  λ_cost · Cost(q,m)  −  λ_energy · Energy(q,m)
subject to Conf(q) ≤ Clearance(m)            (hard: never violated)
           P̂(correct | q,m) ≥ τ              (minimum quality floor)
           ctx_len(q) ≤ ContextWindow(m)
```

- `P̂(correct)` comes from the learned model (section 4). Must be **calibrated**, since thresholds rely on it.
- `Cost = in_tokens·price_in + E[out_tokens]·price_out`. We also predict output length, because output tokens dominate both cost and energy.
- `Energy = E[tokens] · Wh_per_token(m, hardware)`, from ML.ENERGY / AI Energy Score data. Optional: × grid carbon intensity of the region to get gCO₂.
- `λ` values are user-facing sliders ("eco mode", "budget mode"). Sweeping them traces the cost/quality Pareto curve.
- Confidentiality is a **constraint, not a weight**. A trade-off would let a big enough quality gain leak data; that is never acceptable.

---

## 2a. How we detect confidential content

Layered detection. **Any layer can raise the level, no layer can lower it**, and the final level is the maximum across layers. Each layer logs what fired, so the explanation can say *why* a prompt was classified.

| Layer | What it catches | How | Speed |
|---|---|---|---|
| **L0. Caller policy** | Things only the caller knows | The calling app passes metadata: user role, data source, document sensitivity label (e.g. Microsoft Purview / Google DLP labels). "Restricted" from the caller is final. | 0 ms |
| **L1. Deterministic rules** | Structured secrets and identifiers | Regex + validators: credit cards (Luhn check), IBAN (mod-97), SSN / national IDs, phone, email, IPs. Secret scanners (gitleaks / detect-secrets patterns): API keys, private keys, JWTs, high-entropy strings. Company dictionaries: client names, project codenames, internal domains. | < 1 ms |
| **L2. PII NER model** | Free-text personal data | DeBERTa-v3 token classifier fine-tuned on **ai4privacy** (names, addresses, health, finance), run through **Microsoft Presidio** as the framework so L1 and L2 share one pipeline. | ~5 ms CPU (ONNX) |
| **L3. Semantic classifier** | Business-confidential text with no explicit entity: contracts, unreleased financials, internal strategy, private source code | (a) Sentence classifier fine-tuned on a synthetic labelled set (4 levels). (b) Embedding similarity against fingerprints of the company's own confidential documents: "this prompt is 0.92 similar to `Q3-board-deck.pdf`". | ~5-10 ms |

Levels: **public → internal → confidential → restricted**. Each model in `configs/models.yaml` has a clearance (e.g. external API = internal, EU-hosted enterprise API with no-retention contract = confidential, local/on-prem = restricted).

Design rules:
- **Tuned for recall, not precision.** Target ≥ 99% recall on restricted. A false alarm costs a few cents (prompt goes to a local model); a miss leaks data.
- **Uncertain means higher.** If a classifier's confidence is in the grey zone, round up a level.
- **Redact-then-route (option).** For PII-only prompts, mask entities with placeholders (`<PERSON_1>`), send the masked prompt to a cheaper external model, then restore the values in the answer. This keeps confidentiality *and* the cost win when the task doesn't need the real values.
- **Tested in CI.** A fixed set of known-confidential prompts must never route to a model without clearance; one violation fails the build.

---

## 2b. Difficulty: easy prompts to cheap models, hard prompts to premium

This is the core decision rule, using the known data (past prompt × model outcomes):

> **Pick the cheapest allowed model whose predicted chance of answering correctly is above the quality floor τ.**

How it plays out:
- **Easy prompt** ("translate this sentence", "summarise this email"): the small model is predicted to succeed (e.g. 0.93 ≥ τ = 0.85), so it wins on cost and energy. Premium is never paid for.
- **Hard prompt** (multi-step math, tricky code, long legal reasoning): the small model drops to 0.40, mid-tier 0.70, premium 0.91. Only premium clears τ, so it goes to premium.
- **"Needs a bit more work"**: the same model with more reasoning effort (thinking budget) is treated as its own candidate with its own cost and energy. Often "mid model + more thinking" beats "premium model, no thinking" on price.
- **Unsure (grey band)**: cascade. Send to the cheap model, check the answer with a fast verifier (self-consistency, a judge model, or unit tests for code), escalate to premium only if it fails.

Where "known data" comes in:
1. **Cold start**: the IRT predictor is trained on public outcome data (SPROUT, RouterBench, RouterArena) so it knows from day one which kinds of prompts each model family handles.
2. **Learns your traffic**: every routed request logs `(prompt embedding, model, outcome signal)`: user thumbs up/down, judge score, whether it was escalated, retries. Periodic retraining shifts the boundaries toward *your* real prompts.
3. **New models**: a new model only needs its price, energy and a few hundred evaluated prompts to get its ability score θ, then it competes automatically.

The output exposes the reasoning: `difficulty: 0.72 (hard)`, `tier: premium`, predicted success per model, and the cost/energy difference vs. the alternative.

τ and the λ weights come from a policy profile (`eco`, `balanced`, `quality`) so each team can choose how much risk to take for savings.

---

## 2c. Easy integration into existing systems

Principle: **an app should adopt EcoRoute by changing one line**.

1. **OpenAI-compatible gateway (main path).** EcoRoute runs as an HTTP proxy that speaks the OpenAI Chat Completions API. Apps change `base_url` and set `model="ecoroute/auto"`. Everything else (SDKs, streaming, tools) keeps working. The response carries headers `X-EcoRoute-Model`, `X-EcoRoute-Level`, `X-EcoRoute-Difficulty`, `X-EcoRoute-Reason`.
2. **Decision-only mode.** `POST /route` (or `ecoroute.route(prompt)` in the Python SDK) returns the decision and explanation without calling any model, for systems that want to keep their own provider calls.
3. **Provider adapters via LiteLLM.** We don't write 100 provider clients; LiteLLM already speaks OpenAI, Anthropic, Bedrock, Azure, Vertex, vLLM and Ollama. EcoRoute plugs in as its routing strategy.
4. **Deployment.** Single Docker image; config in YAML (`models.yaml`, `policy.yaml`); runs on CPU (router models exported to ONNX), so no GPU is needed to serve. Router overhead target < 20 ms.
5. **Observability.** OpenTelemetry metrics and logs: cost saved, Wh and gCO₂ saved, routing mix, confidentiality hits. Ready for Grafana/Datadog.
6. **Fail-safe.** If the router errors or times out, fall back to a configured default model (respecting the confidentiality level; for restricted prompts the fallback is the local model).

---

## 3. Which graphs, and what each is for

We use three graphs, each with a different job.

### Graph A: Decision graph (the explainable "weighted graph" you pictured)
A layered DAG evaluated at inference time:

```
prompt → [Confidentiality gate] → [Difficulty tier] → [Candidate models] → answer
```
- Edges into a model node are pruned if the clearance constraint fails.
- Edge weights = `−U(q,m)` components (quality, cost, energy).
- Routing = **constrained shortest path**. Since the graph is small (≤ ~20 models), this is exact and instant.
- **The chosen path is the explanation**: "Classified *internal* (found an email address) → pruned 4 external APIs → difficulty 0.31 (easy) → Llama-8B-local: P(correct)=0.91, 0.02¢, 0.3 Wh vs. GPT-class: 0.94, 1.1¢, 4.2 Wh. Gain of 0.03 below the 0.05 margin, so the small model wins."
- Extension: a **cascade edge** (try small model → verifier → escalate) for prompts in the uncertain band.

### Graph B: Prompt similarity graph (kNN)
- Nodes = training prompts (embedded), edges = top-k cosine neighbours, edge weight = similarity.
- For a new prompt: find neighbours, average how each model did on them.
- Why: the strongest simple baseline in the literature, and gives example-based explanations ("5 most similar past prompts were all solved by the small model").

### Graph C: Heterogeneous task–prompt–model graph (GraphRouter-style)
- Node types: **Task** (math, code, chat…), **Prompt**, **Model** (features: size, price, energy, clearance, benchmark profile).
- Edges: task–prompt (membership), prompt–model (observed outcome: correctness, cost, tokens, energy as edge attributes).
- Train a GNN (GraphSAGE / HGT) for **edge prediction** on prompt→model.
- Why: adding a new model only requires its node features plus a handful of evaluated prompts, no retraining from scratch. This is the "research-grade" component.

---

## 4. Which ML model fits best

Recommendation: **a shared text encoder with an IRT-style multi-head predictor as the main model, kNN and matrix factorization as baselines, and the GNN as the upgrade we test against them.** Final choice goes to whichever wins on held-out data.

| Component | Model | Why |
|---|---|---|
| Prompt encoder | Frozen sentence embedder first (bge-m3 / gte-large class), then fine-tuned ModernBERT/DeBERTa-v3 | Routing Plateau shows encoder quality and end-to-end fine-tuning are where gains still are. |
| Quality predictor | **IRT head**: `P = σ(a_q · (θ_m − b_q))` with `a_q, b_q` predicted from the embedding and `θ_m` learned per model | Interpretable difficulty `b_q`, interpretable model ability `θ_m`, new models need few evals. |
| Output-length predictor | Small regression head (log-tokens) on the same embedding | Needed for real cost and energy estimates. |
| Confidentiality classifier | 4-layer pipeline from section 2a: caller policy, rules, PII NER, semantic classifier | Rules give guaranteed recall on known patterns, NER and the semantic layer catch the rest. Tuned for recall. |
| Baselines | kNN (Graph B), matrix factorization (RouteLLM), LightGBM on handcrafted features + SHAP | The bar any fancy model must clear. LightGBM+SHAP is also a second explanation channel. |
| Upgrade | Heterogeneous GNN (Graph C) | Tests whether graph structure beats the IRT head. |
| Calibration | Temperature or isotonic scaling per model head | Thresholds τ and the quality margin are meaningless without it. |

---

## 5. Data: what to use and why

### Routing outcomes (prompt × model → correct? cost?)
| Dataset | What it is | Why we use it |
|---|---|---|
| **SPROUT** (CARROT, HF `CARROT-LLM-Routing/SPROUT`) | Prompts × many recent models with correctness and token costs | Most modern model pool; primary training set. |
| **RouterBench** | 405k+ inference outcomes, 11 models, 8 benchmarks, with cost | Large, standard; lets us compare to published numbers. |
| **RouterArena** | 8,400 queries, 9 domains, 44 categories, **3 difficulty levels (Bloom's taxonomy)** | Labelled difficulty to validate our IRT `b_q`; also the evaluation leaderboard. |
| **RouterEval** | 8,500 LLMs over 12 benchmarks | Pretrain model-ability embeddings `θ_m` across a huge pool. |
| **Chatbot Arena human preference** (arena-human-preference 55k/100k) | Real user prompts with pairwise human judgements | Real-world prompt distribution (RouteLLM's training data). Benchmarks alone are too exam-like. |
| **LMSYS-Chat-1M / WildChat** | Unlabelled real conversations | Distribution coverage; label a sample with an LLM judge to extend training data. |

### Confidentiality
- **ai4privacy pii-masking-400k / openpii-1.5m**: large multilingual PII-labelled text for the NER model.
- Synthetic set we generate: prompts with code secrets, internal project names, contracts, medical/financial snippets, labelled into 4 levels (public / internal / confidential / restricted). Public datasets only cover PII, not business confidentiality.

### Cost and energy (model catalog, not training data)
- Prices: provider pricing pages / OpenRouter models API, stored in `configs/models.yaml`.
- Energy: **ML.ENERGY Leaderboard v3** (measured Wh per request for open models on real GPUs) and **HF AI Energy Score**. For closed APIs, energy is estimated from parameter-count proxies and flagged as an estimate in the explanation.

### Split strategy
Split by **task / source**, not randomly, so we measure generalisation to unseen kinds of prompts. Keep a "hard prompts" slice as its own report, since that is where routers lose.

---

## 6. Evaluation

- **Quality vs. cost Pareto curve**, summarised as APGR (RouteLLM) / AIQ (RouterBench).
- **Energy saved** (Wh and gCO₂) vs. always-strongest-model at equal quality.
- **Confidentiality violation rate** on restricted prompts: target 0, reported separately, blocks release.
- Calibration (ECE), latency overhead of the router itself (target < 20 ms), robustness to paraphrase (RouterArena metric).
- Gap-to-oracle broken down by difficulty tier.

---

## 7. Proposed project structure

```
ecoroute/
├── README.md
├── pyproject.toml                 # torch, transformers, torch-geometric, lightgbm, presidio, litellm, fastapi
├── Dockerfile
├── configs/
│   ├── models.yaml                # catalog: price, Wh/token, clearance, context, family
│   ├── policy.yaml                # profiles (eco/balanced/quality): λ_cost, λ_energy, τ, cascade band
│   ├── confidentiality.yaml       # levels, regex packs, dictionaries, caller-label mapping
│   └── train/*.yaml
├── data/                          # git-ignored; README documents sources and licences
│   ├── raw/ interim/ processed/
├── docs/
│   ├── brainstorm.md              # this file
│   └── decisions/                 # short ADRs for each choice we lock in
├── src/ecoroute/
│   ├── catalog/                   # model registry, pricing, energy lookup
│   ├── data/                      # loaders for SPROUT, RouterBench, Arena, ai4privacy
│   ├── features/                  # embeddings, handcrafted features, token-length estimator
│   ├── confidentiality/           # L0 policy, L1 rules, L2 NER, L3 semantic, redaction
│   ├── predictors/                # knn, matrix_factorization, irt, lightgbm, gnn
│   ├── graphs/                    # similarity graph, hetero graph, decision graph
│   ├── policy/                    # utility, constraints, constrained shortest path, cascade
│   ├── feedback/                  # outcome logging, retraining jobs
│   ├── explain/                   # turns the chosen path into a human explanation
│   ├── eval/                      # Pareto, APGR, energy, violation rate, calibration
│   ├── gateway/                   # OpenAI-compatible proxy + POST /route (FastAPI, LiteLLM)
│   └── sdk/                       # thin Python client
├── scripts/                       # download_data, build_graphs, train, evaluate, export_onnx
├── notebooks/                     # EDA only, nothing the pipeline depends on
└── tests/                         # incl. confidentiality never-leak suite
```

---

## 8. Roadmap

1. **Data foundation**: loaders + unified schema `(prompt, model, correct, in_tok, out_tok, cost, source, task)`; model catalog with prices and energy.
2. **Baselines**: kNN, matrix factorization, LightGBM; evaluation harness and Pareto plots.
3. **Main predictor**: IRT multi-head + output-length head, calibrated.
4. **Confidentiality layer**: L0-L3 pipeline, synthetic business-confidential set, redaction, never-leak suite in CI.
5. **Decision graph + explanations**, cascade option, OpenAI-compatible gateway, Docker image, feedback logging.
6. **GNN upgrade** (Graph C), kept only if it beats step 3.
7. Fine-tune the encoder end-to-end; focus on the hard-prompt slice.

---

## 9. Open questions for Walid

1. **Model pool**: which models should the router choose between? Confidentiality needs at least one model you run locally or on-prem.
2. **Compute**: do you have a GPU (for encoder fine-tuning, energy measurements of local models), or should training fit on CPU / Colab?
3. **Confidentiality levels**: are the 4 levels above right for your use, or do you have specific rules (e.g. a company policy) to encode?

---

## Sources
- RouteLLM overview and benchmarks: https://klymentiev.com/blog/llm-router
- Router comparison 2026: https://dreaming.press/posts/2026-06-21-routellm-vs-notdiamond-vs-martian.html
- AWS Bedrock prompt routing: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html
- GraphRouter (ICLR 2025): https://github.com/ulab-uiuc/GraphRouter, https://arxiv.org/pdf/2410.03834
- IRT-Router (ACL 2025): https://aclanthology.org/2025.acl-long.761/
- CARROT and SPROUT: https://arxiv.org/abs/2502.03261, https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT
- Routing Plateau: https://arxiv.org/pdf/2606.07587
- RouterArena: https://arxiv.org/html/2510.00202v1
- RouterBench: https://arxiv.org/abs/2403.12031
- RouterEval: https://github.com/MilkThink-Lab/RouterEval
- ai4privacy PII datasets: https://huggingface.co/ai4privacy
- Microsoft Presidio: https://github.com/microsoft/presidio
- LiteLLM: https://github.com/BerriAI/litellm
- ML.ENERGY Leaderboard v3: https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/
- Per-query LLM energy: https://muxup.com/2026q1/per-query-energy-consumption-of-llms
