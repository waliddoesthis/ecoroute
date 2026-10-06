# Which models companies use, and what they cost

_Research for EcoRoute, 2026-10-05. Prices are per 1M tokens, standard tier, no caching or batch discount. Anthropic prices come from Anthropic's own model table; the others come from pricing aggregators dated 2026-10-01 and must be checked against each provider's pricing page before production._

---

## 1. What companies actually run

**Enterprises are multi-model by default.**
- 37% of enterprises ran 5+ models in production in 2025, up from 29% in 2024 (a16z CIO survey). They pick models per use case, and security and cost now rank above raw accuracy in purchasing.
- That is exactly the gap EcoRoute fills: most of them pick the model per *application* by hand, not per *prompt*.

**Who gets the API spend (closed models):**
| Source | Date | Anthropic | OpenAI | Google | Open-weight |
|---|---|---|---|---|---|
| Menlo Ventures, enterprise LLM API spend | Dec 2025 | 40% | 27% | 21% | ~6% (Llama) + others |
| Ramp AI Index, business AI spend | May 2026 | 34.4% | 32.3% | n/a | n/a |
| Menlo, enterprise **coding** share | Dec 2025 | 54% | 21% | n/a | n/a |

**What they self-host (open-weight, "on the inside"):**
Companies run open models in-house for three reasons: data sovereignty (GDPR / Schrems II / US CLOUD Act), cost at high volume, and fine-tuning on their own data. Typical 2026 choices:

| Model | Size | Licence | Hardware | Typical use |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 35B MoE, 3B active | Apache 2.0 | 1× 24 GB GPU (Q4, ~20 GB) | Fast general assistant, lowest energy per token |
| Qwen3.6-27B | 27B dense | Apache 2.0 | 1× 24 GB GPU (Q4, ~16 GB) | Strongest single-GPU model, agentic coding |
| gpt-oss-20b | 21B MoE, 3.6B active | Apache 2.0 | 1× 16-24 GB GPU (~14 GB) | Reasoning and tool use |
| Gemma 4 26B | 26B MoE, 3.8B active | Apache 2.0 | 1× 24 GB GPU (~20 GB) | Multilingual, vision |
| Mistral Small 3.2 | 24B dense | Apache 2.0 | 1× 24 GB GPU (~14 GB) | Light, EU-friendly daily assistant |
| gpt-oss-120b | 117B MoE, 5.1B active | Apache 2.0 | 1× 80 GB GPU | o4-mini-level reasoning on-prem |
| DeepSeek V4-Flash / V4-Pro | 284B / 1.6T MoE | MIT | 1-4× / 8× H100 | Frontier-class on-prem, large companies only |
| Llama 4 Scout | 17B active MoE | Llama licence | 1 GPU | Very long context document work |

Pattern seen across sources: **a local or private-cloud open model for sensitive and high-volume work, plus 1-2 frontier APIs for hard tasks**. Self-hosting a 24B model breaks even against a premium API at roughly 10B tokens/month (gosign.de estimate).

---

## 2. API prices (October 2026)

### Anthropic (official)
| Model | Input | Output | Context | Notes |
|---|---|---|---|---|
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K | Fast, cheap tier |
| Claude Sonnet 5.5 | $2.00 | $10.00 | 1M | Everyday coding / agents / enterprise |
| Claude Opus 5.5 | $4.00 | $20.00 | 1M | Default flagship |
| Claude Fable 5.1 | $10.00 | $50.00 | 1M | Most capable; **requires 30-day data retention**, so never cleared for confidential data |

### OpenAI (aggregator, 2026-10-01)
| Model | Input | Output | Context |
|---|---|---|---|
| GPT-6 Luna | $0.10 | $0.50 | 1.05M |
| GPT-5.4 nano | $0.20 | $1.25 | 400K |
| GPT-5.4 mini | $0.75 | $4.50 | 400K |
| GPT-6 Sol | $2.00 | $10.00 | 1.05M |
| GPT-5.5 | $5.00 | $30.00 | 1M |
| GPT-6 Astra | $10.00 | $50.00 | 1.05M |
| GPT-5.5 Pro | $30.00 | $180.00 | 1M |

### Google Gemini (aggregator, 2026-10-01)
| Model | Input | Output | Context |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | 1M |
| Gemini 3.1 Flash-Lite | $0.25 | $1.50 | 1M |
| Gemini 3.8 Flash | $0.75 | $3.75 | 1M |
| Gemini 3.1 Pro | $2.00 | $12.00 | 1M (rises to $4 / $18 above 200K input) |

### Others
| Model | Input | Output | Notes |
|---|---|---|---|
| DeepSeek V4.1 Flash | $0.30 | $1.20 | Peak rate; ~half off-peak. China-hosted API: public data only |
| DeepSeek V4 Pro | $1.32 | $3.96 | Same caveat; weights are MIT, can be self-hosted instead |
| Mistral Small 4 | $0.15 | $0.60 | EU provider; open weights |

**Price spread: the cheapest capable model is ~100× cheaper than the most expensive** (GPT-6 Luna output $0.50 vs. Fable 5.1 / GPT-6 Astra $50). That spread is the money EcoRoute saves.

---

## 3. Energy: what we can and can't know

- **Open models (measured):** ML.ENERGY Leaderboard v3 on B200: ~0.15 J/token for chat, ~0.31 J/token for problem solving. A reasoning answer averages ~4.6 kJ vs. ~0.18 kJ for a chat answer (**25× more**) because it writes ~10× more tokens.
- **MoE beats dense:** Qwen3 30B-A3B uses **3.56× less energy per token** than dense Qwen3 32B at similar size. This is why the local default below is a MoE model.
- **Batching** cuts energy per token 3-5×, so a busy shared GPU is greener than an idle one.
- **Closed APIs:** providers don't publish per-model numbers. Google reported a median Gemini text prompt at ~0.24 Wh (Aug 2025) without saying which model or prompt length. So for APIs we use **order-of-magnitude priors by model tier**, clearly labelled as estimates, and the explanation says "estimated".
- **Your GPU:** we measure local energy ourselves (NVML power readings × time per request) so the local numbers are real, not guessed.

Takeaway for routing: **output length and reasoning drive energy more than model choice alone**. EcoRoute should predict output length and treat "same model, less thinking" as a candidate.

---

## 4. Recommended starting pool for EcoRoute

Eight models across five tiers: enough spread to learn real trade-offs, and each tier has two providers so outages and price changes don't break routing.

| Tier | Model | Why it's in | Clearance (default) |
|---|---|---|---|
| 0. Local | **Qwen3.6-35B-A3B** on your GPU (alt: gpt-oss-20b) | Free per token, lowest energy (MoE), the only home for restricted data | restricted |
| 1. Budget | **GPT-6 Luna** | Cheapest capable API ($0.10 / $0.50) | internal |
| 1. Budget | **Gemini 3.1 Flash-Lite** | Second budget provider | internal |
| 2. Mid | **Claude Haiku 4.5** | Strong small model, popular in enterprise | internal |
| 2. Mid | **Gemini 3.8 Flash** | Good price / quality | internal |
| 3. Strong | **Claude Sonnet 5.5** | The enterprise workhorse for coding and agents | internal |
| 3. Strong | **Gemini 3.1 Pro** | Cheaper strong alternative | internal |
| 4. Premium | **Claude Opus 5.5** | Hardest prompts | internal |

Clearance rule: external APIs default to **internal**. A provider moves up to **confidential** only if you have a zero-data-retention / DPA contract with it, set in `configs/models.yaml`. Fable 5.1 and DeepSeek's hosted API are excluded from confidential data by design.

Optional later: GPT-6 Astra or Claude Fable 5.1 as a "tier 5" for the very hardest prompts, once the data shows Opus 5.5 failing on a slice worth paying for.

### Important: public datasets don't contain these models (see docs/zero-cost-plan.md for the $0 route)
SPROUT, RouterBench and RouterArena were built on older models. Two ways to bridge:
1. **Our own eval run (recommended):** take ~3,000 prompts sampled across RouterArena / SPROUT difficulty levels, run them through every pool model, grade them. Rough API cost: **~$110 without reasoning tokens, likely $300-500 with them** (Opus 5.5 and Gemini 3.1 Pro are most of it). The local model is free on your GPU.
2. **IRT transfer:** fit the router on the big public datasets, then place each new model on the same ability scale from a few hundred of its own graded answers. Cheaper, less accurate. We use this anyway whenever a new model is added.

---

## Sources
- Menlo Ventures enterprise LLM report (via): https://valueaddvc.com/blog/anthropic-market-share-2026-40-percent-enterprise-llm-spend-openai
- Ramp AI Index 2026 (via): https://www.mindstudio.ai/blog/anthropic-vs-openai-business-adoption-2026-ramp-data-2
- a16z, How 100 Enterprise CIOs Are Building and Buying Gen AI: https://a16z.com/ai-enterprise-2025/
- Self-hosted open-source AI 2026: https://www.gosign.de/en/magazine/self-hosted-open-source-ai-2026/
- Best local LLMs on a 24 GB GPU (Jul 2026): https://www.marktechpost.com/2026/07/19/best-local-llms-you-can-run-on-a-single-24gb-gpu-in-2026-qwen-gemma-mistral-deepseek-compared/
- OpenAI pricing (Oct 2026): https://benchlm.ai/openai/api-pricing
- Gemini pricing (Oct 2026): https://benchlm.ai/google/api-pricing
- DeepSeek pricing (Oct 2026): https://benchlm.ai/deepseek/api-pricing
- Mistral pricing (Oct 2026): https://benchlm.ai/mistral/api-pricing
- ML.ENERGY Leaderboard v3: https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/
- Per-query energy of LLMs: https://muxup.com/2026q1/per-query-energy-consumption-of-llms
