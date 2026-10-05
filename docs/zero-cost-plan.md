# Building EcoRoute for $0

_2026-10-05. Constraint: no money spent on APIs or compute during development._

## 1. Where the $300-500 came from, and why we can skip it

That figure was the API bill for making every pool model answer ~3,000 test prompts so we could grade them. It is not training cost. Training the router itself is cheap: it's a small encoder plus a few prediction heads.

We avoid the bill by:
1. **Training on public outcome datasets** (SPROUT, RouterBench, RouterArena, EmbedLLM, Open LLM Leaderboard details). Someone else already paid to have models answer and grade those prompts. That covers ~400K+ graded (prompt, model) pairs at no cost.
2. **Labelling open models ourselves on free GPUs** (Kaggle, Lightning) with vLLM. Free.
3. **Labelling Gemini Flash models through the Gemini API free tier.** Free, but Google may use free-tier inputs to improve its products, so only public benchmark prompts go there, never real data.
4. **Placing paid-only models (Claude, OpenAI, Gemini Pro) from published benchmark scores**, not our own calls. The IRT model puts each one on the same ability scale using its public scores (MMLU-Pro, GPQA, LiveCodeBench, Arena, etc.). Less precise than our own grading. In production, the feedback loop then corrects it from real traffic.

## 2. Free compute

| Where | What you get | Use it for |
|---|---|---|
| **Your PC**: RTX 2070 Super, 8 GB | Fine for coding, tests, running the router (it runs on CPU), and a small 4-8B local model at 4-bit | Daily development, demo of the local tier |
| **Lightning AI free** | ~15 credits/month top-up, no card (aimultiple). One other source says a one-time 30 credits once a card is added; check your account page. At listed rates that's **~19 h of L4 (24 GB)** or **~27 h of T4 (16 GB)** a month. 1 Studio, max 2 GPUs at once, 50 GB storage. | Training the router: embeddings, heads, encoder fine-tune |
| **Kaggle notebooks** | ~30 GPU h/week, T4 ×2 (16 GB each) or P100, phone verification needed | Generating labels with open models (vLLM), bigger experiments |
| **Gemini API free tier** | Flash models, Gemma 4, embeddings; limits shown per project in AI Studio | Labelling Gemini Flash on public prompts |
| **OpenRouter `:free` models** | 20 req/min, 50 req/day (1,000/day after a one-time $10 top-up) | Too slow for bulk; spot checks only |

### Does the free compute cover training? Yes.
- Embedding ~400K prompts with a base-size encoder: about 1 h on an L4.
- Training kNN, matrix factorization, IRT and LightGBM on those embeddings: minutes, even on CPU.
- Fine-tuning a ModernBERT/DeBERTa-base encoder end to end: a few hours on an L4, within one month of Lightning credits.
- GNN (GraphRouter-style) on this graph size: under an hour.

Budget rule: develop and debug on your PC or Kaggle, and spend Lightning L4 hours only on real training runs. Checkpoint every run to storage so a 4-hour Studio restart loses nothing.

## 3. What changes in the design

- **Local tier on your machine:** with 8 GB, the local model becomes a 4-8B model at 4-bit, instead of Qwen3.6-35B-A3B, which needs 24 GB. On a bigger GPU later, it is one line in `configs/models.yaml`.
- **Each model gets a `label_source`:** `public_dataset`, `free_eval` (our own grading), or `benchmark_prior` (placed from published scores). The explanation shows it, so a decision based on a prior is labelled as less certain.
- **Energy:** local energy is still measured for real (on your GPU and the cloud GPUs); API energy stays an estimate.
