# Data

Nothing in `raw/`, `interim/` or `processed/` is committed. Rebuild with:

    python scripts/build_dataset.py --out data/processed

| Source | Hugging Face id | What we use | Licence |
|---|---|---|---|
| SPROUT | `CARROT-LLM-Routing/SPROUT` | ~44K prompts x 13 models, judge scores, token counts | see dataset card |
| RouterBench | `withmartian/routerbench` (`routerbench_0shot.pkl`) | ~36K prompts x 11 models, correctness, cost | see dataset card |
| RouterArena | `RouteWorks/RouterArena` (`full` split) | 8.4K prompts with easy/medium/hard labels, no outcomes | see dataset card |

Check each dataset card's licence before using the data commercially.

## Unified tables (`src/ecoroute/data/schema.py`)
- `outcomes.parquet`: `prompt_id, source, source_id, task, prompt, model, correct, in_tokens, out_tokens, cost_usd, split`
- `prompts.parquet`: `prompt_id, source, source_id, task, domain, difficulty, prompt, answer, split`

`prompt_id` is a hash of the normalised prompt text, so the same prompt in two sources
shares an id. The split comes from that id (80/10/10), so a prompt never lands in both
train and test.
