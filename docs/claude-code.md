# EcoRoute in Claude Code

Claude Code sends every request through EcoRoute, which checks it for confidential data,
picks the cheapest Claude model predicted to handle it, and passes the answer back. A hook
shows each decision in Claude Code:

```
EcoRoute: claude-haiku-4-5 (easy, 75% cheaper than claude-opus-5-5)
```

## Set up (two commands)

```bash
python scripts/serve.py --router artifacts/router.pt      # the gateway, on :8080
python scripts/setup_claude_code.py                       # point Claude Code at it
```

Restart Claude Code. Your Anthropic key or Claude login works as before: the gateway passes
it through to Anthropic and never stores it. For a gateway someone hosts for you:

```bash
python scripts/setup_claude_code.py --url https://ecoroute.example.com --key YOUR_ACCESS_KEY
```

`--project` writes `.claude/settings.json` in the current repository instead of your user
settings, and `--remove` undoes everything (a `.bak` copy of the old file is kept).

What the setup writes:

| setting | effect |
|---|---|
| `env.ANTHROPIC_BASE_URL` | Claude Code sends its requests to the gateway's `/v1/messages` |
| `hooks.UserPromptSubmit` | `integrations/claude-code/ecoroute_hook.py` prints the decision for each prompt |
| `env.ANTHROPIC_CUSTOM_HEADERS` | with `--key`: `X-EcoRoute-Key` for a hosted gateway |

To try it without any API key or cost, run the gateway with `--dry-run`: Claude Code then
answers every prompt with the name of the model that would have been used.

## What happens to each request

1. **Privacy check.** The gateway scans what you typed, every tool call and every tool
   result (the files the agent read). Claude Code's own system prompt and
   `<system-reminder>` blocks are skipped: they are the tool's instructions, the same on
   every request, and full of session IDs that look like secrets. Each block's result is
   cached, so a long session is not scanned again on every turn.
2. **Redaction instead of blocking.** Spans no reachable model is cleared for (an AWS key
   in a config file, a card number) are replaced by placeholders such as
   `<AWS_ACCESS_KEY_3f9a1c2b>` before the request leaves. The real values are put back in
   the answer, including inside tool calls, so when the agent writes the file the real
   value lands on disk. The same value always gets the same placeholder, so Anthropic's
   prompt cache keeps working. `--no-redact` refuses such requests instead (HTTP 403).
   A label you set yourself with `X-EcoRoute-Min-Level` is never redacted away.
3. **The graph picks the model** from what you typed last. The agent's follow-up turns
   (tool results only) share that prompt, so a whole task stays on one model.
4. **Claude Code's background tasks** (titles, summaries) ask for Haiku; a request for the
   cheapest model on offer stays on it.
5. **If the chosen model refuses the request** (HTTP 400, for example a thinking setting a
   smaller model does not support), it is retried without the optional settings, then on
   the model Claude Code asked for. Rate limits and overload errors go back unchanged so
   Claude Code's own retries work.

Response headers (`X-EcoRoute-Model`, `-Reason`, `-Difficulty`, `-Redacted`) carry the
decision; `POST /route` with `X-EcoRoute-Api: messages` returns the full explanation.

## Limits

- **Claude models only inside Claude Code.** Only providers that accept Anthropic's
  Messages format (`anthropic_base_url` in `configs/providers.yaml`) can be chosen. Claude
  Code's work depends on Claude's tool calls; translating them for GPT or Gemini is lossy
  and can break edits, so the gateway does not do it. Apps that use the OpenAI format
  (`/v1/chat/completions`) can still route to any provider.
- **The PII model reads only what you typed** by default. Keys, card numbers and IDs are
  caught in every file by the rules, but the PII model is slow on whole files on a CPU
  and takes code for names (it tags model IDs as usernames). `--pii-on-tools` turns it on
  for tool results too.
- **The quality predictions for Claude models are priors.** The catalog places each model
  through an older benchmark model of a similar tier (`anchor` in `configs/models.yaml`),
  so on some prompts Haiku is predicted as likely to succeed as Opus. Retraining on your
  own graded prompts (see the README) replaces those priors with measured ones.
