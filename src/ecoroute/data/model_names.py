"""Map each source's model naming to one canonical name.

The same model appearing in two datasets must get the same name, otherwise the
predictors learn two separate abilities for it.
"""

from __future__ import annotations

CANONICAL: dict[str, str] = {
    # SPROUT (IBM / CARROT)
    "aws-claude-3-5-sonnet-v1": "claude-3.5-sonnet",
    "aws-titan-text-premier-v1": "titan-text-premier",
    "openai-gpt-4o": "gpt-4o",
    "openai-gpt-4o-mini": "gpt-4o-mini",
    "wxai-granite-3-2b-instruct-8k-max-tokens": "granite-3-2b-instruct",
    "wxai-granite-3-8b-instruct-8k-max-tokens": "granite-3-8b-instruct",
    "wxai-llama-3-1-70b-instruct": "llama-3.1-70b-instruct",
    "wxai-llama-3-1-8b-instruct": "llama-3.1-8b-instruct",
    "wxai-llama-3-2-1b-instruct": "llama-3.2-1b-instruct",
    "wxai-llama-3-2-3b-instruct": "llama-3.2-3b-instruct",
    "wxai-llama-3-3-70b-instruct": "llama-3.3-70b-instruct",
    "wxai-llama-3-405b-instruct": "llama-3.1-405b-instruct",
    "wxai-mixtral-8x7b-instruct-v01": "mixtral-8x7b-instruct",
    # RouterBench (Martian)
    "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b-v1.2",
    "claude-instant-v1": "claude-instant-v1",
    "claude-v1": "claude-v1",
    "claude-v2": "claude-v2",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview": "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat": "codellama-34b-instruct",
    "meta/llama-2-70b-chat": "llama-2-70b-chat",
    "mistralai/mistral-7b-chat": "mistral-7b-instruct",
    "mistralai/mixtral-8x7b-chat": "mixtral-8x7b-instruct",
    "zero-one-ai/Yi-34B-Chat": "yi-34b-chat",
}


def canonical(name: str) -> str:
    """Known names map explicitly; unknown ones fall back to a lower-case slug."""
    if name in CANONICAL:
        return CANONICAL[name]
    return name.split("/")[-1].strip().lower().replace("_", "-").replace(" ", "-")
