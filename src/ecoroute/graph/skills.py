"""Readable skill names for the graph, from the task labels of the outcome data.

Datasets name their tasks by repository path ("lighteval/MATH/all/test"); explanations
need words a person reads at a glance ("math"). Tasks of the same kind share one skill.
"""

from __future__ import annotations

import re

# First matching pattern wins; matched against the lowercased task label.
SKILL_PATTERNS: list[tuple[str, str]] = [
    (r"math|gsm8k|aime", "math"),
    (r"gpqa", "graduate science"),
    (r"musr|bbh|reasoning", "multi-step reasoning"),
    (r"mmlu|arc[-_]|knowledge", "general knowledge"),
    (r"humaneval|mbpp|code|program", "code"),
    (r"ragbench/(cuad)", "reading legal documents"),
    (r"ragbench/(finqa|tatqa)", "reading financial documents"),
    (r"ragbench/(covidqa|pubmedqa)", "reading medical documents"),
    (r"ragbench/(emanual|techqa|delucionqa)", "reading technical manuals"),
    (r"ragbench", "answering from documents"),
    (r"openhermes|chat|alpaca|sharegpt", "general chat"),
    (r"mt[-_]?bench|writing", "writing"),
]


def skill_name(task: str) -> str:
    t = str(task).lower()
    for pattern, name in SKILL_PATTERNS:
        if re.search(pattern, t):
            return name
    return t.rstrip("/").split("/")[-1].replace("_", " ") or "other"
