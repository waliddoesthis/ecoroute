"""Redact-then-route: mask sensitive spans so a cheaper external model can be used.

The masked prompt carries placeholders such as <EMAIL_1>; restore() puts the real values
back into the model's answer. Only use this when the task does not need the real values.
"""

from __future__ import annotations

from collections.abc import Iterable

from ecoroute.confidentiality.rules import Finding


def redact(text: str, findings: Iterable[Finding]) -> tuple[str, dict[str, str]]:
    spans = sorted((f for f in findings if f.end > f.start), key=lambda f: f.start)
    mapping: dict[str, str] = {}
    seen: dict[str, str] = {}
    counters: dict[str, int] = {}
    out, pos = [], 0
    for f in spans:
        if f.start < pos:
            continue
        value = text[f.start : f.end]
        if value not in seen:
            tag = f.kind.split(":")[0].upper()
            counters[tag] = counters.get(tag, 0) + 1
            seen[value] = f"<{tag}_{counters[tag]}>"
            mapping[seen[value]] = value
        out.append(text[pos : f.start])
        out.append(seen[value])
        pos = f.end
    out.append(text[pos:])
    return "".join(out), mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text
