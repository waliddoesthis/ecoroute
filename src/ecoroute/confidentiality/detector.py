"""Run the layers and take the strictest level."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ecoroute.confidentiality.levels import Level
from ecoroute.confidentiality.policy import CallerPolicyLayer
from ecoroute.confidentiality.rules import Finding, RulesLayer


class Layer(Protocol):
    name: str

    def scan(self, text: str, context: Mapping | None = None) -> list[Finding]: ...


@dataclass
class Classification:
    level: Level
    findings: list[Finding] = field(default_factory=list)

    def reasons(self) -> list[str]:
        """Human-readable lines for the explanation, strictest first (no matched text)."""
        ordered = sorted(self.findings, key=lambda f: -f.level)
        return [f"{f.layer}: {f.kind} ({f.level})" for f in ordered]


def combined_level(
    findings: Iterable[Finding], floor: Level, weak_kinds: frozenset[str] = frozenset()
) -> Level:
    """Strictest level among the findings, where findings of a weak kind count only when
    the text also holds a second kind of personal data (a full name, a name and an email).
    Caller-policy findings (no span in the text) are not personal content."""
    findings = list(findings)
    personal = {f.kind for f in findings if f.end > f.start}
    weak_counts = len(personal) >= 2
    levels = [f.level for f in findings if weak_counts or f.kind not in weak_kinds]
    return max([floor, *levels])


class Detector:
    """Any layer can raise the level, none can lower it.

    `floor` is the level every prompt starts at; the default treats unlabelled traffic as
    internal, since it comes from inside a company. `weak_kinds` (e.g. ner.WEAK_KINDS)
    trades recall for fewer false alarms: those kinds only count next to other personal
    data. It is empty by default, because a missed leak costs more than a false alarm.
    """

    def __init__(
        self,
        layers: Iterable[Layer] | None = None,
        floor: Level | str = Level.INTERNAL,
        weak_kinds: Iterable[str] = (),
    ) -> None:
        self.layers = list(layers) if layers is not None else [CallerPolicyLayer(), RulesLayer()]
        self.floor = Level.parse(floor)
        self.weak_kinds = frozenset(weak_kinds)

    def classify(self, text: str, context: Mapping | None = None) -> Classification:
        findings: list[Finding] = []
        for layer in self.layers:
            findings.extend(layer.scan(text, context))
        return Classification(combined_level(findings, self.floor, self.weak_kinds), findings)
