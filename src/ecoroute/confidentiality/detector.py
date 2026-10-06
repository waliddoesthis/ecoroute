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


class Detector:
    """Any layer can raise the level, none can lower it.

    `floor` is the level every prompt starts at; the default treats unlabelled traffic as
    internal, since it comes from inside a company.
    """

    def __init__(
        self, layers: Iterable[Layer] | None = None, floor: Level | str = Level.INTERNAL
    ) -> None:
        self.layers = list(layers) if layers is not None else [CallerPolicyLayer(), RulesLayer()]
        self.floor = Level.parse(floor)

    def classify(self, text: str, context: Mapping | None = None) -> Classification:
        findings: list[Finding] = []
        for layer in self.layers:
            findings.extend(layer.scan(text, context))
        level = max([self.floor, *(f.level for f in findings)])
        return Classification(level, findings)
