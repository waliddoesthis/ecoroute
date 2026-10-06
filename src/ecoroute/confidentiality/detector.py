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

    Weak findings (a lone first name or city) count only when the prompt also holds a
    second kind of personal data, e.g. a full name, or a name with an email address.
    Otherwise word problems and travel questions would all look confidential.
    """

    def __init__(
        self,
        layers: Iterable[Layer] | None = None,
        floor: Level | str = Level.INTERNAL,
        combine_weak: bool = True,
    ) -> None:
        self.layers = list(layers) if layers is not None else [CallerPolicyLayer(), RulesLayer()]
        self.floor = Level.parse(floor)
        self.combine_weak = combine_weak

    def classify(self, text: str, context: Mapping | None = None) -> Classification:
        findings: list[Finding] = []
        for layer in self.layers:
            findings.extend(layer.scan(text, context))
        personal = {f.kind for f in findings if f.end > f.start}  # content, not caller policy
        weak_counts = not self.combine_weak or len(personal) >= 2
        levels = [f.level for f in findings if weak_counts or not f.weak]
        return Classification(max([self.floor, *levels]), findings)
