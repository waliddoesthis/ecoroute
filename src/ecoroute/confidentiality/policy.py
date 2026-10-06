"""L0: what the calling application knows about the data.

The caller can pass a sensitivity label (for example a Microsoft Purview or Google DLP
label already attached to the source document) and a minimum level for its own traffic.
Labels map to levels through a configurable table; an unknown label counts as
confidential rather than being ignored.
"""

from __future__ import annotations

from collections.abc import Mapping

from ecoroute.confidentiality.levels import Level
from ecoroute.confidentiality.rules import Finding

DEFAULT_LABELS = {
    "public": Level.PUBLIC,
    "general": Level.INTERNAL,
    "internal": Level.INTERNAL,
    "confidential": Level.CONFIDENTIAL,
    "highly confidential": Level.RESTRICTED,
    "restricted": Level.RESTRICTED,
    "secret": Level.RESTRICTED,
}


class CallerPolicyLayer:
    name = "caller_policy"

    def __init__(self, labels: Mapping[str, Level | str] | None = None) -> None:
        table = labels if labels is not None else DEFAULT_LABELS
        self.labels = {k.lower(): Level.parse(v) for k, v in table.items()}

    def scan(self, text: str, context: Mapping | None = None) -> list[Finding]:
        context = context or {}
        out = []
        if "min_level" in context:
            level = Level.parse(context["min_level"])
            out.append(Finding("caller_min_level", level, 0, 0, self.name))
        label = context.get("sensitivity_label")
        if label is not None:
            level = self.labels.get(str(label).lower(), Level.CONFIDENTIAL)
            out.append(Finding(f"label:{label}", level, 0, 0, self.name))
        return out
