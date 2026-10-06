"""L2: a token-classification model for personal data written in free text.

The rules in L1 catch structured values (cards, keys, emails); names, street addresses or
dates of birth have no fixed shape and need a model. The default is a DeBERTa-v3 model
fine-tuned on the ai4privacy PII dataset; any Hugging Face token-classification model
whose labels appear in LABEL_LEVELS can be swapped in.

Following the design rule "uncertain means higher", a span scored between `grey` and
`threshold` is still reported, at confidential, instead of being dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ecoroute.confidentiality.levels import Level
from ecoroute.confidentiality.rules import Finding

DEFAULT_MODEL = "iiiorg/piiranha-v1-detect-personal-information"

C, R = Level.CONFIDENTIAL, Level.RESTRICTED
LABEL_LEVELS: dict[str, Level] = {
    # Identify or locate a person.
    "GIVENNAME": C,
    "SURNAME": C,
    "USERNAME": C,
    "EMAIL": C,
    "TELEPHONENUM": C,
    "STREET": C,
    "BUILDINGNUM": C,
    "CITY": C,
    "ZIPCODE": C,
    "DATEOFBIRTH": C,
    # Credentials and official or financial identifiers.
    "PASSWORD": R,
    "ACCOUNTNUM": R,
    "CREDITCARDNUMBER": R,
    "SOCIALNUM": R,
    "TAXNUM": R,
    "IDCARDNUM": R,
    "DRIVERLICENSENUM": R,
}

# Kinds that rarely identify anyone on their own ("Tom has 3 apples", "flights to Paris").
# Pass them as Detector(weak_kinds=...) to count them only next to other personal data;
# by default every finding counts (recall first).
WEAK_KINDS = frozenset({"givenname", "surname", "city", "zipcode", "buildingnum", "username"})

# A callable with the Hugging Face pipeline output shape:
# text -> [{"entity_group": str, "score": float, "start": int, "end": int}, ...]
Tagger = Callable[[str], list[Mapping]]


def hf_tagger(model: str = DEFAULT_MODEL, device: int | str | None = None) -> Tagger:
    from transformers import pipeline

    return pipeline(
        "token-classification", model=model, aggregation_strategy="simple", device=device
    )


class NERLayer:
    name = "pii_ner"

    def __init__(
        self,
        tagger: Tagger | None = None,
        threshold: float = 0.5,
        grey: float = 0.3,
        label_levels: Mapping[str, Level] = LABEL_LEVELS,
        window: int = 1500,
        overlap: int = 200,
    ) -> None:
        self._tagger = tagger
        self.threshold = threshold
        self.grey = grey
        self.label_levels = dict(label_levels)
        # Long prompts are scanned in overlapping character windows: the model's memory
        # grows with the square of the input length, and a value cut by one window's edge
        # is whole in the next.
        if overlap >= window:
            raise ValueError("overlap must be smaller than window")
        self.window = window
        self.overlap = overlap

    @property
    def tagger(self) -> Tagger:
        if self._tagger is None:  # loaded on first use so importing stays cheap
            self._tagger = hf_tagger()
        return self._tagger

    def windows(self, text: str) -> list[int]:
        """Start offsets of the windows covering the whole text."""
        step = self.window - self.overlap
        return list(range(0, max(1, len(text) - self.overlap), step))

    def scan(self, text: str, context: Mapping | None = None) -> list[Finding]:
        if not text.strip():
            return []
        found: dict[tuple[str, int, int], Finding] = {}
        for offset in self.windows(text):
            for ent in self.tagger(text[offset : offset + self.window]):
                label = str(ent.get("entity_group") or ent.get("entity", "")).upper()
                label = label.removeprefix("B-").removeprefix("I-")
                level = self.label_levels.get(label)
                score = float(ent["score"])
                if level is None or score < self.grey:
                    continue
                if score < self.threshold:
                    level = Level.CONFIDENTIAL
                start, end = offset + int(ent["start"]), offset + int(ent["end"])
                key = (label, start, end)
                if key not in found or score > found[key].score:  # overlap seen twice
                    found[key] = Finding(label.lower(), level, start, end, self.name, score)
        return sorted(found.values(), key=lambda f: f.start)
