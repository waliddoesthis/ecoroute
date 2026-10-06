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


def hf_tagger(
    model: str = DEFAULT_MODEL, device: int | str | None = None, fp16: bool = False
) -> Tagger:
    """device: None for CPU, or a GPU index such as 0. fp16 halves the weights on a GPU,
    which is usually faster with no loss for this kind of model; measure it with
    scripts/eval_confidentiality.py --device 0 --fp16.

    8-bit dynamic quantization on CPU was tried (run 11) and broke this DeBERTa model's
    predictions without making it faster, so it is not offered."""
    from transformers import pipeline

    kw = {}
    if fp16:
        if device is None:
            raise ValueError("fp16 needs a GPU device")
        import torch

        import transformers
        from packaging.version import Version

        # transformers 4.56 renamed torch_dtype to dtype and warns on the old name.
        new = Version(transformers.__version__) >= Version("4.56")
        kw["dtype" if new else "torch_dtype"] = torch.float16
    return pipeline(
        "token-classification", model=model, aggregation_strategy="simple", device=device, **kw
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
        batch_size: int = 16,
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
        # Windows (and, in scan_many, prompts) go to a Hugging Face pipeline in batches:
        # on a GPU the per-call overhead costs more than the model itself.
        self.batch_size = batch_size

    @property
    def tagger(self) -> Tagger:
        if self._tagger is None:  # loaded on first use so importing stays cheap
            self._tagger = hf_tagger()
        return self._tagger

    def windows(self, text: str) -> list[int]:
        """Start offsets of the windows covering the whole text."""
        step = self.window - self.overlap
        return list(range(0, max(1, len(text) - self.overlap), step))

    def _tag(self, chunks: list[str]) -> list[list[Mapping]]:
        tagger = self.tagger
        if len(chunks) > 1 and self.batch_size > 1 and hasattr(tagger, "tokenizer"):
            return list(tagger(chunks, batch_size=self.batch_size))
        return [tagger(c) for c in chunks]

    def scan(self, text: str, context: Mapping | None = None) -> list[Finding]:
        return self.scan_many([text])[0]

    def scan_many(self, texts: list[str]) -> list[list[Finding]]:
        """Scan several texts with one batched pass over all their windows."""
        jobs = [
            (i, offset)
            for i, text in enumerate(texts)
            if text.strip()
            for offset in self.windows(text)
        ]
        tagged = self._tag([texts[i][o : o + self.window] for i, o in jobs]) if jobs else []
        per_text: list[list[tuple[int, list[Mapping]]]] = [[] for _ in texts]
        for (i, offset), ents in zip(jobs, tagged, strict=True):
            per_text[i].append((offset, ents))
        return [self._findings(windows) for windows in per_text]

    def _findings(self, windows: list[tuple[int, list[Mapping]]]) -> list[Finding]:
        found: dict[tuple[str, int, int], Finding] = {}
        for offset, ents in windows:
            for ent in ents:
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
