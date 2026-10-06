"""One object that turns a prompt into a routing decision: embed, predict, decide.

    eco = EcoRoute.load("artifacts/router.pt", catalog="configs/models.yaml")
    decision = eco.route("Summarise this email ...", context={"sensitivity_label": "Internal"})
    decision.model, decision.explain(), decision.headers()

The gateway and the Python SDK both sit on top of this.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml

from ecoroute.predictors import CatalogPredictor, Predictor
from ecoroute.routing import Decision, Policy, Router


class EcoRoute:
    def __init__(
        self,
        predictor: Predictor,
        encoder: str,
        catalog: list[Mapping],
        policy: str | Policy = "balanced",
        router: Router | None = None,
    ) -> None:
        self.encoder_name = encoder
        self.catalog = [m for m in catalog if m.get("enabled", True)]
        self.predictor = CatalogPredictor.from_catalog(predictor, self.catalog)
        self.router = router or Router(self.catalog, policy=policy)
        self._encoder = None

    def save(self, path: str | Path) -> None:
        """Save the trained predictor and encoder name (the catalog stays in YAML)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"predictor": self.predictor.base, "encoder": self.encoder_name}, path)

    @classmethod
    def load(cls, path: str | Path, catalog: str | Path = "configs/models.yaml", **kw) -> EcoRoute:
        # The file holds pickled Python objects: only load files you produced yourself.
        state = torch.load(path, map_location="cpu", weights_only=False)
        models = yaml.safe_load(Path(catalog).read_text())["models"]
        return cls(state["predictor"], state["encoder"], models, **kw)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(
            np.float32
        )

    def predict(self, prompt: str) -> dict[str, float]:
        return self.predictor.predict_one(self.embed([prompt])[0])

    def route(
        self,
        prompt: str,
        context: Mapping | None = None,
        out_tokens: int = 500,
        policy: str | Policy | None = None,
        scan_text: str | None = None,
    ) -> Decision:
        return self.router.decide(
            prompt,
            self.predict(prompt),
            context=context,
            out_tokens=out_tokens,
            policy=policy,
            scan_text=scan_text,
        )
