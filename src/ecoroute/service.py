"""One object that turns a prompt into a routing decision: embed, predict, decide.

    eco = EcoRoute.load("artifacts/router.pt", catalog="configs/models.yaml")
    decision = eco.route("Summarise this email ...", context={"sensitivity_label": "Internal"})
    decision.model, decision.explain(), decision.headers()

The gateway and the Python SDK both sit on top of this.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ecoroute.predictors import CatalogPredictor, Predictor
from ecoroute.routing import Decision, Policy, Router, tuned_profiles


class EcoRoute:
    def __init__(
        self,
        predictor: Predictor,
        encoder: str,
        catalog: list[Mapping],
        policy: str | Policy = "balanced",
        router: Router | None = None,
        taus: Mapping[str, float] | None = None,
    ) -> None:
        self.encoder_name = encoder
        self.catalog = [m for m in catalog if m.get("enabled", True)]
        self.predictor = CatalogPredictor.from_catalog(predictor, self.catalog)
        # Profile quality floors measured on held-out data when the router was trained.
        self.taus = dict(taus or {})
        self.router = router or Router(
            self.catalog, policy=policy, profiles=tuned_profiles(self.taus)
        )
        self._encoder = None

    def save(self, path: str | Path) -> None:
        """Save the trained predictor, encoder name and tuned taus (the catalog stays in YAML)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {"predictor": self.predictor.base, "encoder": self.encoder_name, "taus": self.taus}
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, catalog: str | Path = "configs/models.yaml", **kw) -> EcoRoute:
        # The file holds pickled Python objects: only load files you produced yourself.
        state = torch.load(path, map_location="cpu", weights_only=False)
        models = yaml.safe_load(Path(catalog).read_text())["models"]
        kw.setdefault("taus", state.get("taus"))
        return cls(state["predictor"], state["encoder"], models, **kw)

    def _pool(self) -> ThreadPoolExecutor:
        # One worker kept for the process: starting a thread per request costs time too.
        if getattr(self, "_executor", None) is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy")
        return self._executor

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(
            np.float32
        )

    def predict(self, prompt: str) -> dict[str, float]:
        return self.predictor.predict_one(self.embed([prompt])[0])

    def graph(self):
        """The SkillGraph behind the predictor, if there is one (possibly calibrated)."""
        from ecoroute.graph import SkillGraph

        p = self.predictor.base
        while p is not None and not isinstance(p, SkillGraph):
            p = getattr(p, "base", None)
        return p

    def route(
        self,
        prompt: str,
        context: Mapping | None = None,
        out_tokens: int = 500,
        policy: str | Policy | None = None,
        scan_text: str | None = None,
    ) -> Decision:
        # The privacy check and the embedding are independent model calls; running them
        # side by side hides most of the privacy check's latency behind the embedding.
        text = prompt if scan_text is None else scan_text
        t0 = time.perf_counter()
        conf = self._pool().submit(_timed, self.router.detector.classify, text, context)
        x = self.embed([prompt])[0]
        t_embed = time.perf_counter()
        p_success = self.predictor.predict_one(x)
        t_predict = time.perf_counter()
        classification, privacy_ms = conf.result()
        t_wait = time.perf_counter()
        decision = self.router.decide(
            prompt,
            p_success,
            context=context,
            out_tokens=out_tokens,
            policy=policy,
            scan_text=scan_text,
            classification=classification,
        )
        t_decide = time.perf_counter()
        graph = self.graph()
        if graph is not None:
            # Show the graph paths behind the choice, through the chosen model's anchor.
            anchor = self.predictor.anchors.get(decision.model)
            lines = graph.explain(x).lines(anchor.model if anchor else None)
            if anchor is not None and len(lines) > 1:
                lines[1] = lines[1].replace(
                    f"to {anchor.model}:", f"to {decision.model} (via {anchor.model}):"
                )
            decision.steps[1:1] = lines
        t_end = time.perf_counter()
        ms = lambda a, b: 1000 * (b - a)  # noqa: E731
        decision.timings_ms = {
            "privacy": privacy_ms,  # runs in parallel with embed and predict
            "embed": ms(t0, t_embed),
            "predict": ms(t_embed, t_predict),
            "privacy_wait": ms(t_predict, t_wait),
            "decide": ms(t_wait, t_decide),
            "explain": ms(t_decide, t_end),
            "total": ms(t0, t_end),
        }
        return decision


def _timed(fn, *args):
    t0 = time.perf_counter()
    return fn(*args), 1000 * (time.perf_counter() - t0)
