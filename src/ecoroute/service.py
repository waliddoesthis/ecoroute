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
        margins: Mapping[str, float] | None = None,
        parallel_privacy: bool = False,
    ) -> None:
        self.encoder_name = encoder
        self.catalog = [m for m in catalog if m.get("enabled", True)]
        self.predictor = CatalogPredictor.from_catalog(predictor, self.catalog)
        # Profile quality floors measured on held-out data when the router was trained.
        self.taus = dict(taus or {})
        self.margins = dict(margins or {})
        self.router = router or Router(
            self.catalog, policy=policy, profiles=tuned_profiles(self.taus, self.margins)
        )
        self._encoder = None
        # Run the privacy check beside the embedding. It only pays when the two models sit
        # on different devices (e.g. PII model on CPU, encoder on GPU): sharing one GPU,
        # they slow each other down and the total got longer than doing them in turn
        # (run 21: 74 ms in parallel against about 52 ms in sequence).
        self.parallel_privacy = parallel_privacy

    def save(self, path: str | Path) -> None:
        """Save the trained predictor, encoder name and tuned profile settings (the catalog
        stays in YAML)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "predictor": self.predictor.base,
            "encoder": self.encoder_name,
            "taus": self.taus,
            "margins": self.margins,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, catalog: str | Path = "configs/models.yaml", **kw) -> EcoRoute:
        # The file holds pickled Python objects: only load files you produced yourself.
        state = torch.load(path, map_location="cpu", weights_only=False)
        models = yaml.safe_load(Path(catalog).read_text())["models"]
        kw.setdefault("taus", state.get("taus"))
        kw.setdefault("margins", state.get("margins"))
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
        text = prompt if scan_text is None else scan_text
        t0 = time.perf_counter()
        if self.parallel_privacy:
            conf = self._pool().submit(_timed, self.router.detector.classify, text, context)
        else:
            conf = _Done(_timed(self.router.detector.classify, text, context))
        t_privacy = time.perf_counter()
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
            "privacy": privacy_ms,  # overlaps embed and predict when parallel_privacy
            "embed": ms(t_privacy, t_embed),
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


class _Done:
    """A finished result with the Future interface, for the sequential path."""

    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value
