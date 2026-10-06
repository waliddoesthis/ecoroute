"""Build the gateway's router from a trained artifact (shared by serve.py and e2e_check.py)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ecoroute.confidentiality import CallerPolicyLayer, Detector, NERLayer, RulesLayer
from ecoroute.gateway.backends import deployable
from ecoroute.routing import Router
from ecoroute.service import EcoRoute


def load_gateway_router(
    router: str | Path,
    catalog: str | Path,
    providers: Mapping[str, Mapping[str, Any]],
    policy: str = "balanced",
    pii_model: bool = True,
    pii_device: int | None = None,
    pii_fp16: bool = False,
    dry_run: bool = False,
    parallel_privacy: bool = False,
) -> EcoRoute:
    """The trained router limited to the models the gateway can call (all enabled catalog
    models on a dry run), with the full privacy check in front."""
    full = EcoRoute.load(router, catalog=catalog)
    live = full.catalog if dry_run else deployable(full.catalog, providers)
    if not live:
        raise SystemExit("no deployable models: set api_id in models.yaml and the API key env vars")
    layers = [CallerPolicyLayer(), RulesLayer()]
    if pii_model:
        from ecoroute.confidentiality.ner import hf_tagger

        layers.append(NERLayer(hf_tagger(device=pii_device, fp16=pii_fp16)))
    # Keep the quality floors and fallback margins the router was trained with.
    gate = Router(live, detector=Detector(layers), policy=policy, profiles=full.router.profiles)
    return EcoRoute(
        full.predictor.base, full.encoder_name, live, router=gate,
        taus=full.taus, margins=full.margins, parallel_privacy=parallel_privacy,
    )  # fmt: skip
