"""Confidentiality: decide how sensitive a prompt is, before any model sees it.

Layers each propose a level and the strictest one wins (section 2a of the design doc):
L0 caller policy, L1 deterministic rules; L2 (PII NER) and L3 (semantic) plug in through
the same Layer interface.
"""

from ecoroute.confidentiality.detector import Classification, Detector, Layer
from ecoroute.confidentiality.levels import Level, allowed_models
from ecoroute.confidentiality.ner import NERLayer
from ecoroute.confidentiality.policy import CallerPolicyLayer
from ecoroute.confidentiality.redact import redact, restore
from ecoroute.confidentiality.rules import Finding, RulesLayer

__all__ = [
    "CallerPolicyLayer",
    "Classification",
    "Detector",
    "Finding",
    "Layer",
    "Level",
    "NERLayer",
    "RulesLayer",
    "allowed_models",
    "redact",
    "restore",
]
