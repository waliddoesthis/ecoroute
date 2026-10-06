"""Confidentiality levels and model clearance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import IntEnum
from typing import Any


class Level(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def parse(cls, value: str | int | Level) -> Level:
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[str(value).strip().upper()]
        except KeyError:
            raise ValueError(f"unknown confidentiality level: {value!r}") from None

    def __str__(self) -> str:
        return self.name.lower()


def allowed_models(level: Level, catalog: Iterable[Mapping[str, Any]]) -> list[str]:
    """Names of the models whose clearance covers `level`.

    A model with no clearance is treated as public-only, so a missing field can never
    widen what a model may receive.
    """
    return [m["name"] for m in catalog if Level.parse(m.get("clearance", "public")) >= level]
