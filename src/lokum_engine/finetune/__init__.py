"""Finetune helpers + exports."""

from __future__ import annotations

from functools import partial

from lokum_engine.finetune.engine import FinetuneEngine
from lokum_engine.finetune.engine import get_finetune_quality_profile, normalize_finetune_quality


def finetune_engine_base(model_path: str) -> FinetuneEngine:
    return FinetuneEngine(model_path=model_path, quality="base")


def finetune_engine_mid(model_path: str) -> FinetuneEngine:
    return FinetuneEngine(model_path=model_path, quality="mid")


def finetune_engine_fab(model_path: str) -> FinetuneEngine:
    return FinetuneEngine(model_path=model_path, quality="fab")


finetune_engine_fabulous = finetune_engine_fab
finetune_engine_faboulous = finetune_engine_fab


# Constructor gibi
FinetuneEngineBase = partial(FinetuneEngine, quality="base")
FinetuneEngineMid = partial(FinetuneEngine, quality="mid")
FinetuneEngineFab = partial(FinetuneEngine, quality="fab")


__all__ = [
    "FinetuneEngine",
    "normalize_finetune_quality",
    "get_finetune_quality_profile",
    "finetune_engine_base",
    "finetune_engine_mid",
    "finetune_engine_fab",
    "finetune_engine_fabulous",
    "finetune_engine_faboulous",
    "FinetuneEngineBase",
    "FinetuneEngineMid",
    "FinetuneEngineFab",
]
