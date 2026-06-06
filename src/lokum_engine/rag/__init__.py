"""RAG (Retrieval-Augmented Generation) package."""

from __future__ import annotations

from functools import partial

from lokum_engine.rag.engine import RAGEngine
from lokum_engine.rag.engine import get_rag_quality_profile, normalize_rag_quality
from lokum_engine.rag.reader_engine import RAGReaderEngine


def rag_engine_base(storage_dir: str | None = None) -> RAGEngine:
    return RAGEngine(storage_dir=storage_dir, quality="base")


def rag_engine_mid(storage_dir: str | None = None) -> RAGEngine:
    return RAGEngine(storage_dir=storage_dir, quality="mid")


def rag_engine_fab(storage_dir: str | None = None) -> RAGEngine:
    return RAGEngine(storage_dir=storage_dir, quality="fab")


# Kullanıcının yazım varyasyonları için alias’lar
rag_engine_fabulous = rag_engine_fab
rag_engine_faboulous = rag_engine_fab

# Constructor gibi kullanmak istersen:
#   from lokum_engine.rag import RAGEngineFab
#   rag = RAGEngineFab(storage_dir="...")
RAGEngineBase = partial(RAGEngine, quality="base")
RAGEngineMid = partial(RAGEngine, quality="mid")
RAGEngineFab = partial(RAGEngine, quality="fab")


def rag_reader_engine(storage_dir: str, quality: str | None = "mid") -> RAGReaderEngine:
    """
    Reader-only engine:
    - loads an existing RAG store directory
    - provides search() and build_context() for feeding an LLM
    """

    return RAGReaderEngine(storage_dir=storage_dir, quality=quality)


__all__ = [
    "RAGEngine",
    "RAGReaderEngine",
    "normalize_rag_quality",
    "get_rag_quality_profile",
    "rag_engine_base",
    "rag_engine_mid",
    "rag_engine_fab",
    "rag_reader_engine",
    "rag_engine_fabulous",
    "rag_engine_faboulous",
    "RAGEngineBase",
    "RAGEngineMid",
    "RAGEngineFab",
]
