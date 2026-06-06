from __future__ import annotations

from functools import partial

from lokum_engine.rag.engine import RAGEngine
from lokum_engine.rag import rag_engine_base, rag_engine_fab, rag_engine_mid
from lokum_engine.rag import rag_reader_engine
from lokum_engine.rag.reader_engine import RAGReaderEngine
from lokum_engine.finetune.engine import FinetuneEngine
from lokum_engine.finetune import finetune_engine_base, finetune_engine_fab, finetune_engine_mid

# “Tek satır import + constructor gibi kullanım” için:
#   from lokum_engine import RAGEngineFab
#   rag = RAGEngineFab(storage_dir="...", ...)
RAGEngineBase = partial(RAGEngine, quality="base")
RAGEngineMid = partial(RAGEngine, quality="mid")
RAGEngineFab = partial(RAGEngine, quality="fab")

#   from lokum_engine import FinetuneEngineFab
#   ft = FinetuneEngineFab(model_path="...")
FinetuneEngineBase = partial(FinetuneEngine, quality="base")
FinetuneEngineMid = partial(FinetuneEngine, quality="mid")
FinetuneEngineFab = partial(FinetuneEngine, quality="fab")

__all__ = [
    "RAGEngine",
    "RAGReaderEngine",
    "FinetuneEngine",
    "rag_engine_base",
    "rag_engine_mid",
    "rag_engine_fab",
    "rag_reader_engine",
    "RAGEngineBase",
    "RAGEngineMid",
    "RAGEngineFab",
    "finetune_engine_base",
    "finetune_engine_mid",
    "finetune_engine_fab",
    "FinetuneEngineBase",
    "FinetuneEngineMid",
    "FinetuneEngineFab",
]
