"""
RAGReaderEngine
==============

Goal: load an existing LokumAI RAG store directory and provide a *simple*
"give me chunks / give me context for LLM" API.

This is intentionally a *reader*:
- It does NOT ingest documents by default.
- It expects the persistent store files to already exist under `storage_dir`:
  - faiss_index.bin
  - docs_metadata.npy
  - chunks_meta.npy (optional but recommended)
  - rag_state.json / rag_meta.json (optional)

Quality:
- Uses the same base/mid/fab profiles as `RAGEngine` (embedding model + retrieval policy).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

SentenceTransformer = None
HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

from lokum_engine.rag.engine import (  # re-use the canonical store naming + profiles
    DEFAULT_CHUNKS_META_NAME,
    DEFAULT_DOCS_NAME,
    DEFAULT_INDEX_NAME,
    DEFAULT_META_NAME,
    DEFAULT_STATE_NAME,
    get_rag_quality_profile,
)


class RAGReaderEngine:
    """
    Lightweight RAG store reader + retriever.

    Typical usage:
        from lokum_engine import RAGReaderEngine
        rr = RAGReaderEngine(storage_dir="~/.lokumai/rag", quality="mid")
        rr.load()
        ctx = rr.build_context("my question", k=5)
    """

    def __init__(self, storage_dir: str, quality: str | None = "mid"):
        storage_dir = str(storage_dir or "").strip()
        if not storage_dir:
            raise ValueError("storage_dir is required (path to an existing RAG store).")

        self.storage_dir = os.path.abspath(os.path.expanduser(storage_dir))
        self.quality_profile = get_rag_quality_profile(quality)

        # Store file paths
        self.index_path = os.path.join(self.storage_dir, DEFAULT_INDEX_NAME)
        self.docs_path = os.path.join(self.storage_dir, DEFAULT_DOCS_NAME)
        self.meta_path = os.path.join(self.storage_dir, DEFAULT_META_NAME)
        self.chunks_meta_path = os.path.join(self.storage_dir, DEFAULT_CHUNKS_META_NAME)
        self.state_path = os.path.join(self.storage_dir, DEFAULT_STATE_NAME)

        # Retrieval policy from profile
        self.fetch_multiplier = int(self.quality_profile.fetch_multiplier)
        self.fetch_min = int(self.quality_profile.fetch_min)
        self.fetch_cap = int(self.quality_profile.fetch_cap)

        # Runtime state
        self.index: Optional[Any] = None
        self.documents: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.last_error: str = ""
        self.loaded: bool = False

        # Dependencies for retrieval (lazy init for SentenceTransformer)
        self.enabled = bool(HAS_FAISS)

    def _set_last_error(self, msg: str) -> None:
        try:
            self.last_error = (msg or "").strip()
        except Exception:
            pass

    def _ensure_sentence_transformer(self) -> bool:
        global SentenceTransformer, HAS_SENTENCE_TRANSFORMERS
        if HAS_SENTENCE_TRANSFORMERS and SentenceTransformer is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer

            SentenceTransformer = _SentenceTransformer
            HAS_SENTENCE_TRANSFORMERS = True
            return True
        except Exception as e:
            self._set_last_error(f"sentence-transformers not available: {e}")
            HAS_SENTENCE_TRANSFORMERS = False
            SentenceTransformer = None
            return False

    def _fetch_k(self, k: int) -> int:
        fk = int(max(int(k) * int(self.fetch_multiplier), int(self.fetch_min)))
        if fk > int(self.fetch_cap):
            fk = int(self.fetch_cap)
        return fk

    def load(self) -> None:
        """
        Load store files into memory.

        Raises:
            FileNotFoundError: if required files are missing.
        """
        self.loaded = False

        if not os.path.isdir(self.storage_dir):
            raise FileNotFoundError(f"RAG store directory not found: {self.storage_dir}")

        if not os.path.isfile(self.docs_path):
            raise FileNotFoundError(
                f"docs file not found: {self.docs_path}. You need to build the store first (RAGEngine.ingest_*)."
            )

        # docs (required)
        self.documents = np.load(self.docs_path, allow_pickle=True).tolist()

        # chunk meta (optional)
        if os.path.isfile(self.chunks_meta_path):
            try:
                self.chunk_meta = np.load(self.chunks_meta_path, allow_pickle=True).tolist()
            except Exception:
                self.chunk_meta = []

        # faiss index (required for retrieval)
        if not HAS_FAISS:
            self.enabled = False
            self.index = None
            self._set_last_error("faiss not available (install faiss-cpu). Retrieval is disabled.")
            self.loaded = True
            return

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(
                f"faiss index not found: {self.index_path}. You need to build the store first (RAGEngine.ingest_*)."
            )

        self.index = faiss.read_index(self.index_path)
        self.enabled = True
        self.loaded = True

    def list_chunks(self, limit: int | None = 50) -> List[str]:
        if not self.loaded:
            self.load()
        if limit is None:
            return list(self.documents)
        return list(self.documents[: max(0, int(limit))])

    def search(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        Returns structured retrieval output suitable for LLM "citations".
        """
        if not self.loaded:
            self.load()
        if not self.enabled or self.index is None:
            return {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0, "error": self.last_error}
        if not self._ensure_sentence_transformer():
            return {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0, "error": self.last_error}

        # Lazy embedding model init (only created when needed)
        if not hasattr(self, "embedding_model") or getattr(self, "embedding_model", None) is None:
            embed_model_name = str(self.quality_profile.embedding_model)
            device = (os.environ.get("LOKUMAI_EMBED_DEVICE") or "").strip().lower() or "cpu"
            try:
                self.embedding_model = SentenceTransformer(embed_model_name, device=device)
            except TypeError:
                self.embedding_model = SentenceTransformer(embed_model_name)

        query_vector = self.embedding_model.encode([query_text])
        query_vector = np.array(query_vector).astype("float32")
        fetch_k = self._fetch_k(int(k))
        distances, indices = self.index.search(query_vector, int(fetch_k))

        results: List[str] = []
        dists: List[float] = []
        sources: List[Dict[str, Any]] = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.documents):
                continue
            results.append(self.documents[int(idx)])
            dists.append(float(dist))
            src: Dict[str, Any] = {}
            if int(idx) < len(self.chunk_meta) and isinstance(self.chunk_meta[int(idx)], dict):
                src = dict(self.chunk_meta[int(idx)])
            sources.append(src)
            if len(results) >= int(k):
                break

        return {
            "context": "\n\n---\n\n".join(results),
            "chunks": results,
            "distances": dists,
            "sources": sources,
            "count": len(results),
        }

    def build_context(self, query_text: str, k: int = 5) -> str:
        """
        Convenience helper: returns a single string to paste into an LLM prompt.
        """
        res = self.search(query_text, k=int(k))
        return str(res.get("context") or "")

    def get_stats(self) -> Dict[str, Any]:
        if not self.loaded:
            try:
                self.load()
            except Exception:
                # allow stats even if store missing
                pass
        try:
            ntotal = int(getattr(self.index, "ntotal", 0)) if self.index is not None else 0
        except Exception:
            ntotal = 0
        return {
            "loaded": bool(self.loaded),
            "enabled": bool(self.enabled),
            "storage_dir": self.storage_dir,
            "chunk_count": len(self.documents),
            "index_ntotal": ntotal,
            "quality": str(getattr(self.quality_profile, "name", "mid")),
            "last_error": str(getattr(self, "last_error", "")),
        }

