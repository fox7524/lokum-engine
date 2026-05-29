"""
RAG (Retrieval-Augmented Generation) Engine for LokumAI (lokum-engine package).

This module handles the RAG pipeline:
1. PROCESSING: Load various file formats (PDF, DOCX, Code, ZIM)
2. CHUNKING: Split documents into manageable chunks
3. EMBEDDING: Convert text chunks into vector representations
4. INDEXING: Store vectors in FAISS for fast similarity search
5. RETRIEVAL: Find relevant chunks given a user query
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# FAISS for vector similarity search
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False
    print("Warning: faiss not installed. Run: pip install faiss-cpu")

SentenceTransformer = None
HAS_SENTENCE_TRANSFORMERS = False

# PDF processing library - extracts text from PDF files
try:
    import fitz  # PyMuPDF - reads PDFs and extracts text/images

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

# DOCX processing library - extracts text from Word documents
try:
    import docx  # python-docx - reads .docx files

    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("Warning: python-docx not installed. Run: pip install python-docx")

# Image OCR (optional) - extracts text from images (JPG/PNG/etc.)
try:
    from PIL import Image  # pillow

    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False
    print("Warning: pillow not installed. Run: pip install pillow")

try:
    import pytesseract  # requires system tesseract installed

    HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    HAS_TESSERACT = False
    print(
        "Warning: pytesseract not installed. Run: pip install pytesseract (also requires 'tesseract' installed on your OS)"
    )

try:
    # Centralized path handling + override support (LOKUMAI_HOME / LOKUMAI_RAG_DIR)
    from lokum_engine.paths import ensure_dir as _ensure_dir
    from lokum_engine.paths import rag_dir as _rag_dir

    DEFAULT_RAG_DIR = str(_ensure_dir(_rag_dir()))
except Exception:
    # Fallback (kept for robustness in case lokum_engine.paths is missing)
    DEFAULT_RAG_DIR = os.path.join(os.path.expanduser("~"), ".lokumai", "rag")

DEFAULT_INDEX_NAME = "faiss_index.bin"
DEFAULT_DOCS_NAME = "docs_metadata.npy"
DEFAULT_META_NAME = "rag_meta.json"
DEFAULT_CHUNKS_META_NAME = "chunks_meta.npy"
DEFAULT_STATE_NAME = "rag_state.json"
DEFAULT_STAGING_DIRNAME = "staging"


@dataclass(frozen=True)
class RAGQualityProfile:
    """
    RAG kalite profili.

    Not: Embedding model değişirse FAISS index boyutu/dim değişebilir; bu durumda
    eski index ile devam etmek mümkün olmayabilir. Bu yüzden kalite seçimini
    ideal olarak ilk init sırasında yap.
    """

    name: str
    # Chunking
    chunk_size: int
    overlap: int
    # Embedding
    embedding_model: str
    # Retrieval (fetch policy)
    fetch_multiplier: int
    fetch_min: int
    fetch_cap: int


RAG_QUALITY_PROFILES: Dict[str, RAGQualityProfile] = {
    # Varsayılan davranışa en yakın: hızlı + küçük model
    "base": RAGQualityProfile(
        name="base",
        chunk_size=700,
        overlap=80,
        embedding_model="all-MiniLM-L6-v2",
        fetch_multiplier=6,
        fetch_min=30,
        fetch_cap=200,
    ),
    # Mevcut default’larla uyumlu (800/100, multiplier=10, cap=500)
    "mid": RAGQualityProfile(
        name="mid",
        chunk_size=800,
        overlap=100,
        embedding_model="all-MiniLM-L6-v2",
        fetch_multiplier=10,
        fetch_min=50,
        fetch_cap=500,
    ),
    # Daha fazla recall + daha ağır model (kullanıcı “heavy deps OK” dedi)
    "fab": RAGQualityProfile(
        name="fab",
        chunk_size=1000,
        overlap=150,
        embedding_model="all-mpnet-base-v2",
        fetch_multiplier=20,
        fetch_min=80,
        fetch_cap=1000,
    ),
}


def normalize_rag_quality(value: str | None) -> str:
    """
    Kullanıcıdan gelen kalite string’ini normalize eder.

    Kabul edilen örnekler:
      - base: "base", "low", "fast"
      - mid: "mid", "medium", "med", "default"
      - fab: "fab", "fabulous", "faboulous", "fabolous", "high", "hq"
    """

    v = (value or "").strip().lower()
    if not v:
        return "mid"
    if v in ("base", "low", "fast", "lite", "quick"):
        return "base"
    if v in ("mid", "medium", "med", "default", "normal", "std", "standard"):
        return "mid"
    if v in ("fab", "fabulous", "faboulous", "fabolous", "fabulus", "high", "hq", "best"):
        return "fab"
    # Bilinmeyen değer gelirse: mevcut davranışa en yakın
    return "mid"


def get_rag_quality_profile(value: str | None) -> RAGQualityProfile:
    q = normalize_rag_quality(value)
    return RAG_QUALITY_PROFILES.get(q, RAG_QUALITY_PROFILES["mid"])


class RAGEngine:
    """
    Main RAG engine class. Handles:
    - Loading files in various formats
    - Chunking text into manageable pieces
    - Creating embeddings using sentence-transformers
    - Storing and searching vectors using FAISS

    Streamlined (no LangChain).
    """

    def __init__(self, storage_dir: str | None = None, quality: str | None = None):
        # Check if we have all required dependencies
        global SentenceTransformer, HAS_SENTENCE_TRANSFORMERS
        if not HAS_SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer

                SentenceTransformer = _SentenceTransformer
                HAS_SENTENCE_TRANSFORMERS = True
            except Exception as e:
                HAS_SENTENCE_TRANSFORMERS = False
                SentenceTransformer = None
                print(
                    f"Warning: sentence-transformers not available ({e}). Install: pip install sentence-transformers"
                )

        self.enabled = bool(HAS_SENTENCE_TRANSFORMERS and HAS_FAISS)
        if not self.enabled:
            return

        # ---- Quality profile (chunking + retrieval + embedding model selection)
        env_quality = (os.environ.get("LOKUMAI_RAG_QUALITY") or "").strip()
        self.quality_profile = get_rag_quality_profile(quality or env_quality)
        # Chunk defaults (chunk_text() args verilmezse bunlar kullanılır)
        self.chunk_size = int(self.quality_profile.chunk_size)
        self.chunk_overlap = int(self.quality_profile.overlap)
        # Retrieval defaults (query/query_with_sources fetch policy)
        self.fetch_multiplier = int(self.quality_profile.fetch_multiplier)
        self.fetch_min = int(self.quality_profile.fetch_min)
        self.fetch_cap = int(self.quality_profile.fetch_cap)

        self.storage_dir = os.path.abspath(storage_dir or DEFAULT_RAG_DIR)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, DEFAULT_INDEX_NAME)
        self.docs_path = os.path.join(self.storage_dir, DEFAULT_DOCS_NAME)
        self.meta_path = os.path.join(self.storage_dir, DEFAULT_META_NAME)
        self.chunks_meta_path = os.path.join(self.storage_dir, DEFAULT_CHUNKS_META_NAME)
        self.state_path = os.path.join(self.storage_dir, DEFAULT_STATE_NAME)
        self.staging_dir = os.path.join(self.storage_dir, DEFAULT_STAGING_DIRNAME)
        os.makedirs(self.staging_dir, exist_ok=True)
        self.indexed_folder: str = ""

        self.embed_device = self._select_embed_device()
        self.embed_batch_size = self._select_embed_batch_size(self.embed_device)

        embed_model_name = (os.environ.get("LOKUMAI_EMBED_MODEL") or "").strip()
        if not embed_model_name:
            embed_model_name = str(getattr(self, "quality_profile", None).embedding_model)

        try:
            self.embedding_model = SentenceTransformer(embed_model_name, device=self.embed_device)
        except TypeError:
            # Older sentence-transformers doesn't accept device= in constructor
            self.embedding_model = SentenceTransformer(embed_model_name)
            try:
                if hasattr(self.embedding_model, "to"):
                    self.embedding_model.to(self.embed_device)
            except Exception:
                pass

        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {"version": 1, "files": {}}
        self.last_error: str = ""
        self._abort = False

        self.load_index()
        self._load_state()
        self._validate_or_quarantine_existing_store()

    def _select_embed_device(self) -> str:
        val = (os.environ.get("LOKUMAI_EMBED_DEVICE") or "").strip().lower()
        if val in ("cpu", "mps"):
            return val
        try:
            import torch  # type: ignore

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _select_embed_batch_size(self, device: str) -> int:
        raw = (os.environ.get("LOKUMAI_EMBED_BATCH") or "").strip()
        if raw:
            try:
                v = int(raw)
                if 1 <= v <= 2048:
                    return v
            except Exception:
                pass
        if device == "mps":
            return 256
        return 32

    def _checkpoint_policy(self) -> tuple[int, float]:
        raw_chunks = (os.environ.get("LOKUMAI_RAG_CHECKPOINT_CHUNKS") or "").strip()
        raw_secs = (os.environ.get("LOKUMAI_RAG_CHECKPOINT_SECS") or "").strip()
        chunks_default = 20000 if getattr(self, "embed_device", "cpu") == "mps" else 5000
        secs_default = 120.0 if getattr(self, "embed_device", "cpu") == "mps" else 30.0
        chunks = chunks_default
        secs = secs_default
        if raw_chunks:
            try:
                v = int(raw_chunks)
                if 100 <= v <= 500000:
                    chunks = v
            except Exception:
                pass
        if raw_secs:
            try:
                v = float(raw_secs)
                if 1.0 <= v <= 3600.0:
                    secs = v
            except Exception:
                pass
        return int(chunks), float(secs)

    def request_abort(self) -> None:
        try:
            self._abort = True
        except Exception:
            pass

    def clear_abort(self) -> None:
        try:
            self._abort = False
        except Exception:
            pass

    def _check_abort(self) -> None:
        if bool(getattr(self, "_abort", False)):
            raise RuntimeError("RAG operation aborted")

    def _file_id_for(self, path: str) -> str:
        p = os.path.abspath(path or "")
        return hashlib.sha256(p.encode("utf-8", errors="ignore")).hexdigest()

    def mark_deleted(self, source_path: str, deleted: bool = True) -> bool:
        self._load_state()
        if not isinstance(self.state, dict) or not isinstance(self.state.get("files"), dict):
            self.state = {"version": 1, "files": {}}
        p = os.path.abspath(source_path or "")
        if not p:
            return False
        fid = self._file_id_for(p)
        rec = self.state["files"].get(fid)
        if not isinstance(rec, dict):
            rec = {"source_path": p}
            self.state["files"][fid] = rec
        rec["deleted"] = bool(deleted)
        rec["deleted_at"] = time.time() if deleted else None
        try:
            self._atomic_write_json(self.state_path, self.state)
        except Exception:
            pass
        return True

    def _is_file_deleted(self, file_id: str) -> bool:
        try:
            files = (self.state or {}).get("files") if isinstance(self.state, dict) else None
            rec = files.get(file_id) if isinstance(files, dict) else None
            return bool(isinstance(rec, dict) and rec.get("deleted"))
        except Exception:
            return False

    def _set_last_error(self, msg: str) -> None:
        try:
            self.last_error = (msg or "").strip()
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    files = obj.get("files")
                    if isinstance(files, dict):
                        self.state = obj
                        return
        except Exception:
            pass
        self.state = {"version": 1, "files": {}}

    def _atomic_write_json(self, path: str, obj: Any) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _atomic_write_npy(self, path: str, arr: Any) -> None:
        tmp_base = f"{path}.tmp"
        np.save(tmp_base, arr)
        tmp_path = tmp_base if tmp_base.endswith(".npy") else f"{tmp_base}.npy"
        if not os.path.exists(tmp_path) and os.path.exists(tmp_base):
            tmp_path = tmp_base
        os.replace(tmp_path, path)

    def _atomic_write_faiss(self, path: str, index_obj: Any) -> None:
        tmp_path = f"{path}.tmp"
        faiss.write_index(index_obj, tmp_path)
        os.replace(tmp_path, path)

    def validate_store(self) -> Dict[str, Any]:
        ok = True
        problems: List[str] = []

        if self.index is None:
            if self.documents or self.chunk_meta:
                ok = False
                problems.append("Index missing but documents/meta are not empty.")
        else:
            try:
                ntotal = int(getattr(self.index, "ntotal", 0))
            except Exception:
                ntotal = 0
            if ntotal != len(self.documents):
                ok = False
                problems.append(f"FAISS ntotal={ntotal} does not match documents={len(self.documents)}.")
            if self.chunk_meta and len(self.chunk_meta) != len(self.documents):
                ok = False
                problems.append(
                    f"chunks_meta={len(self.chunk_meta)} does not match documents={len(self.documents)}."
                )

        st = self.state if isinstance(self.state, dict) else {}
        files = st.get("files") if isinstance(st, dict) else None
        if isinstance(files, dict) and self.documents:
            for fid, rec in list(files.items())[:5000]:
                if not isinstance(rec, dict):
                    continue
                cs = rec.get("chunk_start")
                ce = rec.get("chunk_end")
                if cs is None or ce is None:
                    continue
                try:
                    cs_i = int(cs)
                    ce_i = int(ce)
                except Exception:
                    ok = False
                    problems.append(f"Invalid chunk range for {fid}.")
                    continue
                if cs_i < 0 or ce_i < cs_i or ce_i > len(self.documents):
                    ok = False
                    problems.append(f"Out-of-bounds chunk range for {fid}: [{cs_i},{ce_i}).")

        return {"ok": ok, "problems": problems}

    def _quarantine_store_files(self, reason: str) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        suffix = f".corrupt.{ts}"
        for p in (self.index_path, self.docs_path, self.chunks_meta_path, self.meta_path, self.state_path):
            try:
                if os.path.exists(p):
                    os.replace(p, p + suffix)
            except Exception:
                pass
        self.index = None
        self.documents = []
        self.chunk_meta = []
        self.indexed_folder = ""
        self.state = {"version": 1, "files": {}}
        self._set_last_error(f"RAG store was quarantined: {reason}")

    def _validate_or_quarantine_existing_store(self) -> None:
        try:
            res = self.validate_store()
            if not res.get("ok", True):
                self._quarantine_store_files(" | ".join(res.get("problems") or [])[:300])
        except Exception:
            pass

    def load_index(self) -> None:
        meta_folder = ""
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                meta_folder = str(meta.get("folder") or "").strip()
        except Exception:
            meta_folder = ""

        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.documents = np.load(self.docs_path, allow_pickle=True).tolist()
                if os.path.exists(self.chunks_meta_path):
                    try:
                        self.chunk_meta = np.load(self.chunks_meta_path, allow_pickle=True).tolist()
                    except Exception:
                        self.chunk_meta = []
                self.indexed_folder = meta_folder
                print(f"[RAG] Loaded index with {len(self.documents)} chunks.")
            except Exception as e:
                self._quarantine_store_files(f"load_index failed: {e}")
                print(f"[RAG] Error loading index: {e}")

    def save_index(self) -> None:
        if self.index is not None:
            self._atomic_write_faiss(self.index_path, self.index)
            self._atomic_write_npy(self.docs_path, np.array(self.documents, dtype=object))
            if self.chunk_meta:
                self._atomic_write_npy(self.chunks_meta_path, np.array(self.chunk_meta, dtype=object))
            try:
                self._atomic_write_json(self.meta_path, {"folder": self.indexed_folder})
            except Exception:
                pass
            try:
                if isinstance(self.state, dict):
                    self._atomic_write_json(self.state_path, self.state)
            except Exception:
                pass
            print(f"[RAG] Saved {len(self.documents)} chunks to index.")

    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
        s = (text or "").strip()
        if not s:
            return []

        # Eğer argüman verilmezse kalite profili default’larını kullan.
        if chunk_size is None:
            chunk_size = int(getattr(self, "chunk_size", 800))
        if overlap is None:
            overlap = int(getattr(self, "chunk_overlap", 100))

        chunk_size = max(1, int(chunk_size))
        overlap = max(0, int(overlap))
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 4)

        step = chunk_size - overlap
        if step <= 0:
            step = max(1, chunk_size)

        chunks: List[str] = []
        for i in range(0, len(s), step):
            chunk = s[i : i + chunk_size]
            if chunk:
                chunks.append(chunk.strip())
        return chunks

    def extract_from_pdf(self, file_path: str) -> str:
        if not HAS_PYMUPDF:
            return ""
        try:
            text_parts = []
            doc = fitz.open(file_path)
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n\n--- Page Break ---\n\n".join(text_parts)
        except Exception as e:
            print(f"[RAG] PDF extraction error for {file_path}: {e}")
            return ""

    def extract_from_docx(self, file_path: str) -> str:
        if not HAS_DOCX:
            return ""
        try:
            doc = docx.Document(file_path)
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            return "\n\n".join(paragraphs)
        except Exception as e:
            print(f"[RAG] DOCX extraction error for {file_path}: {e}")
            return ""

    def extract_from_zim(self, file_path: str) -> str:
        """
        Extract text content from a ZIM archive.

        Tries libzim first (preferred), then pyzim.
        """
        text_parts = []
        try:
            pyzim_err = None
            pyzim_mod = None
            try:
                import pyzim as _pyzim  # published on PyPI as python-zim

                pyzim_mod = _pyzim
            except Exception as e:
                pyzim_err = str(e)

            libzim_err = None
            LibZimArchive = None
            try:
                from libzim.reader import Archive as _LibZimArchive

                LibZimArchive = _LibZimArchive
            except Exception as e:
                libzim_err = str(e)

            if LibZimArchive is not None:
                try:
                    zf = LibZimArchive(file_path)
                    scanned = 0
                    kept = 0
                    skipped_ns = 0
                    skipped_nontext = 0
                    skipped_resource = 0
                    read_fail = 0

                    def is_article_entry(entry) -> bool:
                        ns = getattr(entry, "namespace", None)
                        if isinstance(ns, str) and ns:
                            return ns.upper() in ("A", "C")
                        return True

                    it = None
                    try:
                        if hasattr(zf, "iterByPath"):
                            it = zf.iterByPath()
                        elif hasattr(zf, "iter_by_path"):
                            it = zf.iter_by_path()
                        elif hasattr(zf, "iterByUrl"):
                            it = zf.iterByUrl()
                        elif hasattr(zf, "iter_by_url"):
                            it = zf.iter_by_url()
                        elif hasattr(zf, "iter_entries"):
                            it = zf.iter_entries()
                        elif hasattr(zf, "entries"):
                            it = getattr(zf, "entries")
                            if callable(it):
                                it = it()
                        if it is not None:
                            iter(it)
                    except Exception:
                        it = None

                    def iter_entries():
                        nonlocal scanned, read_fail
                        if it is not None:
                            try:
                                it_iter = iter(it)
                                first = next(it_iter, None)
                                if first is not None:
                                    yield first
                                    for entry in it_iter:
                                        yield entry
                                    return
                            except Exception:
                                pass

                        n = getattr(zf, "entry_count", None)
                        if callable(n):
                            n = n()
                        if not isinstance(n, int):
                            n = getattr(zf, "article_count", None)
                            if callable(n):
                                n = n()
                        if not isinstance(n, int):
                            n = 0
                        max_scan = min(max(500, n), 20000) if n > 0 else 20000
                        for i in range(max_scan):
                            try:
                                entry = None
                                getter = None
                                for name in (
                                    "get_entry_by_id",
                                    "getEntryById",
                                    "_get_entry_by_id",
                                    "get_entry",
                                    "getEntry",
                                    "get_article_by_id",
                                    "getArticleById",
                                    "get_article",
                                    "getArticle",
                                ):
                                    if hasattr(zf, name):
                                        getter = getattr(zf, name)
                                        break
                                if callable(getter):
                                    entry = getter(i)
                                if entry is None:
                                    continue
                                yield entry
                            except Exception:
                                read_fail += 1
                                continue

                    for entry in iter_entries():
                        try:
                            scanned += 1
                            title = (getattr(entry, "title", "") or "").strip()
                            if not title:
                                continue
                            if not is_article_entry(entry):
                                skipped_ns += 1
                                continue
                            mimetype = (getattr(entry, "mimetype", "") or "").lower()
                            if title.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")):
                                skipped_resource += 1
                                continue
                            if mimetype and not mimetype.startswith("text/") and "xml" not in mimetype and "html" not in mimetype:
                                skipped_nontext += 1
                                continue
                            content = ""
                            try:
                                if hasattr(entry, "read"):
                                    raw = entry.read()
                                    if isinstance(raw, bytes):
                                        content = raw.decode("utf-8", errors="ignore").strip()
                                    else:
                                        content = str(raw or "").strip()
                                else:
                                    item = entry.get_item() if hasattr(entry, "get_item") else None
                                    raw = bytes(item.content) if (item is not None and hasattr(item, "content")) else b""
                                    content = raw.decode("utf-8", errors="ignore").strip()
                            except Exception:
                                read_fail += 1
                                content = ""
                            if content:
                                text_parts.append(f"## {title}\n\n{content}")
                                kept += 1
                            if len(text_parts) >= 200:
                                break
                        except Exception:
                            read_fail += 1
                            continue

                    out = "\n\n".join(text_parts).strip()
                    if out:
                        return out

                    ec = None
                    ac = None
                    aec = None
                    try:
                        ec = getattr(zf, "entry_count", None)
                        if callable(ec):
                            ec = ec()
                    except Exception:
                        ec = None
                    try:
                        ac = getattr(zf, "article_count", None)
                        if callable(ac):
                            ac = ac()
                    except Exception:
                        ac = None
                    try:
                        aec = getattr(zf, "all_entry_count", None)
                        if callable(aec):
                            aec = aec()
                    except Exception:
                        aec = None
                    self._set_last_error(
                        f"ZIM (libzim) extracted 0 text entries (scanned={scanned}, skipped_ns={skipped_ns}, skipped_nontext={skipped_nontext}, skipped_resource={skipped_resource}, read_fail={read_fail}, entry_count={ec}, article_count={ac}, all_entry_count={aec})."
                    )
                    return ""
                except Exception as e:
                    self._set_last_error(f"ZIM (libzim) read failed: {e}")
                    return ""

            if pyzim_mod is not None:
                try:
                    with pyzim_mod.Zim.open(file_path) as zf:
                        it = None
                        if hasattr(zf, "iter_entries"):
                            it = zf.iter_entries()
                        elif hasattr(zf, "iter_content_entries"):
                            it = zf.iter_content_entries()
                        elif hasattr(zf, "entries"):
                            it = getattr(zf, "entries")
                        if it is None:
                            self._set_last_error("ZIM (pyzim) could not iterate entries (unsupported API).")
                            return ""

                        scanned = 0
                        kept = 0
                        skipped_ns = 0
                        skipped_nontext = 0
                        skipped_resource = 0
                        read_fail = 0

                        def is_article_entry(entry) -> bool:
                            ns = getattr(entry, "namespace", None)
                            if isinstance(ns, str) and ns:
                                return ns.upper() in ("A", "C")
                            return True

                        for entry in it:
                            scanned += 1
                            title = (getattr(entry, "title", "") or "").strip()
                            mimetype = (getattr(entry, "mimetype", "") or "").lower()
                            if not title:
                                continue
                            if not is_article_entry(entry):
                                skipped_ns += 1
                                continue
                            if title.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")):
                                skipped_resource += 1
                                continue
                            if mimetype and not mimetype.startswith("text/"):
                                skipped_nontext += 1
                                continue
                            try:
                                content = entry.read()
                                if isinstance(content, bytes):
                                    content = content.decode("utf-8", errors="ignore")
                                content = (content or "").strip()
                            except Exception:
                                read_fail += 1
                                content = ""
                            if content:
                                text_parts.append(f"## {title}\n\n{content}")
                                kept += 1
                            if len(text_parts) >= 200:
                                break
                    out = "\n\n".join(text_parts).strip()
                    if out:
                        return out
                    self._set_last_error(
                        f"ZIM (pyzim) extracted 0 text entries (scanned={scanned}, skipped_ns={skipped_ns}, skipped_nontext={skipped_nontext}, skipped_resource={skipped_resource}, read_fail={read_fail})."
                    )
                    return ""
                except Exception as e:
                    self._set_last_error(f"ZIM (pyzim) read failed: {e}")
                    return ""

            detail = []
            if libzim_err:
                detail.append(f"libzim import error: {libzim_err}")
            if pyzim_err:
                detail.append(f"pyzim import error: {pyzim_err}")
            msg = "ZIM support not available. Install: pip install libzim OR pip install 'python-zim[all]'."
            if detail:
                msg += " " + " | ".join(detail)
            self._set_last_error(msg)
            return ""
        except Exception as e:
            self._set_last_error(f"ZIM extraction error: {e}")
            print(f"[RAG] ZIM extraction error for {file_path}: {e}")
            return ""

    def extract_from_image(self, file_path: str) -> str:
        if not (HAS_PIL and HAS_TESSERACT):
            return ""
        try:
            img = Image.open(file_path)
            txt = pytesseract.image_to_string(img)
            return (txt or "").strip()
        except Exception as e:
            print(f"[RAG] Image OCR error for {file_path}: {e}")
            return ""

    def extract_from_code(self, file_path: str) -> str:
        try:
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            print(f"[RAG] Could not decode file: {file_path}")
            return ""
        except Exception as e:
            print(f"[RAG] Code extraction error for {file_path}: {e}")
            return ""

    def process_file(self, file_path: str) -> List[str]:
        if not self.enabled:
            return []

        ext = os.path.splitext(file_path)[1].lower()
        self._set_last_error("")
        try:
            if ext == ".pdf":
                content = self.extract_from_pdf(file_path)
            elif ext in [".docx", ".doc"]:
                content = self.extract_from_docx(file_path)
            elif ext == ".zim":
                content = self.extract_from_zim(file_path)
            elif ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
                content = self.extract_from_image(file_path)
            elif ext in [
                ".py",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".js",
                ".ts",
                ".html",
                ".htm",
                ".css",
                ".scss",
                ".sass",
                ".less",
                ".txt",
                ".md",
                ".markdown",
                ".rst",
                ".json",
                ".xml",
                ".yaml",
                ".yml",
                ".toml",
                ".ini",
                ".cfg",
                ".sh",
                ".bash",
                ".zsh",
                ".csh",
                ".ps1",
                ".r",
                ".java",
                ".kt",
                ".swift",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".pl",
                ".pm",
                ".lua",
                ".scala",
                ".clj",
                ".ex",
                ".exs",
                ".sql",
                ".graphql",
                ".gql",
                ".vim",
                ".editorconfig",
                ".gitignore",
                ".dockerfile",
                ".makefile",
                ".cmake",
            ]:
                content = self.extract_from_code(file_path)
            else:
                content = self.extract_from_code(file_path)

            if content:
                return self.chunk_text(content)
            return []
        except Exception as e:
            self._set_last_error(f"Error processing {file_path}: {e}")
            print(f"[RAG] Error processing {file_path}: {e}")
            return []

    def ingest_documents(self, file_paths: List[str]) -> bool:
        if not self.enabled:
            return False
        self._load_state()
        self._validate_or_quarantine_existing_store()
        return bool(self._ingest_paths(file_paths, save_on_checkpoint=True))

    def _ingest_paths(self, file_paths: List[str], save_on_checkpoint: bool = True) -> int:
        self._check_abort()
        added = 0
        failures: List[str] = []
        pending_save = 0
        last_save_at = time.time()
        checkpoint_chunks, checkpoint_secs = self._checkpoint_policy()

        def encode_batch(texts: List[str]):
            try:
                bs = getattr(self, "embed_batch_size", 32)
                return self.embedding_model.encode(texts, batch_size=int(bs), show_progress_bar=False)
            except TypeError:
                return self.embedding_model.encode(texts)

        def should_index(path: str, fid: str) -> bool:
            rec = (self.state.get("files") or {}).get(fid) if isinstance(self.state, dict) else None
            if not isinstance(rec, dict):
                return True
            if rec.get("deleted"):
                return False
            if rec.get("status") != "ok":
                return True
            try:
                st = os.stat(path)
            except Exception:
                return False
            try:
                prev_size = int(rec.get("size", -1))
                prev_mtime = float(rec.get("mtime", -1))
            except Exception:
                return True
            if prev_size == int(st.st_size) and prev_mtime == float(st.st_mtime):
                return False
            return True

        txn_id = f"{int(time.time() * 1000)}"
        txn_dir = os.path.join(self.staging_dir, txn_id)
        try:
            os.makedirs(txn_dir, exist_ok=True)
        except Exception:
            txn_dir = ""

        try:
            for path in file_paths:
                self._check_abort()
                p = os.path.abspath(path or "")
                if not p:
                    continue
                fid = self._file_id_for(p)
                try:
                    st = os.stat(p)
                    size = int(st.st_size)
                    mtime = float(st.st_mtime)
                except Exception:
                    size = -1
                    mtime = -1.0

                if isinstance(self.state, dict) and isinstance(self.state.get("files"), dict):
                    rec = self.state["files"].get(fid)
                    if not isinstance(rec, dict):
                        rec = {}
                        self.state["files"][fid] = rec
                    rec["source_path"] = p
                    rec["last_seen_at"] = time.time()
                    if size >= 0:
                        rec["size"] = size
                    if mtime >= 0:
                        rec["mtime"] = mtime

                if size < 0:
                    continue
                if not should_index(p, fid):
                    continue

                chunks = self.process_file(p)
                if not chunks:
                    ext = os.path.splitext(p)[1].lower()
                    if ext == ".zim":
                        le = getattr(self, "last_error", "") or ""
                        failures.append(f"{os.path.basename(p)}: {le or 'No text extracted from this ZIM.'}")
                    if isinstance(self.state, dict) and isinstance(self.state.get("files"), dict):
                        rec = self.state["files"].get(fid)
                        if isinstance(rec, dict):
                            rec["status"] = "failed"
                            rec["error"] = getattr(self, "last_error", "") or "No content extracted."
                            rec["indexed_at"] = time.time()
                    continue

                ext = os.path.splitext(p)[1].lower()
                max_per_file = 2500 if ext == ".zim" else 1200
                if len(chunks) > max_per_file:
                    chunks = chunks[:max_per_file]

                try:
                    self._check_abort()
                    dim = None
                    if self.index is not None:
                        try:
                            dim = int(getattr(self.index, "d"))
                        except Exception:
                            dim = None

                    start_idx = len(self.documents)
                    bs = int(getattr(self, "embed_batch_size", 32))
                    window = max(512, bs * 16)
                    window = min(window, 8192)
                    for off in range(0, len(chunks), window):
                        self._check_abort()
                        batch = chunks[off : off + window]
                        if not batch:
                            continue
                        emb = encode_batch(batch)
                        emb = np.array(emb).astype("float32")
                        if emb.ndim != 2 or emb.shape[0] != len(batch):
                            raise RuntimeError("Embedding model returned invalid shape.")
                        if dim is None:
                            dim = int(emb.shape[1])
                        if self.index is None:
                            self.index = faiss.IndexFlatL2(int(dim))
                        self.index.add(emb)
                        self.documents.extend(batch)
                        for _ in batch:
                            self.chunk_meta.append({"file_id": fid, "source_path": p})

                    end_idx = len(self.documents)
                    if end_idx <= start_idx:
                        raise RuntimeError("No embeddings created.")

                    if isinstance(self.state, dict) and isinstance(self.state.get("files"), dict):
                        rec = self.state["files"].get(fid)
                        if not isinstance(rec, dict):
                            rec = {}
                            self.state["files"][fid] = rec
                        rec["status"] = "ok"
                        rec["deleted"] = False
                        rec["indexed_at"] = time.time()
                        rec["chunk_start"] = int(start_idx)
                        rec["chunk_end"] = int(end_idx)
                        rec["chunks"] = int(end_idx - start_idx)
                        rec["size"] = size
                        rec["mtime"] = mtime
                        rec.pop("error", None)

                    delta = end_idx - start_idx
                    added += delta
                    pending_save += delta

                    if save_on_checkpoint:
                        now = time.time()
                        if pending_save >= int(checkpoint_chunks) or (now - last_save_at) >= float(checkpoint_secs):
                            self.save_index()
                            pending_save = 0
                            last_save_at = now
                except Exception as e:
                    failures.append(f"{os.path.basename(p)}: {e}")
                    if isinstance(self.state, dict) and isinstance(self.state.get("files"), dict):
                        rec = self.state["files"].get(fid)
                        if isinstance(rec, dict):
                            rec["status"] = "failed"
                            rec["indexed_at"] = time.time()
                            rec["error"] = str(e)
                    continue

            if added <= 0:
                msg = "No content extracted from files."
                if failures:
                    msg += "\n\n" + "\n".join(failures[:6])
                raise RuntimeError(msg)

            if save_on_checkpoint and pending_save > 0:
                self.save_index()
            return int(added)
        finally:
            try:
                if txn_dir and os.path.isdir(txn_dir):
                    for root, dirs, files in os.walk(txn_dir, topdown=False):
                        for fn in files:
                            try:
                                os.remove(os.path.join(root, fn))
                            except Exception:
                                pass
                        for dn in dirs:
                            try:
                                os.rmdir(os.path.join(root, dn))
                            except Exception:
                                pass
                    try:
                        os.rmdir(txn_dir)
                    except Exception:
                        pass
            except Exception:
                pass

    def ingest_folder(self, folder_path: str, recursive: bool = True) -> bool:
        if not os.path.isdir(folder_path):
            print(f"[RAG] Invalid folder: {folder_path}")
            return False

        folder_abs = os.path.abspath(folder_path)
        self.indexed_folder = folder_abs
        self._load_state()
        self._validate_or_quarantine_existing_store()

        exts = {
            ".pdf",
            ".docx",
            ".doc",
            ".zim",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
            ".py",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".js",
            ".ts",
            ".html",
            ".htm",
            ".css",
            ".txt",
            ".md",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".sh",
            ".ino",
        }
        batch: List[str] = []
        seen = 0
        added_total = 0
        batch_size = 2000

        def flush():
            nonlocal batch, added_total
            if not batch:
                return
            added_total += int(self._ingest_paths(batch, save_on_checkpoint=True))
            batch = []

        self._check_abort()
        if recursive:
            for root, _dirs, files in os.walk(folder_abs):
                self._check_abort()
                for fn in files:
                    self._check_abort()
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in exts:
                        continue
                    seen += 1
                    batch.append(os.path.join(root, fn))
                    if len(batch) >= batch_size:
                        flush()
        else:
            try:
                for fn in os.listdir(folder_abs):
                    self._check_abort()
                    p = os.path.join(folder_abs, fn)
                    if not os.path.isfile(p):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in exts:
                        continue
                    seen += 1
                    batch.append(p)
                    if len(batch) >= batch_size:
                        flush()
            except Exception:
                pass

        flush()
        print(f"[RAG] Found {seen} supported files in {folder_abs}")
        return bool(added_total > 0)

    def _fetch_k(self, k: int) -> int:
        mult = int(getattr(self, "fetch_multiplier", 10))
        mn = int(getattr(self, "fetch_min", 50))
        cap = int(getattr(self, "fetch_cap", 500))
        fk = int(max(k * mult, mn))
        if fk > cap:
            fk = cap
        return fk

    def query(self, query_text: str, k: int = 3) -> str:
        if not self.enabled or self.index is None:
            return ""

        try:
            self._check_abort()
            query_vector = self.embedding_model.encode([query_text])
            query_vector = np.array(query_vector).astype("float32")
            self._check_abort()

            fetch_k = self._fetch_k(int(k))
            distances, indices = self.index.search(query_vector, fetch_k)

            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.documents):
                    if idx < len(self.chunk_meta):
                        meta = self.chunk_meta[idx]
                        if isinstance(meta, dict):
                            fid = meta.get("file_id")
                            if isinstance(fid, str) and fid and self._is_file_deleted(fid):
                                continue
                    results.append(self.documents[idx])
                    if len(results) >= int(k):
                        break

            if not results:
                return ""
            return "\n\n---\n\n".join(results)
        except Exception as e:
            if "aborted" in str(e).lower():
                self._set_last_error(str(e))
                try:
                    self.clear_abort()
                except Exception:
                    pass
                return ""
            print(f"[RAG] Query error: {e}")
            return ""

    def query_with_sources(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        if not self.enabled or self.index is None:
            return {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0}

        try:
            self._check_abort()
            query_vector = self.embedding_model.encode([query_text])
            query_vector = np.array(query_vector).astype("float32")
            self._check_abort()

            fetch_k = self._fetch_k(int(k))
            distances, indices = self.index.search(query_vector, fetch_k)

            results = []
            dists = []
            sources = []

            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.documents):
                    results.append(self.documents[idx])
                    dists.append(float(dist))
                    src = {}
                    if idx < len(self.chunk_meta):
                        meta = self.chunk_meta[idx]
                        if isinstance(meta, dict):
                            src = dict(meta)
                    fid = src.get("file_id") if isinstance(src, dict) else None
                    if isinstance(fid, str) and fid and self._is_file_deleted(fid):
                        results.pop()
                        dists.pop()
                        continue
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
        except Exception as e:
            if "aborted" in str(e).lower():
                self._set_last_error(str(e))
                try:
                    self.clear_abort()
                except Exception:
                    pass
                return {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0}
            print(f"[RAG] Query error: {e}")
            return {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0}

    def get_stats(self) -> Dict[str, Any]:
        if self.index is None:
            return {
                "enabled": self.enabled,
                "chunk_count": 0,
                "index_size": 0,
                "embedding_dim": 0,
                "indexed": False,
                "folder": self.indexed_folder,
            }

        return {
            "enabled": self.enabled,
            "chunk_count": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dim": self.index.d if hasattr(self.index, "d") else 384,
            "indexed": True,
            "folder": self.indexed_folder,
        }

    def set_quality(self, quality: str) -> None:
        """
        Runtime’da kalite profilini değiştirir (chunking + retrieval policy).

        Uyarı: embedding_model seçimi __init__ sırasında yapılıyor.
        Kalite profilin embedding_model’ı farklıysa ve bunu da değiştirmek istiyorsan,
        en sağlıklısı yeni RAGEngine oluşturmak ve index’i yeniden üretmek.
        """

        prof = get_rag_quality_profile(quality)
        self.quality_profile = prof
        self.chunk_size = int(prof.chunk_size)
        self.chunk_overlap = int(prof.overlap)
        self.fetch_multiplier = int(prof.fetch_multiplier)
        self.fetch_min = int(prof.fetch_min)
        self.fetch_cap = int(prof.fetch_cap)

    def reset_database(self) -> None:
        for p in (self.index_path, self.docs_path, self.meta_path, self.chunks_meta_path, self.state_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        self.index = None
        self.documents = []
        self.chunk_meta = []
        self.indexed_folder = ""
        self.state = {"version": 1, "files": {}}
        print("[RAG] Index reset. All data cleared.")

    def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        result = self.query_with_sources(query, top_k)
        return result.get("chunks", [])
