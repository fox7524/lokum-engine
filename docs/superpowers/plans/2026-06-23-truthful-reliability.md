# Truthful Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `lokum-engine` report failures truthfully across RAG and fine-tune flows instead of silently returning empty results or accepting broken inputs.

**Architecture:** Keep the current public surface mostly stable, but harden the engine with explicit error state, structured logging, strict dataset validation, and failure-focused regression tests. Work top-down through the highest-risk paths: RAG query truthfulness, RAG persistence truthfulness, then fine-tune dataset and validation truthfulness.

**Tech Stack:** Python 3.10+, pytest, logging, dataclasses/typing, FAISS, sentence-transformers, MLX LoRA runner

---

## File Structure

- Modify: `src/lokum_engine/rag/engine.py`
  - Add structured logging
  - Normalize `last_error` handling in query paths
  - Surface persistence failures instead of swallowing them
  - Return richer ingest outcome without hiding failures

- Modify: `src/lokum_engine/rag/reader_engine.py`
  - Normalize reader error state and logging
  - Make reader search failures explicit

- Modify: `src/lokum_engine/finetune/engine.py`
  - Add dataset preflight validation
  - Add strict JSONL validation mode
  - Stop swallowing validation preparation failures

- Create: `tests/test_rag_query_truthfulness.py`
  - Query/search failure behavior
  - Reader failure behavior

- Create: `tests/test_rag_persistence_truthfulness.py`
  - Save/delete failure surfacing
  - Ingest outcome behavior

- Create: `tests/test_finetune_validation.py`
  - Dataset preflight behavior
  - Strict malformed JSONL behavior
  - Validation setup failure behavior

- Modify: `tests/test_rag_reader_engine.py`
  - Keep existing reader behavior coverage passing after truthfulness changes

- Modify: `tests/test_finetune_presplit_chatml.py`
  - Keep existing ChatML split behavior coverage passing after strict-mode changes

## Task 1: RAG Query Truthfulness

**Files:**
- Create: `tests/test_rag_query_truthfulness.py`
- Modify: `src/lokum_engine/rag/engine.py`
- Modify: `src/lokum_engine/rag/reader_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import numpy as np

from lokum_engine.rag.engine import RAGEngine
from lokum_engine.rag.reader_engine import RAGReaderEngine


class _ExplodingEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        raise RuntimeError("embedding boom")


class _FakeIndex:
    def search(self, query_vector, fetch_k):
        raise RuntimeError("faiss boom")


class TestRagQueryTruthfulness(unittest.TestCase):
    def test_query_sets_last_error_when_embedding_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.enabled = True
        eng.index = _FakeIndex()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.last_error = ""
        eng.embedding_model = _ExplodingEmbedder()
        eng._abort = False

        result = eng.query("hello", k=1)

        self.assertEqual(result, "")
        self.assertIn("embedding boom", eng.last_error)

    def test_query_with_sources_returns_error_payload_when_search_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.enabled = True
        eng.index = _FakeIndex()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.last_error = ""
        eng.embedding_model = type(
            "_OkEmbedder",
            (),
            {"encode": lambda self, texts, batch_size=32, show_progress_bar=False: np.ones((1, 3), dtype="float32")},
        )()
        eng._abort = False

        result = eng.query_with_sources("hello", k=1)

        self.assertEqual(result["context"], "")
        self.assertEqual(result["count"], 0)
        self.assertIn("faiss boom", result["error"])
        self.assertIn("faiss boom", eng.last_error)

    def test_reader_search_returns_error_payload_when_embedding_fails(self):
        eng = RAGReaderEngine.__new__(RAGReaderEngine)
        eng.loaded = True
        eng.enabled = True
        eng.index = type("_DummyIndex", (), {"search": lambda self, q, k: (np.array([[0.1]], dtype="float32"), np.array([[0]]))})()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.last_error = ""
        eng.fetch_multiplier = 1
        eng.fetch_min = 1
        eng.fetch_cap = 1
        eng.embedding_model = _ExplodingEmbedder()
        eng._ensure_sentence_transformer = lambda: True

        result = eng.search("hello", k=1)

        self.assertEqual(result["context"], "")
        self.assertEqual(result["count"], 0)
        self.assertIn("embedding boom", result["error"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_query_truthfulness.py`

Expected: FAIL because `query()` and `query_with_sources()` do not yet set/return explicit error state for these failures.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lokum_engine/rag/engine.py
import logging

logger = logging.getLogger(__name__)

def query(self, query_text: str, k: int = 3) -> str:
    if not self.enabled or self.index is None:
        return ""
    try:
        self._set_last_error("")
        self._check_abort()
        query_vector = self.embedding_model.encode([query_text])
        query_vector = np.array(query_vector).astype("float32")
        self._check_abort()
        fetch_k = self._fetch_k(int(k))
        distances, indices = self.index.search(query_vector, fetch_k)
        results = []
        for idx in indices[0]:
            if idx == -1 or not self._is_chunk_searchable(int(idx)):
                continue
            results.append(self.documents[int(idx)])
            if len(results) >= int(k):
                break
        return "\n\n---\n\n".join(results) if results else ""
    except Exception as e:
        self._set_last_error(str(e))
        logger.exception("RAG query failed")
        return ""

def query_with_sources(self, query_text: str, k: int = 3) -> Dict[str, Any]:
    empty = {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0, "error": ""}
    if not self.enabled or self.index is None:
        return empty
    try:
        self._set_last_error("")
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
            if idx == -1 or not self._is_chunk_searchable(int(idx)):
                continue
            results.append(self.documents[int(idx)])
            dists.append(float(dist))
            src = {}
            if int(idx) < len(self.chunk_meta):
                meta = self.chunk_meta[int(idx)]
                if isinstance(meta, dict):
                    src = dict(meta)
            sources.append(src)
            if len(results) >= int(k):
                break
        return {
            "context": "\n\n---\n\n".join(results),
            "chunks": results,
            "distances": dists,
            "sources": sources,
            "count": len(results),
            "error": "",
        }
    except Exception as e:
        self._set_last_error(str(e))
        logger.exception("RAG query_with_sources failed")
        empty["error"] = self.last_error
        return empty
```

```python
# src/lokum_engine/rag/reader_engine.py
import logging

logger = logging.getLogger(__name__)

def search(self, query_text: str, k: int = 5) -> Dict[str, Any]:
    empty = {"context": "", "chunks": [], "distances": [], "sources": [], "count": 0, "error": self.last_error}
    if not self.loaded:
        self.load()
    if not self.enabled or self.index is None:
        return empty
    if not self._ensure_sentence_transformer():
        return empty
    try:
        self._set_last_error("")
        query_vector = self.embedding_model.encode([query_text])
        query_vector = np.array(query_vector).astype("float32")
        fetch_k = self._fetch_k(int(k))
        distances, indices = self.index.search(query_vector, int(fetch_k))
        results = []
        dists = []
        sources = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or not self._is_chunk_searchable(int(idx)):
                continue
            results.append(self.documents[int(idx)])
            dists.append(float(dist))
            src = {}
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
            "error": "",
        }
    except Exception as e:
        self._set_last_error(str(e))
        logger.exception("RAG reader search failed")
        empty["error"] = self.last_error
        return empty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_query_truthfulness.py`

Expected: PASS

- [ ] **Step 5: Run nearby existing tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_reader_engine.py tests/test_rag_persistence.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lokum_engine/rag/engine.py src/lokum_engine/rag/reader_engine.py tests/test_rag_query_truthfulness.py
git commit -m "feat: surface rag query failures clearly"
```

## Task 2: RAG Persistence Truthfulness

**Files:**
- Create: `tests/test_rag_persistence_truthfulness.py`
- Modify: `src/lokum_engine/rag/engine.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from unittest.mock import patch

from lokum_engine.rag.engine import RAGEngine


class TestRagPersistenceTruthfulness(unittest.TestCase):
    def test_mark_deleted_returns_false_and_sets_error_when_state_save_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.state = {"version": 1, "files": {}}
        eng.state_path = "/tmp/rag_state.json"
        eng.last_error = ""
        eng._file_id_for = lambda path: "fid"

        def boom(path, obj):
            raise OSError("disk full")

        eng._atomic_write_json = boom

        ok = eng.mark_deleted("/tmp/a.txt", deleted=True)

        self.assertFalse(ok)
        self.assertIn("disk full", eng.last_error)

    def test_save_index_raises_when_meta_write_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.index = object()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.index_path = "/tmp/faiss_index.bin"
        eng.docs_path = "/tmp/docs_metadata.npy"
        eng.meta_path = "/tmp/rag_meta.json"
        eng.state_path = "/tmp/rag_state.json"
        eng.indexed_folder = "/tmp/source"
        eng.state = {"version": 1, "files": {}}

        eng._atomic_write_faiss = lambda path, index_obj: None
        eng._atomic_write_npy = lambda path, arr: None

        def fake_json(path, obj):
            if path.endswith("rag_meta.json"):
                raise OSError("write meta failed")

        eng._atomic_write_json = fake_json

        with self.assertRaises(RuntimeError, match="write meta failed"):
            eng.save_index()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_persistence_truthfulness.py`

Expected: FAIL because current persistence paths still swallow important write failures.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lokum_engine/rag/engine.py
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
    except Exception as e:
        self._set_last_error(str(e))
        logger.exception("Failed to persist deleted state")
        return False
    return True

def save_index(self) -> None:
    if self.index is None:
        return
    try:
        self._atomic_write_faiss(self.index_path, self.index)
        self._atomic_write_npy(self.docs_path, np.array(self.documents, dtype=object))
        if self.chunk_meta:
            self._atomic_write_npy(self.chunks_meta_path, np.array(self.chunk_meta, dtype=object))
        self._atomic_write_json(self.meta_path, {"folder": self.indexed_folder})
        if isinstance(self.state, dict):
            self._atomic_write_json(self.state_path, self.state)
    except Exception as e:
        self._set_last_error(str(e))
        logger.exception("Failed to save RAG index")
        raise RuntimeError(str(e)) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_persistence_truthfulness.py`

Expected: PASS

- [ ] **Step 5: Run nearby existing tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_persistence.py tests/test_rag_quality.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lokum_engine/rag/engine.py tests/test_rag_persistence_truthfulness.py
git commit -m "feat: surface rag persistence failures"
```

## Task 3: Fine-Tune Dataset Preflight

**Files:**
- Create: `tests/test_finetune_validation.py`
- Modify: `src/lokum_engine/finetune/engine.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
import os
import tempfile
import unittest

from lokum_engine.finetune.engine import FinetuneEngine


class TestFinetuneValidation(unittest.TestCase):
    def test_start_training_rejects_missing_train_file(self):
        with tempfile.TemporaryDirectory() as td:
            eng = FinetuneEngine.__new__(FinetuneEngine)
            eng.model_path = "/tmp/model"
            eng.dataset_dir = td
            eng.quality_profile = type(
                "_Prof",
                (),
                {
                    "batch_size": 1,
                    "num_layers": 8,
                    "iters": 10,
                    "grad_checkpoint": True,
                    "val_batches": 1,
                    "steps_per_eval": 10,
                    "max_seq_length": 128,
                    "clear_cache_threshold": 1.0,
                    "presplit_chars_per_token": 4.0,
                },
            )()

            with self.assertRaises(RuntimeError, match="train.jsonl"):
                eng.start_training(dataset_path=td)

    def test_start_training_rejects_row_without_text(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "train.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"bad": "row"}) + "\n")
            with open(os.path.join(td, "valid.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "ok"}) + "\n")

            eng = FinetuneEngine.__new__(FinetuneEngine)
            eng.model_path = "/tmp/model"
            eng.dataset_dir = td
            eng.quality_profile = type(
                "_Prof",
                (),
                {
                    "batch_size": 1,
                    "num_layers": 8,
                    "iters": 10,
                    "grad_checkpoint": True,
                    "val_batches": 1,
                    "steps_per_eval": 10,
                    "max_seq_length": 128,
                    "clear_cache_threshold": 1.0,
                    "presplit_chars_per_token": 4.0,
                },
            )()

            with self.assertRaises(RuntimeError, match="text"):
                eng.start_training(dataset_path=td)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py -k start_training`

Expected: FAIL because `start_training()` does not yet validate dataset structure before subprocess launch.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lokum_engine/finetune/engine.py
def _validate_jsonl_dataset_file(fp: str, *, label: str) -> None:
    if not os.path.isfile(fp):
        raise RuntimeError(f"{label} not found: {fp}")
    rows = 0
    with open(fp, "r", encoding="utf-8") as f:
        for lineno, ln in enumerate(f, start=1):
            s = (ln or "").strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict) or "text" not in obj or not str(obj.get("text") or "").strip():
                raise RuntimeError(f"{label} has invalid row {lineno}: missing text")
            rows += 1
    if rows == 0:
        raise RuntimeError(f"{label} is empty: {fp}")

def _validate_training_dataset_dir(data_dir: str) -> None:
    abs_dir = os.path.abspath(data_dir)
    if not os.path.isdir(abs_dir):
        raise RuntimeError(f"dataset directory not found: {abs_dir}")
    _validate_jsonl_dataset_file(os.path.join(abs_dir, "train.jsonl"), label="train.jsonl")
    _validate_jsonl_dataset_file(os.path.join(abs_dir, "valid.jsonl"), label="valid.jsonl")

def start_training(
    self,
    batch_size: int | None = None,
    num_layers: int | None = None,
    iters: int | None = None,
    dataset_path=None,
    adapter_path=None,
    config_path=None,
    resume_adapter_file: str | None = None,
) -> subprocess.Popen:
    data_dir = dataset_path if dataset_path else self.dataset_dir
    _validate_training_dataset_dir(data_dir)
    prof = getattr(self, "quality_profile", FINETUNE_QUALITY_PROFILES["mid"])
    eff_batch = int(batch_size) if batch_size is not None else int(prof.batch_size)
    eff_layers = int(num_layers) if num_layers is not None else int(prof.num_layers)
    eff_iters = int(iters) if iters is not None else int(prof.iters)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        self.model_path,
        "--train",
        "--data",
        data_dir,
        "--batch-size",
        str(eff_batch),
        "--num-layers",
        str(eff_layers),
        "--iters",
        str(eff_iters),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py -k start_training`

Expected: PASS

- [ ] **Step 5: Run nearby existing tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_quality.py tests/test_finetune_presplit_chatml.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lokum_engine/finetune/engine.py tests/test_finetune_validation.py
git commit -m "feat: validate finetune datasets before launch"
```

## Task 4: Strict Malformed JSONL Handling

**Files:**
- Modify: `tests/test_finetune_validation.py`
- Modify: `src/lokum_engine/finetune/engine.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
import os
import tempfile
import unittest

from lokum_engine.finetune.engine import _presplit_jsonl_file


class TestFinetuneValidation(unittest.TestCase):
    def test_presplit_jsonl_file_strict_mode_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "train.jsonl")
            with open(fp, "w", encoding="utf-8") as f:
                f.write('{"text":"ok"}\n')
                f.write('not-json\n')

            with self.assertRaises(RuntimeError, match="Malformed JSONL"):
                _presplit_jsonl_file(fp, max_seq_length=128, batch_size=1, strict=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py -k strict_mode`

Expected: FAIL because `_presplit_jsonl_file()` currently coerces malformed lines into plain text.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lokum_engine/finetune/engine.py
def _presplit_jsonl_file(fp: str, max_seq_length: int, batch_size: int, strict: bool = False) -> int:
    if not fp or not os.path.isfile(fp):
        return 0
    changed = 0
    out_lines: list[str] = []
    with open(fp, "r", encoding="utf-8") as f:
        for lineno, ln in enumerate(f.read().splitlines(), start=1):
            s = (ln or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Malformed JSONL at {fp}:{lineno}") from e
                obj = {"text": s}
            if not isinstance(obj, dict) or "text" not in obj:
                out_lines.append(json.dumps(obj, ensure_ascii=False))
                continue
            text = str(obj.get("text") or "")
            pieces = _presplit_text(text, max_seq_length=max_seq_length, batch_size=batch_size)
            if len(pieces) > 1:
                changed += 1
            for p in pieces:
                obj2 = dict(obj)
                obj2["text"] = p
                out_lines.append(json.dumps(obj2, ensure_ascii=False))
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in out_lines:
            if ln.strip():
                f.write(ln + "\n")
    os.replace(tmp, fp)
    return changed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py -k strict_mode`

Expected: PASS

- [ ] **Step 5: Run nearby existing tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py tests/test_finetune_presplit_chatml.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lokum_engine/finetune/engine.py tests/test_finetune_validation.py
git commit -m "feat: add strict finetune jsonl validation"
```

## Task 5: Validation Preparation Truthfulness

**Files:**
- Modify: `tests/test_finetune_validation.py`
- Modify: `src/lokum_engine/finetune/engine.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import tempfile
import unittest
from unittest.mock import patch

from lokum_engine.finetune.engine import FinetuneEngine


class TestFinetuneValidation(unittest.TestCase):
    def test_start_validation_surfaces_presplit_failure(self):
        with tempfile.TemporaryDirectory() as td:
            valid_fp = os.path.join(td, "valid.jsonl")
            with open(valid_fp, "w", encoding="utf-8") as f:
                f.write('{"text":"ok"}\n')

            eng = FinetuneEngine.__new__(FinetuneEngine)
            eng.model_path = "/tmp/model"
            eng.dataset_dir = td
            eng.quality_profile = type("_Prof", (), {"max_seq_length": 128, "clear_cache_threshold": 1.0})()

            with patch("lokum_engine.finetune.engine._presplit_jsonl_file", side_effect=RuntimeError("presplit boom")):
                with self.assertRaises(RuntimeError, match="presplit boom"):
                    eng.start_validation(dataset_path=td, adapter_path="/tmp/adapter")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py -k start_validation`

Expected: FAIL because `start_validation()` currently swallows presplit exceptions.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lokum_engine/finetune/engine.py
def start_validation(self, dataset_path: str, adapter_path: str, config_path: str | None = None) -> subprocess.Popen:
    data_dir = dataset_path if dataset_path else self.dataset_dir
    valid_fp = os.path.join(os.path.abspath(data_dir), "valid.jsonl")
    if not os.path.isfile(valid_fp):
        raise RuntimeError("valid.jsonl not found in dataset directory.")
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.abspath(self.dataset_dir or "lora_data")
    eval_dir = os.path.abspath(os.path.join(base, "validate_only", f"run_{ts}"))
    os.makedirs(eval_dir, exist_ok=True)
    shutil.copyfile(valid_fp, os.path.join(eval_dir, "test.jsonl"))
    if os.environ.get("LOKUMAI_FT_PRESPLIT", "1") != "0":
        max_seq = int(os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip() or "512")
        try:
            _presplit_jsonl_file(os.path.join(eval_dir, "test.jsonl"), max_seq, 1, strict=True)
        except Exception as e:
            raise RuntimeError(str(e)) from e
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", self.model_path, "--data", eval_dir, "--test"]
    if adapter_path:
        cmd += ["--adapter-path", str(adapter_path)]
    if config_path:
        cmd += ["--config", str(config_path)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py -k start_validation`

Expected: PASS

- [ ] **Step 5: Run nearby existing tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_finetune_validation.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lokum_engine/finetune/engine.py tests/test_finetune_validation.py
git commit -m "feat: surface finetune validation prep failures"
```

## Task 6: Logging Cleanup And Full Verification

**Files:**
- Modify: `src/lokum_engine/rag/engine.py`
- Modify: `src/lokum_engine/rag/reader_engine.py`
- Modify: `src/lokum_engine/finetune/engine.py`
- Test: `tests/test_rag_query_truthfulness.py`
- Test: `tests/test_rag_persistence_truthfulness.py`
- Test: `tests/test_finetune_validation.py`

- [ ] **Step 1: Write the failing tests**

```python
import logging
import unittest

from lokum_engine.rag.engine import logger as rag_logger


class TestLoggingUsage(unittest.TestCase):
    def test_logger_exists_for_rag_engine(self):
        self.assertIsInstance(rag_logger, logging.Logger)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_query_truthfulness.py -k logger_exists`

Expected: FAIL until module-level loggers exist consistently.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lokum_engine/rag/engine.py
import logging
logger = logging.getLogger(__name__)

# src/lokum_engine/rag/reader_engine.py
import logging
logger = logging.getLogger(__name__)

# src/lokum_engine/finetune/engine.py
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run targeted tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_rag_query_truthfulness.py tests/test_rag_persistence_truthfulness.py tests/test_finetune_validation.py`

Expected: PASS

- [ ] **Step 5: Run full test suite and diagnostics**

Run: `PYTHONPATH=src python3 -m pytest -q`

Expected: PASS

Then verify diagnostics for:

- `src/lokum_engine/rag/engine.py`
- `src/lokum_engine/rag/reader_engine.py`
- `src/lokum_engine/finetune/engine.py`

Expected: no new diagnostics

- [ ] **Step 6: Commit**

```bash
git add src/lokum_engine/rag/engine.py src/lokum_engine/rag/reader_engine.py src/lokum_engine/finetune/engine.py tests/test_rag_query_truthfulness.py tests/test_rag_persistence_truthfulness.py tests/test_finetune_validation.py
git commit -m "feat: complete truthful reliability milestone"
```

## Spec Coverage Check

- RAG query truthfulness -> Task 1
- RAG ingest and persistence truthfulness -> Task 2
- fine-tune dataset preflight validation -> Task 3
- strict malformed JSONL handling -> Task 4
- validation preparation truthfulness -> Task 5
- logging cleanup + regression verification -> Task 6

No spec gaps remain for this milestone.
