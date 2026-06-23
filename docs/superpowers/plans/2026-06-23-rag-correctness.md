# Milestone 2: RAG Correctness Hardening

## Goal
Ensure RAG retrieval is correct and predictable across file changes, model changes, and different ingestion paths.

## Motivation
Currently, `ingest_folder()` doesn't clean up chunks for files that have been deleted from disk. Furthermore, changing the embedding model doesn't invalidate old vectors, leading to silent retrieval failures. Finally, `ingest_folder` and direct file ingestion disagree on supported extensions.

## Tasks

### Task 1: Reconcile deleted files
- **Target**: `src/lokum_engine/rag/engine.py` (`_ingest_paths` or `ingest_folder`)
- **Action**: When scanning a folder, compare the found files against the files already tracked in `rag_state.json`. If a tracked file is no longer on disk, mark it as deleted (using the existing `mark_deleted` logic).
- **Test**: Add a regression test where a file is ingested, deleted from disk, folder re-ingested, and the file is confirmed deleted in state.

### Task 2: Store compatibility validation
- **Target**: `src/lokum_engine/rag/engine.py` & `reader_engine.py`
- **Action**: Persist the active embedding model identifier (e.g., name or dimension) in `store_meta` during `save_index()`. On load (`_load_state` or `__init__`), verify the current model matches the stored model. If mismatched, raise an explicit error (or force a re-index if appropriate, but fail-fast is safer).
- **Test**: Add a test that initializes an engine with model A, saves, and asserts initializing with model B against the same store fails cleanly.

### Task 3: Unify extension allowlist
- **Target**: `src/lokum_engine/rag/engine.py`
- **Action**: Consolidate the supported extensions used by `ingest_folder()` to match exactly what `process_file()` supports. Define this list once.
- **Test**: Add a test that verifies `ingest_folder()` picks up all supported file types.

### Task 4: Full verification
- **Action**: Run the complete suite to ensure correctness patches don't break existing reliability or retrieval tests.
