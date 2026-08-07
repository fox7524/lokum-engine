# Lokum Engine Upgrade Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the `lokum-engine` RAG and Finetune engines using `Lokum-f` as the definitive base, then elevate them with industry-leading features (Hybrid Search, Semantic Chunking, Auto Curation, Extreme MLX Speed, and DPO/ORPO support) to make them top-tier Python libraries.

**Architecture:** 
1. **Base Porting**: Ensure `lokum-engine` precisely matches `Lokum-f`'s logic (e.g., `faiss.IndexFlatIP` with L2 normalization, `compact_index()`, advanced ZIM extraction, robust ChatML pre-splitting).
2. **RAG Innovations**: Introduce BM25 (Hybrid Search) with Reciprocal Rank Fusion (RRF), Semantic Chunking, and HyDE (Hypothetical Document Embeddings).
3. **Finetune Innovations**: Add data curation (deduplication & quality scoring), advanced MLX optimization flags, and preference dataset (DPO/ORPO) builders.

**Tech Stack:** Python, FAISS, Sentence-Transformers, MLX, BM25 (rank_bm25), NLTK (for semantic chunking).

---

### Task 1: Establish the Lokum-f Base (RAG)

**Files:**
- Modify: `src/lokum_engine/rag/engine.py`

- [ ] **Step 1: Port `Lokum-f` Vector Similarity Logic**
Update `_ingest_paths` and `query_with_sources` to use `faiss.normalize_L2` and `IndexFlatIP` (Cosine Similarity) instead of `IndexFlatL2` (Euclidean).

- [ ] **Step 2: Port `compact_index` Method**
Copy the `compact_index` method directly from `Lokum-f/core/rag_engine.py` into `src/lokum_engine/rag/engine.py` to allow garbage collection of superseded chunks. Integrate it at the end of `_ingest_paths`.

- [ ] **Step 3: Port ZIM Extraction and Robustness Updates**
Ensure `extract_from_zim` uses the exact robust iterator logic from `Lokum-f` to handle `libzim` and `pyzim` edge cases.

### Task 2: Implement RAG Industry Standards (Magnificent Features)

**Files:**
- Modify: `src/lokum_engine/rag/engine.py`
- Modify: `pyproject.toml` (Add dependencies: `rank_bm25`, `nltk`)

- [ ] **Step 1: Add Semantic Chunking**
Enhance `chunk_text` to optionally use `nltk.tokenize.sent_tokenize` (or regex-based sentence boundaries) so chunks break cleanly on sentences rather than arbitrary character cuts.

- [ ] **Step 2: Add Hybrid Search (BM25 + RRF)**
Integrate `rank_bm25.BM25Okapi`. During `ingest_documents`, build a BM25 index alongside FAISS. During `query_with_sources`, retrieve top-K from both FAISS and BM25, then fuse them using Reciprocal Rank Fusion (RRF).

- [ ] **Step 3: Add HyDE Support (Query Expansion)**
Add a configuration flag to optionally expand user queries using a provided LLM completion function before vector retrieval, dramatically increasing recall for complex questions.

### Task 3: Establish the Lokum-f Base (Finetune)

**Files:**
- Modify: `src/lokum_engine/finetune/engine.py`

- [ ] **Step 1: Sync Presplit Logic**
Ensure `_presplit_text` and `_presplit_jsonl_file` exactly match the ChatML-aware chunking rules from `Lokum-f`.

- [ ] **Step 2: Sync MLX Subprocess Arguments**
Ensure `start_training` sets all environment variables (`MTL_LOG_LEVEL`, `MLX_LOG_LEVEL`, `MTL_DEBUG_LAYER`) and passes arguments exactly as `Lokum-f` does to avoid Metal spam and maximize stability.

### Task 4: Implement Finetune Industry Standards (Magnificent Features)

**Files:**
- Create: `src/lokum_engine/finetune/curation.py`
- Modify: `src/lokum_engine/finetune/engine.py`

- [ ] **Step 1: Auto Data Curation**
Implement `curation.py` with utilities to deduplicate JSONL datasets based on MinHash/Jaccard similarity, and a pipeline to score datasets (LLM-as-a-judge stub) to remove low-quality rows before training.

- [ ] **Step 2: DPO / ORPO Support**
Add a `prepare_preference_dataset` method in `FinetuneEngine` that formats data for Direct Preference Optimization (requires `prompt`, `chosen`, `rejected` keys) so researchers can perform alignment, not just SFT.

- [ ] **Step 3: Extreme MLX Speed Optimizations**
Expose Unsloth-inspired MLX arguments in `start_training` (e.g., dynamic batching flags, advanced gradient checkpointing tuning, and metal memory limits via environment variables) to push Apple Silicon to the absolute limit.

### Task 5: Verification & Tests

**Files:**
- Modify: `tests/test_rag_quality.py`
- Modify: `tests/test_finetune_quality.py`

- [ ] **Step 1: RAG Tests**
Add unit tests verifying `IndexFlatIP` is used, BM25 fusion returns expected combined ranks, and `compact_index` properly shrinks the FAISS index.

- [ ] **Step 2: Finetune Tests**
Add unit tests for `prepare_preference_dataset` and the `curation` deduplication logic.

---
**Verification Steps:**
- Run `pytest tests/` to ensure all existing and new tests pass.
- Execute a sample script utilizing both the RAG Engine (with BM25 enabled) and Finetune Engine (generating a DPO dataset) to verify end-to-end functionality.