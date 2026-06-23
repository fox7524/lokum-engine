# Phase 2: Advanced Retrieval & Quality
## Sub-project A: Re-ranking Pipeline

### Goal
Implement a Cross-Encoder re-ranking pipeline to dramatically improve retrieval precision. FAISS (dense retrieval) will fetch a larger candidate pool (e.g., top 20), and the Cross-Encoder will score and sort them to return the absolute best top-k (e.g., top 5).

### Tasks

#### Task 1: Add Reranker to Quality Profiles
- **Target**: `src/lokum_engine/rag/profiles.py` (or where profiles are defined)
- **Action**: Add `rerank_model_name` and `rerank_multiplier` (how many extra chunks to fetch from FAISS) to `RAGQualityProfile`.
- **Profiles**:
  - `Base`: `rerank_model_name = None` (Fastest, no reranking)
  - `Mid`: `rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"` (Lightweight)
  - `Fab`: `rerank_model_name = "BAAI/bge-reranker-base"` (Enterprise SOTA)

#### Task 2: Load Reranker in Engines
- **Target**: `src/lokum_engine/rag/engine.py` and `src/lokum_engine/rag/reader_engine.py`
- **Action**: In `__init__`, if `rerank_model_name` is present, initialize `sentence_transformers.CrossEncoder`. Respect the device (CPU/MPS).

#### Task 3: Implement Reranking Logic in Search
- **Target**: `src/lokum_engine/rag/reader_engine.py` (`search` method) and `engine.py` (`query` methods)
- **Action**: 
  - If reranker is active: fetch `top_n = k * rerank_multiplier` from FAISS.
  - Construct pairs: `[(query, chunk_text) for chunk in candidates]`.
  - Score pairs using `cross_encoder.predict()`.
  - Sort candidates by score descending and return the top `k`.
- **Test**: Add a test that proves reranking changes the order of results compared to raw FAISS.

#### Task 4: Full Verification
- **Action**: Run the complete test suite to ensure no regressions and that reranking works seamlessly.