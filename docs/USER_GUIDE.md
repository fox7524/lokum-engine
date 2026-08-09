# Lokum Engine User Guide

This guide provides technical details on the architecture, components, and configuration of Lokum Engine.

---

## 1. Quality Profiles

Lokum Engine uses predefined configuration profiles (`Base`, `Mid`, `Fab`) to standardize hyper-parameters across different hardware constraints.

- **`Base`:** Prioritizes execution speed and memory efficiency. Optimized for entry-level Apple Silicon (e.g., M1/M2 with 8GB-16GB RAM). Uses lower chunk sizes and smaller fine-tuning batch sizes.
- **`Mid`:** The default profile. Balances feature usage with memory constraints. Suitable for standard development environments.
- **`Fab`:** Unlocks maximum context windows, aggressive fetch multipliers, and higher MLX layers. Requires higher memory capacity (e.g., M-Series Max/Ultra chips).

---

## 2. RAG Engine Capabilities

The RAG Engine provides a pipeline for document extraction, semantic chunking, indexing, and retrieval.

### Initialization & Ingestion

```python
from lokum_engine import RAGEngineMid

rag = RAGEngineMid(storage_dir="./index")
rag.ingest_folder("/path/to/docs", recursive=True)
```
The ingestion process maintains a local state file. Subsequent calls to `ingest_folder` will only process new or modified files.

### Semantic Chunking
Instead of standard character-count splitting, the engine uses the NLTK library to tokenize documents at sentence boundaries. This ensures that context chunks contain complete semantic units before reaching the specified size limit.

### Hybrid Search (FAISS + BM25)
The engine maintains both a dense vector index (FAISS) and a sparse index (BM25). During a query:
1. The dense index returns Top-K results based on semantic similarity.
2. The sparse index returns Top-K results based on exact keyword matching.
3. The results are merged using Reciprocal Rank Fusion (RRF) to normalize the scoring.

### HyDE (Hypothetical Document Embeddings)
HyDE is supported by passing a completion function to the query. The engine will generate a hypothetical response using the provided LLM function and use it to execute the retrieval.

```python
def llm_completion(prompt: str) -> str:
    # Implementation for LLM inference
    return "..."

results = rag.query(
    search_text="System configuration steps", 
    k=5, 
    hyde_completion_fn=llm_completion
)
```

---

## 3. Fine-Tuning Engine Capabilities

The Fine-Tuning Engine acts as an abstraction layer over `mlx-lm`, focusing on dataset integrity and memory management.

### ChatML-Aware Presplitting
To mitigate Out-Of-Memory (OOM) errors caused by large token sequences, the engine pre-splits the dataset. The `_presplit_text` function parses ChatML markers (`<|im_start|>` and `<|im_end|>`) and splits large contexts at paragraph boundaries to ensure instruction tags remain intact.

### Data Curation
The `curation.py` module provides functions to sanitize datasets before training:

```python
from lokum_engine.finetune.curation import deduplicate_dataset, auto_score_dataset

# Deduplicate dataset using MinHash (Jaccard similarity)
deduplicate_dataset(
    input_path="raw_data.jsonl", 
    output_path="clean_data.jsonl", 
    similarity_threshold=0.85
)

# Filter dataset using an LLM-as-a-judge scoring function
auto_score_dataset(
    input_path="clean_data.jsonl",
    output_path="filtered_data.jsonl",
    scoring_fn=scoring_function,
    min_score=7.0
)
```

### Preference Datasets (DPO / ORPO)
The engine provides formatting utilities to convert prompt/chosen/rejected triplets into the JSONL schema required for MLX Direct Preference Optimization.

```python
from lokum_engine import FinetuneEngineMid

ft = FinetuneEngineMid(model_path="mlx-community/Llama-3-8B-Instruct-4bit")

preference_data = [
    {
        "prompt": "User query",
        "chosen": "Correct response",
        "rejected": "Incorrect response"
    }
]

train_path, valid_path = ft.prepare_preference_dataset(preference_data)
```

### MLX Optimizations
The training process automatically applies the following configurations:
- **Gradient Checkpointing:** Enabled by default in `Mid` and `Fab` profiles to reduce memory footprint.
- **Dynamic Batching:** Scales batch sizes relative to the `max_seq_length`.
- **Log Suppression:** Modifies `MTL_LOG_LEVEL` environment variables to reduce standard output noise on macOS.

---

## 4. Environment Variables

Engine parameters can be overridden using environment variables.

**RAG Variables:**
- `LOKUMAI_RAG_CHUNK_SIZE` (int)
- `LOKUMAI_RAG_OVERLAP` (int)
- `LOKUMAI_RAG_FETCH_MULTIPLIER` (int)
- `LOKUMAI_RAG_USE_HYBRID` (1 or 0)

**Fine-Tuning Variables:**
- `LOKUMAI_FT_QUALITY` (base, mid, fab)
- `LOKUMAI_FT_MAX_SEQ_LENGTH` (int)
- `LOKUMAI_FT_GRAD_CHECKPOINT` (1 or 0)
- `LOKUMAI_FT_BATCH_SIZE` (int)
- `LOKUMAI_FT_PRESPLIT_CHARS_PER_TOKEN` (float)
