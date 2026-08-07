# 📖 Lokum Engine: Comprehensive User Guide

Welcome to the ultimate deep-dive into the **Lokum Engine** (v1.0.0). This guide is designed to take you from a beginner to an absolute master of enterprise-grade Retrieval-Augmented Generation (RAG) and MLX LoRA Fine-Tuning.

---

## 📑 Table of Contents
1. [The Philosophy of Quality Profiles](#1-the-philosophy-of-quality-profiles)
2. [RAG Engine Deep Dive](#2-rag-engine-deep-dive)
    - [Initialization & Ingestion](#initialization--ingestion)
    - [Semantic Chunking with NLTK](#semantic-chunking-with-nltk)
    - [Hybrid Search (FAISS + BM25) & RRF](#hybrid-search-faiss--bm25--rrf)
    - [HyDE (Hypothetical Document Embeddings)](#hyde-hypothetical-document-embeddings)
3. [Fine-Tuning Engine Deep Dive](#3-fine-tuning-engine-deep-dive)
    - [ChatML-Safe Presplitting](#chatml-safe-presplitting)
    - [Auto Data Curation (Deduplication & LLM-as-a-judge)](#auto-data-curation)
    - [DPO / ORPO Preference Datasets](#dpo--orpo-preference-datasets)
    - [Extreme MLX Speed Optimizations](#extreme-mlx-speed-optimizations)
4. [Environment Variables Reference](#4-environment-variables-reference)

---

## 1. The Philosophy of Quality Profiles

Lokum Engine introduces a unique concept called **Quality Profiles** (`Base`, `Mid`, `Fab`). Instead of forcing developers to manually configure 20 different hyper-parameters (like chunk sizes, overlap, faiss metric types, MLX layers, etc.), you simply pick a profile based on your hardware and target quality.

- **`FinetuneEngineBase` / `RAGEngineBase`:** Prioritizes speed and memory efficiency. Great for M1/M2 Airs with 8GB-16GB RAM.
- **`FinetuneEngineMid` / `RAGEngineMid`:** The sweet spot. Balances enterprise features with reasonable execution times.
- **`FinetuneEngineFab` / `RAGEngineFab`:** The "Fabulous" profile. Unlocks the largest context windows, most aggressive RAG fetch multipliers, highest MLX layers, and maximum accuracy. Perfect for M-Series Max/Ultra chips (32GB+ RAM) or production cloud deployments.

---

## 2. RAG Engine Deep Dive

The Lokum RAG Engine is not just a wrapper around FAISS. It is a complete pipeline that handles text extraction (PDF, DOCX, Markdown, Code, ZIM), chunking, vectorization, and retrieval.

### Initialization & Ingestion

```python
from lokum_engine import RAGEngineFab

rag = RAGEngineFab(storage_dir="./my_enterprise_index")

# Ingest an entire directory recursively
rag.ingest_folder("/path/to/docs", recursive=True)

# Lokum Engine automatically handles state. 
# If you run `ingest_folder` again, it only processes new or modified files!
```

### Semantic Chunking with NLTK
Traditional RAG engines split text by character count (e.g., every 500 characters). This often slices sentences in half, destroying the semantic meaning. 
Lokum Engine uses **NLTK (Natural Language Toolkit)** under the hood. It tokenizes the document by actual sentence boundaries, grouping sentences together until they reach the target chunk size.

### Hybrid Search (FAISS + BM25) & RRF
Relying solely on dense vector embeddings (Cosine Similarity) is dangerous. It's great for conceptual questions but terrible at exact keyword matching (e.g., searching for a specific product ID like "LKM-992").

Lokum Engine natively builds a **Sparse Index (BM25)** alongside the **Dense Index (FAISS)**.
When you call `rag.query()`, the engine:
1. Fetches the Top-K results using semantic FAISS vectors.
2. Fetches the Top-K results using exact-match BM25.
3. Merges them using **Reciprocal Rank Fusion (RRF)**, mathematically calculating the optimal rank for each chunk.

*This is enabled by default in the `Fab` profile. You don't have to write a single extra line of code!*

### HyDE (Hypothetical Document Embeddings)
HyDE is an advanced RAG technique. Instead of searching the database using the user's short question, it uses an LLM to generate a "fake" (hypothetical) answer, and then searches the database using that generated answer. This drastically improves recall.

```python
# To use HyDE, you just need to pass an LLM completion function to the query
def my_llm_completion(prompt: str) -> str:
    # Call OpenAI, Anthropic, or a local MLX model here
    return "..."

results = rag.query(
    search_text="How do I configure the firewall?", 
    k=5, 
    hyde_completion_fn=my_llm_completion
)
```

---

## 3. Fine-Tuning Engine Deep Dive

Lokum Engine wraps the incredible `mlx-lm` library, adding enterprise-grade safety nets, data curation pipelines, and extreme optimizations for Apple Silicon.

### ChatML-Safe Presplitting
The #1 cause of bad fine-tuning runs is Out-Of-Memory (OOM) errors caused by massive text samples. 
To prevent this, Lokum Engine pre-splits your dataset. **However**, unlike naive splitters, our `_presplit_text` logic is **ChatML-aware**. 

If it detects `<|im_start|>` and `<|im_end|>` tags, it will *never* slice a string in the middle of a tag. It carefully dissects the conversation at paragraph boundaries, ensuring the model never learns broken prompt templates.

### Auto Data Curation
Garbage in, garbage out. Lokum Engine v1.0.0 ships with a brand new `curation.py` module.

```python
from lokum_engine.finetune.curation import deduplicate_dataset, auto_score_dataset

# 1. Deduplicate your dataset using MinHash (Jaccard similarity)
# Removes slightly reworded or duplicate rows that cause overfitting
deduplicate_dataset(
    input_path="raw_data.jsonl", 
    output_path="clean_data.jsonl", 
    similarity_threshold=0.85
)

# 2. Score your dataset (LLM-as-a-judge)
# Removes low-quality or nonsensical rows before training
auto_score_dataset(
    input_path="clean_data.jsonl",
    output_path="premium_data.jsonl",
    scoring_fn=my_llm_scoring_function, # Should return 1-10
    min_score=7.0
)
```

### DPO / ORPO Preference Datasets
Standard fine-tuning (SFT) just teaches the model to talk like the dataset. **Direct Preference Optimization (DPO)** teaches the model *what not to say*. 

Lokum Engine provides a native builder for preference datasets:

```python
from lokum_engine import FinetuneEngineFab

ft = FinetuneEngineFab(model_path="mlx-community/Llama-3-8B-Instruct-4bit")

preference_data = [
    {
        "prompt": "Write a python script to delete all files.",
        "chosen": "I cannot help with destructive actions.",
        "rejected": "import os; os.system('rm -rf /')"
    }
]

# Automatically formats to the exact JSONL schema required for MLX DPO
train_path, valid_path = ft.prepare_preference_dataset(preference_data)
```

### Extreme MLX Speed Optimizations
Under the hood, `start_training()` passes a highly optimized set of flags to the MLX compiler.
- **`--grad-checkpoint`**: Enabled by default in `Mid` and `Fab` profiles, saving massive amounts of RAM at a slight compute cost.
- **Metal Log Suppression**: Automatically silences the noisy `MTL_LOG_LEVEL` and IOSurface warnings on macOS, keeping your terminal clean.
- **Dynamic Batching**: Automatically scales batch sizes based on the `max_seq_length` and your chosen profile.

---

## 4. Environment Variables Reference

Power users can override *any* Quality Profile setting using environment variables. 

**RAG Variables:**
- `LOKUMAI_RAG_CHUNK_SIZE` (int)
- `LOKUMAI_RAG_OVERLAP` (int)
- `LOKUMAI_RAG_FETCH_MULTIPLIER` (int)
- `LOKUMAI_RAG_USE_HYBRID` (1 or 0)

**Fine-Tuning Variables:**
- `LOKUMAI_FT_QUALITY` (base, mid, fab)
- `LOKUMAI_FT_MAX_SEQ_LENGTH` (int, e.g., 2048)
- `LOKUMAI_FT_GRAD_CHECKPOINT` (1 or 0)
- `LOKUMAI_FT_BATCH_SIZE` (int)
- `LOKUMAI_FT_PRESPLIT_CHARS_PER_TOKEN` (float, default 4.0)

---
*Built with ❤️ by developers who got tired of fighting infrastructure and just wanted to train great AI models.*
