# Lokum Engine

[![PyPI Version](https://img.shields.io/pypi/v/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Lokum Engine is a Python library that provides integrated pipelines for Retrieval-Augmented Generation (RAG) and MLX-based LLM fine-tuning. It is designed to simplify local AI development on Apple Silicon by handling data processing, vector indexing, and MLX memory optimizations natively.

## Features

### RAG Engine
* **Hybrid Search:** Combines dense retrieval (FAISS Cosine Similarity) with sparse retrieval (BM25) using Reciprocal Rank Fusion (RRF).
* **Semantic Chunking:** Uses NLTK for sentence-boundary aware text splitting instead of fixed-character limits.
* **HyDE Support:** Allows integration with external LLM functions to perform Hypothetical Document Embeddings for expanded query recall.
* **Multi-format Ingestion:** Supports automated text extraction from PDF, DOCX, Markdown, Code files, and ZIM archives.

### Fine-Tuning Engine
* **Apple Silicon Optimized:** Built on top of `mlx-lm` with configurable profiles for gradient checkpointing, batch sizing, and memory management.
* **ChatML-Safe Presplitting:** Pre-processes long training samples to prevent Out-Of-Memory (OOM) errors without breaking `<|im_start|>` and `<|im_end|>` boundaries.
* **Preference Optimization:** Includes dataset formatting utilities for Direct Preference Optimization (DPO) and ORPO workflows.
* **Data Curation:** Provides MinHash-based deduplication and LLM-as-a-judge scoring functions to clean training datasets.

## Installation

Install via pip:

```bash
pip install lokum-engine
```

## Usage

### RAG Pipeline

```python
from lokum_engine import RAGEngineMid

# Initialize the engine with a default storage directory
rag = RAGEngineMid(storage_dir="./index_storage")

# Ingest documents from a directory
rag.ingest_folder("/path/to/documents", recursive=True)

# Query the hybrid index
results = rag.query("How to configure the network?", k=5)
print(results)
```

### Fine-Tuning Pipeline

```python
from lokum_engine import FinetuneEngineMid

# Initialize the fine-tuning engine
ft = FinetuneEngineMid(model_path="mlx-community/Llama-3-8B-Instruct-4bit")

# Pre-split the dataset to avoid OOM issues during training
ft.presplit_dataset(
    dataset_path="train.jsonl", 
    max_seq_length=2048, 
    batch_size=4
)

# Start the MLX training process
process = ft.start_training(
    dataset_path="train.jsonl",
    batch_size=4,
    num_layers=16,
    iters=1000
)
```

## Documentation

For detailed information on configuration profiles, environment variables, and advanced usage, please refer to the [User Guide](docs/USER_GUIDE.md).

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). You are free to use, share, and adapt the code for non-commercial purposes, provided you give appropriate credit. See the [LICENSE](LICENSE) file for full details.
