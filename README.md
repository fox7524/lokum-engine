# Lokum Engine

[![PyPI Version](https://img.shields.io/pypi/v/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Lokum Engine is a Python library I built for RAG and MLX-based LLM fine-tuning. I made it to simplify local AI development on Apple Silicon so you don't have to deal with data processing, vector indexing, or MLX memory optimizations manually.

## What's inside?

### RAG Engine
* **Hybrid Search:** Mixes FAISS (dense) and BM25 (sparse) using RRF.
* **Semantic Chunking:** Uses NLTK to split text by sentences, not just blind character counts.
* **HyDE Support:** Bring your own LLM function to generate hypothetical docs for better recall.
* **Format Support:** Reads PDF, DOCX, Markdown, Code, and ZIM files directly.

### Fine-Tuning Engine
* **Apple Silicon Native:** Runs on `mlx-lm`. Handles gradient checkpointing, batch sizing, and memory management out of the box.
* **ChatML-Safe Presplitting:** Pre-splits long samples so you don't get OOM errors, without messing up `<|im_start|>` and `<|im_end|>` tags.
* **DPO/ORPO:** Utilities to format datasets for preference optimization.
* **Curation:** MinHash deduplication and LLM-as-a-judge scoring to clean up your data.

## Installation

Install via pip:

```bash
pip install lokum-engine
```

## Quick Start

### RAG Pipeline

```python
from lokum_engine import RAGEngineMid

# Initialize the engine
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

# Pre-split so it doesn't crash on long sequences
ft.presplit_dataset(
    dataset_path="train.jsonl", 
    max_seq_length=2048, 
    batch_size=4
)

# Start MLX training
process = ft.start_training(
    dataset_path="train.jsonl",
    batch_size=4,
    num_layers=16,
    iters=1000
)
```

## Docs

Check out the [User Guide](docs/USER_GUIDE.md) for configuration profiles, environment variables, and advanced usage.

## License

CC BY-NC 4.0. You can use it for non-commercial stuff as long as you give credit. Check [LICENSE](LICENSE) for details.
