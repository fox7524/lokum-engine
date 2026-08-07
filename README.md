<div align="center">

# 🌌 Lokum Engine 🌟

**The Undisputed King of RAG & LLM Fine-Tuning**

[![PyPI Version](https://img.shields.io/pypi/v/lokum-engine.svg?style=for-the-badge&color=blue)](https://pypi.org/project/lokum-engine/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lokum-engine.svg?style=for-the-badge)](https://pypi.org/project/lokum-engine/)
[![License](https://img.shields.io/pypi/l/lokum-engine.svg?style=for-the-badge&color=green)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Enterprise_Ready-purple?style=for-the-badge)](#)
[![Downloads](https://static.pepy.tech/badge/lokum-engine)](https://pepy.tech/project/lokum-engine)

*From local experimentation to Fortune 500 production in 3 lines of code.*

[Quickstart](#quickstart) • [Features](#why-lokum-engine) • [User Guide](docs/USER_GUIDE.md) • [Roadmap](docs/ROADMAP.md)

</div>

---

## ⚡ Why Lokum Engine?

Lokum Engine is the developer-first building block for **Retrieval-Augmented Generation (RAG)** and **State-of-the-Art LLM Fine-Tuning**. We abstracted away the infrastructure headaches, OOM crashes, and broken data pipelines so you can focus on building intelligent agents.

With the **v1.0.0** release, Lokum Engine sets the industry standard. It is specifically designed to push Apple Silicon to its absolute limits while providing features that usually require a sprawling microservice architecture.

### 🚀 Groundbreaking Features
- **Hybrid Search (FAISS + BM25):** Combines dense semantic vector search (Cosine Similarity) with sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF) for unparalleled accuracy.
- **HyDE (Hypothetical Document Embeddings):** Expands user queries intelligently before retrieval, dramatically increasing recall for complex questions.
- **Semantic Chunking:** Slices text at logical sentence boundaries instead of arbitrary character counts. Context is never awkwardly cut in half.
- **DPO / ORPO Support:** Move beyond standard fine-tuning (SFT) and align models with human preferences natively using our automated preference dataset builders.
- **Auto Data Curation:** Built-in tools for MinHash deduplication and LLM-as-a-judge dataset scoring to ensure only high-quality data reaches your model.
- **Extreme MLX Speed Optimizations:** Dynamic batching, advanced gradient checkpointing, and environment variable tuning to squeeze every drop of performance out of Apple Silicon.
- **ChatML-Safe Presplitting:** Guarantee your fine-tuning data never splits across critical instruction boundaries.

---

## 📦 Install

```bash
pip install lokum-engine
```
*(Note: Lokum Engine intentionally includes heavy, production-grade dependencies like FAISS, sentence-transformers, PyMuPDF, rank_bm25, nltk, and MLX out of the box).*

---

## 🧠 Quickstart: RAG (Retrieval-Augmented Generation)

Turn any folder of documents into a highly accurate semantic search engine instantly.

```python
from lokum_engine import RAGEngineFab

# Initialize with the 'Fab' profile for maximum enterprise-grade retrieval quality
rag = RAGEngineFab()  

# Recursively ingest PDFs, Markdown, Code, and text files
# Now with Semantic Chunking and BM25 Hybrid Indexing!
rag.ingest_folder("/path/to/your/enterprise/docs", recursive=True)

# Query with semantic understanding (Hybrid Search + RRF applied automatically)
context = rag.query("How do we scale our distributed training pipeline?", k=5)
print(context)
```

---

## 🎯 Quickstart: Fine-Tuning (MLX LoRA)

Train state-of-the-art models on your own data without wrestling with CUDA errors or dataset corruption.

```python
from lokum_engine import FinetuneEngineFab

# Initialize the engine
ft = FinetuneEngineFab(model_path="/path/to/mlx/base-model")

# Safely presplit the dataset to avoid OOMs while perfectly preserving ChatML tags
ft.presplit_dataset(
    dataset_path="/path/to/raw/data", 
    max_seq_length=2048, 
    batch_size=4
)

# Launch the highly optimized training loop
process = ft.start_training(
    dataset_path="/path/to/raw/data",
    batch_size=4,
    num_layers=16,
    iters=1000,
)

print(f"🚀 Training launched successfully! PID: {process.pid}")
```

---

## 🎛️ Quality Profiles: The Magic of Lokum

Stop guessing hyper-parameters. Lokum Engine ships with three tuned profiles for both RAG and Fine-Tuning:

| Profile | Target Audience | Focus | RAG Behavior | Fine-Tune Behavior |
|---------|-----------------|-------|--------------|--------------------|
| `Base` | Local Devs | Speed & Efficiency | Lighter embedding models, faster retrieval | Smaller batch sizes, faster epochs |
| `Mid` | Startups | The Sweet Spot | Balanced chunking and embedding | Standard LoRA parameters |
| `Fab` | Enterprises | Maximum Quality | Heavy embeddings, hybrid search, semantic chunking | High-layer targeting, max context length |

---

## 📚 Comprehensive Documentation

Want to learn how to generate a DPO dataset? Curious about how Reciprocal Rank Fusion works under the hood? Need to deduplicate a massive dataset before training?

👉 **[Read the Extensive User Guide Here](docs/USER_GUIDE.md)**

---

## 🤝 Contributing & Community

Lokum Engine is built by developers, for developers. We welcome PRs, issues, and ideas. 
If this project helped you build something awesome, **please leave a ⭐ on GitHub!** It helps the community grow.

## 📜 License

MIT License - free for indie hackers and Fortune 500s alike.
