<div align="center">

# 🌌 Lokum Engine 🌟

**The Undisputed King of RAG & LLM Fine-Tuning**

[![PyPI](https://img.shields.io/pypi/v/lokum-engine.svg?style=for-the-badge&color=blue)](https://pypi.org/project/lokum-engine/)
[![Python](https://img.shields.io/pypi/pyversions/lokum-engine.svg?style=for-the-badge)](https://pypi.org/project/lokum-engine/)
[![License](https://img.shields.io/pypi/l/lokum-engine.svg?style=for-the-badge&color=green)](https://pypi.org/project/lokum-engine/)
[![Status](https://img.shields.io/badge/Status-Enterprise_Ready-purple?style=for-the-badge)](#)

*From local experimentation to Fortune 500 production in 3 lines of code.*

[Quickstart](#quickstart) • [Features](#why-lokum-engine) • [Roadmap](docs/ROADMAP.md) • [Documentation](https://github.com/fox7524/lokum-engine)

</div>

---

## ⚡ Why Lokum Engine?

Lokum Engine is the developer-first building block for **Retrieval-Augmented Generation (RAG)** and **State-of-the-Art LLM Fine-Tuning**. We abstracted away the infrastructure headaches, OOM crashes, and broken data pipelines so you can focus on building intelligent agents.

### 🚀 For Developers
- **Drop-in Simplicity:** Setup RAG or start an MLX LoRA training loop in 3 lines of Python.
- **Quality Profiles:** Sensible, pre-tuned defaults (`base`, `mid`, `fab`) that automatically balance speed vs. quality.
- **Fail-Fast Reliability:** Strict data validation, deleted file reconciliation, and explicit error reporting. No silent failures.

### 🏢 For Enterprises
- **ChatML-Safe Presplitting:** Guarantee your fine-tuning data never splits across critical instruction boundaries.
- **Persistent RAG State:** Robust chunk tombstoning, metadata validation, and persistent storage.
- **Hardware Aware:** Automatically detects and leverages Apple Silicon (MPS) and optimizes batch sizes.

---

## 📦 Install

```bash
pip install lokum-engine
```
*(Note: Lokum Engine intentionally includes heavy, production-grade dependencies like FAISS, sentence-transformers, PyMuPDF, and MLX out of the box).*

---

## 🧠 Quickstart: RAG (Retrieval-Augmented Generation)

Turn any folder of documents into a highly accurate semantic search engine instantly.

```python
from lokum_engine import RAGEngineFab

# Initialize with the 'Fab' profile for maximum enterprise-grade retrieval quality
rag = RAGEngineFab()  

# Recursively ingest PDFs, Markdown, Code, and text files
rag.ingest_folder("/path/to/your/enterprise/docs", recursive=True)

# Query with semantic understanding
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
    dataset_dir="/path/to/raw/data", 
    max_seq_length=2048, 
    batch_size=4
)

# Launch the training loop
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
| `Fab` | Enterprises | Maximum Quality | Heavy embeddings, aggressive retrieval | High-layer targeting, max context length |

---

## 🗺️ The Master Roadmap

We are on a mission to become the industry standard. Here is a sneak peek at what's next:
- **Hybrid Search & Re-ranking:** BM25 + Dense embeddings sorted by Cohere/BGE.
- **Enterprise Vector DBs:** Native Milvus, Pinecone, and Qdrant support.
- **RAG & Fine-Tune Eval:** Built-in LLM-as-a-judge to measure precision and recall.
- **DPO / PPO Support:** Move beyond SFT and align models with human preferences natively.

👉 **[View the full Master Roadmap here](docs/ROADMAP.md)**

---

## 🤝 Contributing & Community

Lokum Engine is built by developers, for developers. We welcome PRs, issues, and ideas. 
If this project helped you build something awesome, **please leave a ⭐ on GitHub!**

## 📜 License

MIT License - free for indie hackers and Fortune 500s alike.
