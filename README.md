# lokum-engine

[![PyPI](https://img.shields.io/pypi/v/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)
[![Python](https://img.shields.io/pypi/pyversions/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)
[![License](https://img.shields.io/pypi/l/lokum-engine.svg)](https://pypi.org/project/lokum-engine/)

Developer-first building blocks extracted from **LokumAI**:

- **RAGEngine**: persistent retrieval (FAISS + sentence-transformers) over local files
- **FinetuneEngine**: MLX LoRA runner utilities + ChatML-safe dataset presplitting

Repo: https://github.com/fox7524/lokum-engine

---

## Install

```bash
pip install lokum-engine
```

This package intentionally includes “heavy” deps (FAISS, sentence-transformers, PyMuPDF, MLX, …).

---

## Quickstart (RAG)

```python
from lokum_engine import RAGEngineMid

rag = RAGEngineMid()  # base | mid | fab
rag.ingest_folder("/path/to/your/docs", recursive=True)

ctx = rag.query("What is this project about?", k=5)
print(ctx)
```

### Quality profiles (RAG)

RAG has 3 preset profiles:

- **base**: faster / lighter defaults
- **mid**: balanced defaults (matches the original engine behavior)
- **fab**: quality-oriented (more aggressive retrieval + heavier embedding model)

Use whichever style you prefer:

```python
from lokum_engine import RAGEngineBase, RAGEngineMid, RAGEngineFab

rag_fast = RAGEngineBase()
rag_balanced = RAGEngineMid()
rag_best = RAGEngineFab()
```

---

## Persistence & storage paths

By default RAG state is stored under `~/.lokumai/rag`.

Override with env vars:

- `LOKUMAI_HOME` — base app folder (default: `~/.lokumai`)
- `LOKUMAI_RAG_DIR` — RAG store dir (default: `~/.lokumai/rag`)

Tip: If you change the embedding model later, old FAISS indexes might not be compatible (different vector dim).

---

## Configuration (RAG)

- `LOKUMAI_RAG_QUALITY` — `base|mid|fab` (if you don’t pass `quality=` in code)
- `LOKUMAI_EMBED_MODEL` — override embedding model name (HuggingFace / sentence-transformers)
- `LOKUMAI_EMBED_DEVICE` — `cpu` or `mps` (auto-detects MPS if available)
- `LOKUMAI_EMBED_BATCH` — embedding batch size
- `LOKUMAI_RAG_CHECKPOINT_CHUNKS` — periodic save threshold (chunk count)
- `LOKUMAI_RAG_CHECKPOINT_SECS` — periodic save threshold (seconds)

---

## Quickstart (MLX LoRA fine-tuning)

```python
from lokum_engine import FinetuneEngineMid

ft = FinetuneEngineMid(model_path="/path/to/mlx/model")

# (Optional) build a basic dataset from raw text chunks
train_fp, valid_fp = ft.prepare_dataset(["some text chunk", "another chunk"])

# Recommended: presplit to avoid OOM & never cut ChatML tags
ft.presplit_dataset(ft.dataset_dir, max_seq_length=512, batch_size=2)

proc = ft.start_training(
    dataset_path=ft.dataset_dir,
    batch_size=2,
    num_layers=16,
    iters=100,
)
print("PID:", proc.pid)
```

### Quality profiles (Fine-tune)

Fine-tune also has 3 preset profiles:

```python
from lokum_engine import FinetuneEngineBase, FinetuneEngineMid, FinetuneEngineFab

ft_fast = FinetuneEngineBase(model_path="...")
ft_balanced = FinetuneEngineMid(model_path="...")
ft_best = FinetuneEngineFab(model_path="...")
```

Notes:
- **fab** is more aggressive and can OOM depending on model + hardware. Use env overrides to dial it down.

### Configuration (Fine-tune)

- `LOKUMAI_FT_QUALITY` — `base|mid|fab`
- `LOKUMAI_FT_PRESPLIT` — `1|0`
- `LOKUMAI_FT_PRESPLIT_CHARS_PER_TOKEN` — presplit aggressiveness (lower = more splitting)
- `LOKUMAI_FT_MAX_SEQ_LENGTH`
- `LOKUMAI_FT_CLEAR_CACHE_THRESHOLD`
- `LOKUMAI_FT_STEPS_PER_EVAL`
- `LOKUMAI_FT_VAL_BATCHES`
- `LOKUMAI_FT_GRAD_CHECKPOINT` — `1|0`

---

## Troubleshooting

### `RAGEngine.enabled == False`
RAG requires (at minimum):
- `sentence-transformers`
- `faiss-cpu`

### OCR returns empty text
`pytesseract` needs the system `tesseract` binary installed.

macOS:
```bash
brew install tesseract
```

---

## License

MIT
