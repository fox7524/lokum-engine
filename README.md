# lokum-engine

LokumAI engines packaged as a Python library:
- RAG (FAISS + sentence-transformers)
- Fine-tuning runner (MLX LoRA) + ChatML-safe presplitting

## Install
```bash
pip install lokum-engine
```

## Usage
```python
from lokum_engine.rag import RAGEngine
from lokum_engine.finetune import FinetuneEngine
```
