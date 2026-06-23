# 🌟 Lokum Engine Master Roadmap 🌟

Our mission is to make `lokum-engine` the **undisputed king of the industry** for RAG and LLM Fine-Tuning. We are building the engine that both indie developers and Fortune 500 enterprises will trust and love. 

This roadmap transitions `lokum-engine` from a reliable local library to a distributed, SOTA (State-of-the-Art) enterprise platform.

---

## 🟢 Phase 1: Core Reliability (Completed ✅)
*The foundation. The engine must never lie, never fail silently, and never corrupt data.*

- [x] **Truthful Error Reporting:** No silent failures in retrieval or persistence.
- [x] **Deleted File Reconciliation:** No zombie chunks from deleted files.
- [x] **Store Compatibility:** Strict embedding model version checks to prevent index corruption.
- [x] **Dataset Preflight:** Strict validation of JSONL datasets before fine-tuning starts.
- [x] **Unified Ingestion:** Consistent file parsing across single files and directories.

---

## 🟡 Phase 2: Advanced Retrieval & Quality (Next Up 🚀)
*The "Smart" phase. Making RAG results incredibly accurate using SOTA techniques.*

- [ ] **Hybrid Search (BM25 + Dense):** Combine exact keyword matching with semantic similarity for enterprise-grade recall.
- [ ] **Re-ranking Pipeline:** Integrate Cohere, BGE-Reranker, or Cross-Encoders to sort the top-k results for maximum precision.
- [ ] **Semantic & Hierarchical Chunking:** Stop splitting sentences in half. Parse markdown, code syntax trees (AST), and document headers intelligently.
- [ ] **RAG Eval Harness:** Built-in LLM-as-a-judge (RAGAS style) to automatically measure recall, precision, and faithfulness on datasets.
- [ ] **Metadata Filtering:** Add explicit metadata filters (e.g., `date > 2024`, `author = fox`) directly into the search engine.

---

## 🟠 Phase 3: Developer UX & CLI
*Making the engine a joy to use. "Developer love" drives adoption.*

- [ ] **Beautiful CLI:** `lokum ingest ./docs`, `lokum query "what is x"`, `lokum train --dataset ./data`.
- [ ] **Progress Bars & Rich Output:** Use `rich` for gorgeous, colorful terminal outputs during long ingestions and training loops.
- [ ] **Dataset Inspector:** CLI tools to validate, preview, and clean JSONL training data instantly.
- [ ] **Auto-Tuning:** The engine analyzes the user's hardware (VRAM/RAM) and automatically sets the perfect batch sizes and sequence lengths to prevent OOMs.

---

## 🔴 Phase 4: Enterprise Scale & Integrations
*The "Big Company" phase. Moving beyond local FAISS into distributed production.*

- [ ] **External Vector DB Integrations:** Drop-in support for Milvus, Qdrant, Pinecone, and pgvector. 
- [ ] **Multi-Tenant Namespaces:** Isolate RAG indexes per user or organization natively in the engine.
- [ ] **Async / Batch Ingestion Pipelines:** Queue-based ingestion (e.g., Celery/Redis) to handle millions of documents without blocking.
- [ ] **Experiment Tracking:** First-class integrations with Weights & Biases (W&B) and MLflow for fine-tune runs.
- [ ] **Auto-Compaction:** Background tasks to physically remove tombstoned/deleted chunks from the index, reclaiming storage.

---

## 🟣 Phase 5: SOTA Alignment & Production Ops
*The bleeding edge. Preparing models for real-world deployment.*

- [ ] **Advanced Alignment:** Add built-in support for DPO (Direct Preference Optimization), PPO, and ORPO, moving beyond simple SFT (Supervised Fine-Tuning).
- [ ] **LoRA Merging & Quantization:** 1-click tools to merge adapters into base models and export to GGUF, AWQ, or EXL2 formats.
- [ ] **API Server (FastAPI):** A built-in production-ready REST API for querying and training.
- [ ] **Observability (OpenTelemetry):** Native traces, spans, and metrics for every RAG query and training step (Prometheus/Grafana ready).
- [ ] **Kubernetes/Docker Ready:** Official Helm charts and Docker images for instant cloud deployment.
