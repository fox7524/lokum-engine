import os
import tempfile
import unittest
import json
import sys

import numpy as np

from lokum_engine.rag.reader_engine import RAGReaderEngine

reader_engine = sys.modules[RAGReaderEngine.__module__]


class _StubEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        n = len(texts)
        out = np.zeros((n, 3), dtype="float32")
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t or ""))
            out[i, 1] = float(i)
            out[i, 2] = 1.0
        return out


class TestRagReaderEngine(unittest.TestCase):
    def test_requires_storage_dir(self):
        with self.assertRaises(ValueError):
            RAGReaderEngine(storage_dir="")

    def test_missing_store_files_raises(self):
        with tempfile.TemporaryDirectory() as td:
            eng = RAGReaderEngine(storage_dir=td, quality="mid")
            with self.assertRaises(FileNotFoundError):
                eng.load()

    def test_can_load_docs_without_faiss(self):
        # Even if FAISS is missing in the environment, we should at least be able
        # to load docs and list chunks (retrieval will be disabled).
        with tempfile.TemporaryDirectory() as td:
            docs_path = os.path.join(td, "docs_metadata.npy")
            np.save(docs_path, np.array(["a", "b", "c"], dtype=object))
            prev_has = reader_engine.HAS_FAISS
            reader_engine.HAS_FAISS = False
            try:
                eng = RAGReaderEngine(storage_dir=td, quality="mid")
                eng.load()
                self.assertEqual(eng.list_chunks(limit=2), ["a", "b"])
            finally:
                reader_engine.HAS_FAISS = prev_has

    def test_search_skips_deleted_sources_from_state(self):
        if not getattr(reader_engine, "HAS_FAISS", False):
            self.skipTest("faiss not available")

        with tempfile.TemporaryDirectory() as td:
            docs_path = os.path.join(td, "docs_metadata.npy")
            chunks_meta_path = os.path.join(td, "chunks_meta.npy")
            index_path = os.path.join(td, "faiss_index.bin")
            state_path = os.path.join(td, "rag_state.json")

            deleted_text = "D" * 30
            active_text = "A" * 200
            deleted_file_id = "deleted-file"
            active_file_id = "active-file"

            np.save(docs_path, np.array([deleted_text, active_text], dtype=object))
            np.save(
                chunks_meta_path,
                np.array(
                    [
                        {"file_id": deleted_file_id, "source_path": "/tmp/deleted.txt"},
                        {"file_id": active_file_id, "source_path": "/tmp/active.txt"},
                    ],
                    dtype=object,
                ),
            )

            index = reader_engine.faiss.IndexFlatL2(3)
            index.add(
                np.array(
                    [
                        [float(len(deleted_text)), 0.0, 1.0],
                        [float(len(active_text)), 0.0, 1.0],
                    ],
                    dtype="float32",
                )
            )
            reader_engine.faiss.write_index(index, index_path)

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "files": {
                            deleted_file_id: {
                                "source_path": "/tmp/deleted.txt",
                                "deleted": True,
                            },
                            active_file_id: {
                                "source_path": "/tmp/active.txt",
                                "deleted": False,
                            },
                        },
                    },
                    f,
                )

            prev_cls = reader_engine.SentenceTransformer
            prev_has = reader_engine.HAS_SENTENCE_TRANSFORMERS
            reader_engine.SentenceTransformer = lambda *args, **kwargs: _StubEmbedder()
            reader_engine.HAS_SENTENCE_TRANSFORMERS = True
            try:
                eng = RAGReaderEngine(storage_dir=td, quality="mid")
                result = eng.search("Q" * 30, k=5)
            finally:
                reader_engine.SentenceTransformer = prev_cls
                reader_engine.HAS_SENTENCE_TRANSFORMERS = prev_has

            self.assertEqual(result["chunks"], [active_text])

    def test_embedding_model_mismatch_raises_error(self):
        if not getattr(reader_engine, "HAS_FAISS", False):
            self.skipTest("faiss not available")
            
        with tempfile.TemporaryDirectory() as td:
            docs_path = os.path.join(td, "docs_metadata.npy")
            index_path = os.path.join(td, "faiss_index.bin")
            meta_path = os.path.join(td, "rag_meta.json")
            
            np.save(docs_path, np.array(["chunk1"], dtype=object))
            index = reader_engine.faiss.IndexFlatL2(3)
            index.add(np.array([[1.0, 0.0, 1.0]], dtype="float32"))
            reader_engine.faiss.write_index(index, index_path)
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"folder": "/tmp/test", "model": "model-A"}, f)
                
            eng = RAGReaderEngine(storage_dir=td, quality="mid")
            # Override to make sure it thinks we are initializing with model-B
            eng.quality_profile = reader_engine.get_rag_quality_profile("mid")
            
            from lokum_engine.rag.engine import RAGQualityProfile
            eng.quality_profile = RAGQualityProfile(
                name="test",
                chunk_size=700,
                overlap=80,
                embedding_model="model-B",
                fetch_multiplier=6,
                fetch_min=30,
                fetch_cap=200,
            )
            
            with self.assertRaisesRegex(ValueError, "Embedding model mismatch"):
                eng.load()

if __name__ == "__main__":

    unittest.main()
