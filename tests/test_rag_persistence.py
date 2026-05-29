import json
import os
import tempfile
import unittest

import numpy as np

import lokum_engine.rag.engine as rag_engine
from lokum_engine.rag import RAGEngine


class _StubEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        n = len(texts)
        out = np.zeros((n, 3), dtype="float32")
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t or ""))
            out[i, 1] = float(i)
            out[i, 2] = 1.0
        return out


class TestRagPersistence(unittest.TestCase):
    def _make_engine(self, storage_dir: str):
        # __init__ içindeki ağır bağımlılıkları ve model indirmeyi bypass etmek için:
        eng = RAGEngine.__new__(RAGEngine)
        eng.enabled = True
        eng.embedding_model = _StubEmbedder()
        eng.index = None
        eng.documents = []
        eng.chunk_meta = []
        eng.last_error = ""
        eng.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(eng.storage_dir, exist_ok=True)
        eng.index_path = os.path.join(eng.storage_dir, "faiss_index.bin")
        eng.docs_path = os.path.join(eng.storage_dir, "docs_metadata.npy")
        eng.meta_path = os.path.join(eng.storage_dir, "rag_meta.json")
        eng.chunks_meta_path = os.path.join(eng.storage_dir, "chunks_meta.npy")
        eng.state_path = os.path.join(eng.storage_dir, "rag_state.json")
        eng.staging_dir = os.path.join(eng.storage_dir, "staging")
        os.makedirs(eng.staging_dir, exist_ok=True)
        eng.indexed_folder = ""
        eng.state = {"version": 1, "files": {}}
        return eng

    def test_index_persists_across_restart(self):
        if not getattr(rag_engine, "HAS_FAISS", False):
            self.skipTest("faiss not available")

        with tempfile.TemporaryDirectory() as td:
            eng1 = self._make_engine(td)
            src_dir = os.path.join(td, "src")
            os.makedirs(src_dir, exist_ok=True)
            p = os.path.join(src_dir, "a.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("hello world")
            eng1.indexed_folder = "/tmp/folderA"
            ok = eng1.ingest_documents([p])
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(eng1.index_path))
            self.assertTrue(os.path.exists(eng1.docs_path))
            self.assertTrue(os.path.exists(eng1.meta_path))

            eng2 = self._make_engine(td)
            eng2.load_index()
            self.assertGreater(len(eng2.documents), 0)
            with open(eng2.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(str(meta.get("folder")), "/tmp/folderA")

    def test_folder_change_preserves_existing_index(self):
        if not getattr(rag_engine, "HAS_FAISS", False):
            self.skipTest("faiss not available")

        with tempfile.TemporaryDirectory() as td:
            folder_a = os.path.join(td, "a")
            folder_b = os.path.join(td, "b")
            os.makedirs(folder_a, exist_ok=True)
            os.makedirs(folder_b, exist_ok=True)

            a_txt = os.path.join(folder_a, "x.txt")
            b_txt = os.path.join(folder_b, "y.txt")
            with open(a_txt, "w", encoding="utf-8") as f:
                f.write("AAA")
            with open(b_txt, "w", encoding="utf-8") as f:
                f.write("BBB")

            eng = self._make_engine(td)
            eng.ingest_folder(folder_a, recursive=True)
            self.assertEqual(os.path.abspath(folder_a), os.path.abspath(eng.indexed_folder))
            self.assertTrue(any("AAA" in c for c in eng.documents))

            eng.ingest_folder(folder_b, recursive=True)
            self.assertEqual(os.path.abspath(folder_b), os.path.abspath(eng.indexed_folder))
            self.assertTrue(any("AAA" in c for c in eng.documents))
            self.assertTrue(any("BBB" in c for c in eng.documents))

