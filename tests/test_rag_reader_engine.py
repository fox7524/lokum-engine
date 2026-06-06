import os
import tempfile
import unittest

import numpy as np

from lokum_engine.rag.reader_engine import RAGReaderEngine


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
            eng = RAGReaderEngine(storage_dir=td, quality="mid")
            eng.load()
            self.assertEqual(eng.list_chunks(limit=2), ["a", "b"])


if __name__ == "__main__":
    unittest.main()

