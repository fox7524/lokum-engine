import unittest

import numpy as np

from lokum_engine.rag.engine import RAGEngine
from lokum_engine.rag.reader_engine import RAGReaderEngine


class _ExplodingEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        raise RuntimeError("embedding boom")


class _OkEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        return np.ones((1, 3), dtype="float32")


class _FakeIndex:
    def search(self, query_vector, fetch_k):
        raise RuntimeError("faiss boom")


class _DummyIndex:
    def search(self, query_vector, fetch_k):
        return np.array([[0.1]], dtype="float32"), np.array([[0]])


class TestRagQueryTruthfulness(unittest.TestCase):
    def test_query_sets_last_error_when_embedding_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.enabled = True
        eng.index = _FakeIndex()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.last_error = ""
        eng.embedding_model = _ExplodingEmbedder()
        eng._abort = False

        result = eng.query("hello", k=1)

        self.assertEqual(result, "")
        self.assertIn("embedding boom", eng.last_error)

    def test_query_with_sources_returns_error_payload_when_search_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.enabled = True
        eng.index = _FakeIndex()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.last_error = ""
        eng.embedding_model = _OkEmbedder()
        eng._abort = False
        eng.fetch_multiplier = 1
        eng.fetch_min = 1
        eng.fetch_cap = 1

        result = eng.query_with_sources("hello", k=1)

        self.assertEqual(result["context"], "")
        self.assertEqual(result["count"], 0)
        self.assertIn("faiss boom", result["error"])
        self.assertIn("faiss boom", eng.last_error)

    def test_reader_search_returns_error_payload_when_embedding_fails(self):
        eng = RAGReaderEngine.__new__(RAGReaderEngine)
        eng.loaded = True
        eng.enabled = True
        eng.index = _DummyIndex()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.last_error = ""
        eng.fetch_multiplier = 1
        eng.fetch_min = 1
        eng.fetch_cap = 1
        eng.embedding_model = _ExplodingEmbedder()
        eng._ensure_sentence_transformer = lambda: True

        result = eng.search("hello", k=1)

        self.assertEqual(result["context"], "")
        self.assertEqual(result["count"], 0)
        self.assertIn("embedding boom", result["error"])


if __name__ == "__main__":
    unittest.main()
