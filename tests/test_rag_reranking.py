import unittest
import os
import shutil
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Patch them before importing RAGEngine
import sys
from unittest.mock import MagicMock
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.CrossEncoder'] = MagicMock()

from lokum_engine.rag.engine import RAGEngine

class TestRagReranking(unittest.TestCase):
    def setUp(self):
        # Patch device detection to return CPU so torch doesn't try to load MPS drivers
        self.device_patcher = patch('lokum_engine.rag.engine.RAGEngine._select_embed_device', return_value='cpu')
        self.device_patcher.start()
        
        self.mock_st = MagicMock()
        def mock_encode(texts, **kwargs):
            rng = np.random.RandomState(42)
            if isinstance(texts, str):
                return rng.rand(1, 384).astype(np.float32)
            elif hasattr(texts, "__len__"):
                n = len(texts)
                if n == 0:
                    return np.array([]).astype(np.float32)
                return rng.rand(n, 384).astype(np.float32)
            else:
                return rng.rand(1, 384).astype(np.float32)
        self.mock_st.encode.side_effect = mock_encode

        self.mock_ce = MagicMock()
        
        # Ensure that when SentenceTransformer(...) or CrossEncoder(...) is called, they return our mocks
        sys.modules['sentence_transformers'].SentenceTransformer.return_value = self.mock_st
        sys.modules['sentence_transformers'].CrossEncoder.return_value = self.mock_ce

        self.temp_dir = tempfile.mkdtemp()
        # Initialize an engine with a small model and reranker
        self.engine = RAGEngine(storage_dir=self.temp_dir, quality="mid")
        
        # We need a small mock to not download huge models during tests if possible,
        # but RAGEngine defaults to "all-MiniLM-L6-v2" and "cross-encoder/ms-marco-MiniLM-L-6-v2".
        # Let's create dummy documents.
        docs = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast dark colored fox leaps above a sleepy hound.",
            "Python is a great programming language.",
            "RAG pipelines use retrieval augmented generation.",
            "Machine learning involves training models on data."
        ]
        
        # Write dummy files to ingest
        self.test_files = []
        for i, text in enumerate(docs):
            p = os.path.join(self.temp_dir, f"doc_{i}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            self.test_files.append(p)
            
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.device_patcher.stop()
        
    def test_reranking_changes_order(self):
        """Test that CrossEncoder scoring reorders the raw FAISS distances."""
        # Force the mock cross encoder to return specific scores
        # We will feed it 2 documents. We want it to score the second doc much higher.
        self.mock_ce.predict.return_value = np.array([-10.0, 10.0], dtype=np.float32)

        # Ingest documents
        self.engine.ingest_documents(self.test_files)
        
        # First query without reranking (bypass reranker)
        original_reranker = self.engine.cross_encoder
        self.engine.cross_encoder = None
        res_no_rerank = self.engine.query_with_sources("fast fox", k=2)
        chunks_no_rerank = res_no_rerank["chunks"]
        
        # Restore reranker
        self.engine.cross_encoder = original_reranker
        if self.engine.cross_encoder is None:
            self.skipTest("CrossEncoder not loaded (maybe sentence-transformers missing or no internet).")
            
        res_rerank = self.engine.query_with_sources("fast fox", k=2)
        chunks_rerank = res_rerank["chunks"]
        
        # It's possible the top result is the same, but we want to test that reranking code is executed.
        # We can mock the cross_encoder.predict to return specific scores to force a reorder.
        class MockCrossEncoder:
            def predict(self, pairs):
                scores = []
                for query, text in pairs:
                    if "Python" in text:
                        scores.append(100.0) # artificially boost
                    else:
                        scores.append(0.0)
                return np.array(scores)
                
        self.engine.cross_encoder = MockCrossEncoder()
        
        # Now query again. "fast fox" should match the fox documents in FAISS, but let's 
        # add a doc with "Python" in it. Wait, if "Python" doc is not in top FAISS candidates, 
        # it won't be reranked. Let's set k large enough so FAISS fetches it.
        # Actually, let's just make the MockCrossEncoder reverse the order of whatever is passed.
        class ReverseCrossEncoder:
            def predict(self, pairs):
                # Return scores such that the first pair gets lowest score, last gets highest
                return np.array([float(i) for i in range(len(pairs))])
                
        self.engine.cross_encoder = ReverseCrossEncoder()
        # We need rerank_multiplier > 1 to fetch more candidates than k
        self.engine.rerank_multiplier = 2
        
        res_mock = self.engine.query_with_sources("fast fox", k=2)
        chunks_mock = res_mock["chunks"]
        
        # Because ReverseCrossEncoder gives highest score to the last candidate, 
        # the candidates at the end of the FAISS results (which are worst for FAISS) 
        # will become best for Reranker.
        # Thus, chunks_mock should be completely different or reversed compared to chunks_no_rerank.
        self.assertNotEqual(chunks_no_rerank, chunks_mock)

        # Now test RAGReaderEngine in the same test to avoid re-initializing models
        from lokum_engine.rag.reader_engine import RAGReaderEngine
        
        print("Creating reader...")
        reader = RAGReaderEngine(storage_dir=self.temp_dir, quality="mid")
        reader.load()
        reader.embedding_model = self.engine.embedding_model
        
        # Bypass reranker
        reader.cross_encoder = None
        print("Running search without reranker...")
        res_no_rerank = reader.search("fast fox", k=2)
        chunks_no_rerank = res_no_rerank["chunks"]
        
        reader.cross_encoder = ReverseCrossEncoder()
        reader.rerank_multiplier = 2
        
        print("Running search with reranker...")
        res_mock = reader.search("fast fox", k=2)
        chunks_mock = res_mock["chunks"]
        
        print("Asserting...")
        self.assertNotEqual(chunks_no_rerank, chunks_mock)
        print("Done!")

if __name__ == "__main__":
    unittest.main()
