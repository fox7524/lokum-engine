import os
import tempfile
import unittest
import numpy as np

import lokum_engine.rag.engine as rag_engine
from lokum_engine.rag.engine import RAGEngine

class _StubEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        n = len(texts)
        # Create dummy embeddings of dimension 3
        out = np.ones((n, 3), dtype="float32")
        return out

class TestNewRagFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = RAGEngine.__new__(RAGEngine)
        self.engine.enabled = True
        self.engine.embedding_model = _StubEmbedder()
        self.engine.index = None
        self.engine.documents = []
        self.engine.chunk_meta = []
        self.engine.last_error = ""
        self.engine.storage_dir = os.path.abspath(self.temp_dir.name)
        os.makedirs(self.engine.storage_dir, exist_ok=True)
        self.engine.index_path = os.path.join(self.engine.storage_dir, "faiss_index.bin")
        self.engine.docs_path = os.path.join(self.engine.storage_dir, "docs_metadata.npy")
        self.engine.meta_path = os.path.join(self.engine.storage_dir, "rag_meta.json")
        self.engine.chunks_meta_path = os.path.join(self.engine.storage_dir, "chunks_meta.npy")
        self.engine.state_path = os.path.join(self.engine.storage_dir, "rag_state.json")
        self.engine.staging_dir = os.path.join(self.engine.storage_dir, "staging")
        os.makedirs(self.engine.staging_dir, exist_ok=True)
        self.engine.indexed_folder = ""
        self.engine.state = {"version": 1, "files": {}}
        self.engine.bm25_index = None
        self.engine._bm25_doc_count = 0
        
        self.engine.chunk_size = 500
        self.engine.chunk_overlap = 50
        self.engine.fetch_multiplier = 2
        self.engine.fetch_min = 5
        self.engine.fetch_cap = 10
        self.engine.embed_batch_size = 32

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_semantic_chunking(self):
        # Even if NLTK is missing, it should gracefully fallback or use it if available
        text = "This is a sentence. This is another sentence! And a third one?"
        chunks = self.engine.chunk_text(text, chunk_size=25, overlap=5, semantic=True)
        self.assertTrue(len(chunks) > 0)

    def test_compact_index(self):
        if not rag_engine.HAS_FAISS:
            self.skipTest("faiss not available")
            
        import faiss
        
        # Simulate an index with 3 chunks
        self.engine.documents = ["chunk1", "chunk2", "chunk3"]
        self.engine.chunk_meta = [
            {"file_id": "file1", "active": True},
            {"file_id": "file2", "active": False}, # This one should be removed
            {"file_id": "file1", "active": True}
        ]
        
        dim = 3
        self.engine.index = faiss.IndexFlatIP(dim)
        dummy_vectors = np.ones((3, 3), dtype="float32")
        faiss.normalize_L2(dummy_vectors)
        self.engine.index.add(dummy_vectors)
        
        self.engine.state = {
            "version": 1, 
            "files": {
                "file1": {"deleted": False},
                "file2": {"deleted": True} # File2 is deleted
            }
        }
        
        removed = self.engine.compact_index()
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.engine.documents), 2)
        self.assertEqual(self.engine.documents, ["chunk1", "chunk3"])
        
    def test_hyde_document_generation(self):
        def mock_llm(prompt):
            return "HYDE_RESPONSE"
            
        hyde_doc = self.engine.generate_hyde_document("Hello?", mock_llm)
        self.assertEqual(hyde_doc, "HYDE_RESPONSE")
        
        # Without LLM
        hyde_doc2 = self.engine.generate_hyde_document("Hello?", None)
        self.assertEqual(hyde_doc2, "Hello?")

if __name__ == "__main__":
    unittest.main()
