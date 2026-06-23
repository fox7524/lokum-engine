import os
import tempfile
import unittest

import lokum_engine.rag.engine as rag_engine
from lokum_engine.rag import RAGEngine

class _StubEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        import numpy as np
        n = len(texts)
        out = np.zeros((n, 3), dtype="float32")
        return out

class TestRagExtensions(unittest.TestCase):
    def _make_engine(self, storage_dir: str):
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

    def test_ingest_folder_extensions(self):
        if not getattr(rag_engine, "HAS_FAISS", False):
            self.skipTest("faiss not available")
        with tempfile.TemporaryDirectory() as td:
            engine = self._make_engine(td)
            
            with tempfile.TemporaryDirectory() as src_dir:
                # Create files with various extensions
                supported_exts = [
                    ".pdf", ".docx", ".zim", ".jpg", ".py", ".cpp", ".c", ".h", ".hpp",
                    ".js", ".ts", ".html", ".css", ".scss", ".txt", ".md", ".json",
                    ".xml", ".yaml", ".sh", ".rs", ".go", ".java", ".sql", ".dockerfile"
                ]
                
                unsupported_exts = [
                    ".exe", ".bin", ".dll", ".so", ".dylib"
                ]
                
                for ext in supported_exts:
                    with open(os.path.join(src_dir, f"test{ext}"), "w", encoding="utf-8") as f:
                        f.write(f"content for {ext}")
                
                for ext in unsupported_exts:
                    with open(os.path.join(src_dir, f"test{ext}"), "w", encoding="utf-8") as f:
                        f.write(f"binary content for {ext}")
                
                engine.ingest_folder(src_dir, recursive=False)
                
                # Check that only supported files were ingested
                files_in_state = engine.state.get("files", {})
                ingested_exts = set()
                for fid, rec in files_in_state.items():
                    if rec.get("status") == "ok":
                        ext = os.path.splitext(rec.get("source_path", ""))[1].lower()
                        ingested_exts.add(ext)
                
                # Assert that ALL supported exts are ingested (or attempted and processed)
                for ext in supported_exts:
                    # Note: pdf, docx, zim, jpg might fail extraction if dependencies are missing,
                    # but they should be *attempted*.
                    # In our test, we stubbed embedder, but extraction methods might still be called.
                    # Wait, process_file tries to extract. If extraction fails, it sets error.
                    # Let's check if they are AT LEAST in the state files, either ok or failed.
                    pass
                
                attempted_exts = set()
                for fid, rec in files_in_state.items():
                    ext = os.path.splitext(rec.get("source_path", ""))[1].lower()
                    attempted_exts.add(ext)
                
                for ext in supported_exts:
                    self.assertIn(ext, attempted_exts, f"Extension {ext} was not picked up by ingest_folder")
                
                for ext in unsupported_exts:
                    self.assertNotIn(ext, attempted_exts, f"Extension {ext} should not be picked up")

if __name__ == "__main__":
    unittest.main()
