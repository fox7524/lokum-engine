import unittest

from lokum_engine.rag.engine import RAGEngine


class TestRagPersistenceTruthfulness(unittest.TestCase):
    def test_mark_deleted_returns_false_and_sets_error_when_state_save_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.state = {"version": 1, "files": {}}
        eng.state_path = "/tmp/rag_state.json"
        eng.last_error = ""
        eng._file_id_for = lambda path: "fid"
        eng._load_state = lambda: None

        def boom(path, obj):
            raise OSError("disk full")

        eng._atomic_write_json = boom

        ok = eng.mark_deleted("/tmp/a.txt", deleted=True)

        self.assertFalse(ok)
        self.assertIn("disk full", eng.last_error)

    def test_save_index_raises_when_meta_write_fails(self):
        eng = RAGEngine.__new__(RAGEngine)
        eng.index = object()
        eng.documents = ["a"]
        eng.chunk_meta = []
        eng.index_path = "/tmp/faiss_index.bin"
        eng.docs_path = "/tmp/docs_metadata.npy"
        eng.meta_path = "/tmp/rag_meta.json"
        eng.state_path = "/tmp/rag_state.json"
        eng.indexed_folder = "/tmp/source"
        eng.state = {"version": 1, "files": {}}
        eng.last_error = ""

        eng._atomic_write_faiss = lambda path, index_obj: None
        eng._atomic_write_npy = lambda path, arr: None

        def fake_json(path, obj):
            if path.endswith("rag_meta.json"):
                raise OSError("write meta failed")

        eng._atomic_write_json = fake_json

        with self.assertRaisesRegex(RuntimeError, "write meta failed"):
            eng.save_index()


if __name__ == "__main__":
    unittest.main()
