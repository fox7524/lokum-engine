import json
import os
import tempfile
import unittest

from lokum_engine.finetune.engine import FinetuneEngine


class TestFinetuneValidation(unittest.TestCase):
    def test_start_training_rejects_missing_train_file(self):
        with tempfile.TemporaryDirectory() as td:
            eng = FinetuneEngine.__new__(FinetuneEngine)
            eng.model_path = "/tmp/model"
            eng.dataset_dir = td
            eng.quality_profile = type(
                "_Prof",
                (),
                {
                    "batch_size": 1,
                    "num_layers": 8,
                    "iters": 10,
                    "grad_checkpoint": True,
                    "val_batches": 1,
                    "steps_per_eval": 10,
                    "max_seq_length": 128,
                    "clear_cache_threshold": 1.0,
                    "presplit_chars_per_token": 4.0,
                },
            )()

            with self.assertRaisesRegex(RuntimeError, "train.jsonl"):
                eng.start_training(dataset_path=td)

    def test_start_training_rejects_row_without_text(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "train.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"bad": "row"}) + "\n")
            with open(os.path.join(td, "valid.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "ok"}) + "\n")

            eng = FinetuneEngine.__new__(FinetuneEngine)
            eng.model_path = "/tmp/model"
            eng.dataset_dir = td
            eng.quality_profile = type(
                "_Prof",
                (),
                {
                    "batch_size": 1,
                    "num_layers": 8,
                    "iters": 10,
                    "grad_checkpoint": True,
                    "val_batches": 1,
                    "steps_per_eval": 10,
                    "max_seq_length": 128,
                    "clear_cache_threshold": 1.0,
                    "presplit_chars_per_token": 4.0,
                },
            )()

            with self.assertRaisesRegex(RuntimeError, "text"):
                eng.start_training(dataset_path=td)


if __name__ == "__main__":
    unittest.main()
