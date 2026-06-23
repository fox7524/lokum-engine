import json
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from lokum_engine.finetune.engine import FinetuneEngine, _presplit_jsonl_file, logger as finetune_logger


class TestFinetuneValidation(unittest.TestCase):
    def test_logger_exists_for_finetune_engine(self):
        self.assertIsInstance(finetune_logger, logging.Logger)

    def test_presplit_jsonl_file_strict_mode_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "train.jsonl")
            with open(fp, "w", encoding="utf-8") as f:
                f.write('{"text":"ok"}\n')
                f.write("not-json\n")

            with self.assertRaisesRegex(RuntimeError, "Malformed JSONL"):
                _presplit_jsonl_file(fp, max_seq_length=128, batch_size=1, strict=True)

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

    def test_start_validation_surfaces_presplit_failure(self):
        with tempfile.TemporaryDirectory() as td:
            valid_fp = os.path.join(td, "valid.jsonl")
            with open(valid_fp, "w", encoding="utf-8") as f:
                f.write('{"text":"ok"}\n')

            eng = FinetuneEngine.__new__(FinetuneEngine)
            eng.model_path = "/tmp/model"
            eng.dataset_dir = td
            eng.quality_profile = type("_Prof", (), {"max_seq_length": 128, "clear_cache_threshold": 1.0})()

            with patch("lokum_engine.finetune.engine._presplit_jsonl_file", side_effect=RuntimeError("presplit boom")):
                with self.assertRaisesRegex(RuntimeError, "presplit boom"):
                    eng.start_validation(dataset_path=td, adapter_path="/tmp/adapter")


if __name__ == "__main__":
    unittest.main()
