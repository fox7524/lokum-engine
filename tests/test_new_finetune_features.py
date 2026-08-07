import os
import json
import tempfile
import unittest

from lokum_engine.finetune.engine import FinetuneEngine
from lokum_engine.finetune.curation import deduplicate_dataset, auto_score_dataset

class TestNewFinetuneFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Ensure we don't pollute the real lora_dir
        os.environ["LOKUMAI_LORA_DIR"] = self.temp_dir.name
        self.engine = FinetuneEngine("dummy_model")
        
    def tearDown(self):
        self.temp_dir.cleanup()
        if "LOKUMAI_LORA_DIR" in os.environ:
            del os.environ["LOKUMAI_LORA_DIR"]

    def test_prepare_preference_dataset(self):
        pairs = [
            {"prompt": "Hello", "chosen": "Hi there", "rejected": "Go away"},
            {"prompt": "2+2", "chosen": "4", "rejected": "5"}
        ]
        
        train_path, valid_path = self.engine.prepare_preference_dataset(pairs)
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(valid_path))
        
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 0)
            obj = json.loads(lines[0])
            self.assertIn("prompt", obj)
            self.assertIn("chosen", obj)
            self.assertIn("rejected", obj)

    def test_deduplicate_dataset(self):
        input_file = os.path.join(self.temp_dir.name, "input.jsonl")
        output_file = os.path.join(self.temp_dir.name, "output.jsonl")
        
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "Hello world"}) + "\n")
            f.write(json.dumps({"text": "Hello world"}) + "\n") # duplicate
            f.write(json.dumps({"text": "Different text"}) + "\n")
            
        stats = deduplicate_dataset(input_file, output_file)
        self.assertEqual(stats["original_count"], 3)
        self.assertEqual(stats["deduplicated_count"], 2)
        self.assertEqual(stats["removed_count"], 1)

    def test_auto_score_dataset(self):
        input_file = os.path.join(self.temp_dir.name, "input.jsonl")
        output_file = os.path.join(self.temp_dir.name, "output.jsonl")
        
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "Good response"}) + "\n")
            f.write(json.dumps({"text": "Bad response"}) + "\n")
            
        def mock_score(text):
            return 0.9 if "Good" in text else 0.1
            
        stats = auto_score_dataset(input_file, output_file, scoring_fn=mock_score, threshold=0.5)
        self.assertEqual(stats["original_count"], 2)
        self.assertEqual(stats["kept_count"], 1)
        self.assertEqual(stats["removed_count"], 1)

if __name__ == "__main__":
    unittest.main()
