import json
import os
import tempfile
import unittest

from lokum_engine.finetune.engine import _presplit_jsonl_file


class TestFinetunePresplitChatML(unittest.TestCase):
    def test_presplit_never_breaks_tags(self):
        # Force small limit so we actually split.
        os.environ["LOKUMAI_FT_PRESPLIT_CHARS_PER_TOKEN"] = "1.0"

        chatml = (
            "<|im_start|>system\nS\n<|im_end|>\n"
            "<|im_start|>user\n" + ("U " * 400) + "\n<|im_end|>\n"
            "<|im_start|>assistant\n" + ("A " * 400) + "\n<|im_end|>\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "train.jsonl")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": chatml}) + "\n")

            changed = _presplit_jsonl_file(fp, max_seq_length=64, batch_size=1)
            self.assertGreaterEqual(int(changed), 1)

            with open(fp, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            self.assertGreater(len(lines), 1)
            for ln in lines:
                obj = json.loads(ln)
                txt = obj["text"]
                # Basic sanity: tags should remain intact and in order.
                self.assertIn("<|im_start|>system", txt)
                self.assertTrue(txt.strip().endswith("<|im_end|>"))
                self.assertNotIn(
                    "<|im_start|>system\nS\n<|im_end|>\n<|im_start|>",
                    txt.replace("<|im_start|>system\nS\n<|im_end|>\n", "", 1),
                )


if __name__ == "__main__":
    unittest.main()

