import os
import unittest
from pathlib import Path

from lokum_engine.paths import lokumai_home, lora_dir, models_dir, rag_dir


class TestPaths(unittest.TestCase):
    def setUp(self) -> None:
        # Keep env changes local to each test.
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_overrides(self):
        os.environ["LOKUMAI_HOME"] = "/tmp/lokumai_home_test"
        expected_home = str(Path("/tmp/lokumai_home_test").expanduser().resolve())
        self.assertEqual(str(lokumai_home()), expected_home)
        self.assertEqual(str(rag_dir()), str((Path(expected_home) / "rag").resolve()))
        self.assertEqual(str(lora_dir()), str((Path(expected_home) / "lora_data").resolve()))
        self.assertEqual(str(models_dir()), str((Path(expected_home) / "models").resolve()))
