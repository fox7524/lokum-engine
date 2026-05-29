import os
import unittest

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
        self.assertEqual(str(lokumai_home()), "/tmp/lokumai_home_test")
        self.assertEqual(str(rag_dir()), "/tmp/lokumai_home_test/rag")
        self.assertEqual(str(lora_dir()), "/tmp/lokumai_home_test/lora_data")
        self.assertEqual(str(models_dir()), "/tmp/lokumai_home_test/models")

