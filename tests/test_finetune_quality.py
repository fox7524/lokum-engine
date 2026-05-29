import unittest

from lokum_engine.finetune.engine import get_finetune_quality_profile, normalize_finetune_quality


class TestFinetuneQuality(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_finetune_quality(None), "mid")
        self.assertEqual(normalize_finetune_quality(""), "mid")
        self.assertEqual(normalize_finetune_quality("base"), "base")
        self.assertEqual(normalize_finetune_quality("fast"), "base")
        self.assertEqual(normalize_finetune_quality("mid"), "mid")
        self.assertEqual(normalize_finetune_quality("medium"), "mid")
        self.assertEqual(normalize_finetune_quality("fab"), "fab")
        self.assertEqual(normalize_finetune_quality("fabulous"), "fab")
        self.assertEqual(normalize_finetune_quality("faboulous"), "fab")

    def test_profiles_escalate_reasonably(self):
        base = get_finetune_quality_profile("base")
        mid = get_finetune_quality_profile("mid")
        fab = get_finetune_quality_profile("fab")

        # "Fab" genelde daha çok kapasite
        self.assertLessEqual(base.num_layers, mid.num_layers)
        self.assertLessEqual(mid.num_layers, fab.num_layers)

        self.assertLessEqual(base.iters, mid.iters)
        self.assertLessEqual(mid.iters, fab.iters)

