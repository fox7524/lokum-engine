import unittest

from lokum_engine.rag.engine import get_rag_quality_profile, normalize_rag_quality


class TestRagQuality(unittest.TestCase):
    def test_normalize_quality(self):
        self.assertEqual(normalize_rag_quality(None), "mid")
        self.assertEqual(normalize_rag_quality(""), "mid")
        self.assertEqual(normalize_rag_quality("base"), "base")
        self.assertEqual(normalize_rag_quality("fast"), "base")
        self.assertEqual(normalize_rag_quality("medium"), "mid")
        self.assertEqual(normalize_rag_quality("default"), "mid")
        self.assertEqual(normalize_rag_quality("fab"), "fab")
        self.assertEqual(normalize_rag_quality("fabulous"), "fab")
        self.assertEqual(normalize_rag_quality("faboulous"), "fab")

    def test_profile_values_are_reasonable(self):
        base = get_rag_quality_profile("base")
        mid = get_rag_quality_profile("mid")
        fab = get_rag_quality_profile("fab")

        # Chunking büyüdükçe kalite/recall artar (genel beklenti)
        self.assertLess(base.chunk_size, mid.chunk_size)
        self.assertLess(mid.chunk_size, fab.chunk_size)

        # Fetch policy daha agresif olmalı
        self.assertLess(base.fetch_multiplier, mid.fetch_multiplier)
        self.assertLess(mid.fetch_multiplier, fab.fetch_multiplier)

