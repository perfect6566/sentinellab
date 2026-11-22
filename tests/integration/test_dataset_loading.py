from __future__ import annotations

import unittest
from pathlib import Path

from ml.dataset import load_partition


@unittest.skipUnless(
    Path("data/raw/brazilian-malware.csv").is_file(),
    "full raw dataset is intentionally not stored in Git",
)
class DatasetLoadingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.development = load_partition("development")

    def test_sealed_development_partition_shape(self) -> None:
        self.assertEqual(self.development.features.shape, (38122, 25))
        self.assertEqual(len(self.development.labels), 38122)
        self.assertEqual(len(self.development.groups), 38122)

    def test_forbidden_columns_are_absent(self) -> None:
        forbidden = {"Label", "SHA1", "FirstSeenDate"}
        self.assertFalse(forbidden & set(self.development.features.columns))

    def test_development_groups_match_sealed_count(self) -> None:
        self.assertEqual(len(set(self.development.groups)), 34651)


if __name__ == "__main__":
    unittest.main()
