from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ml.split_data import grouped_stratified_split, prepare_rows


class SplitDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def dataset(self) -> Path:
        path = self.directory / "dataset.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "SHA1", "Label"])
            for label in (0, 1):
                for group in range(20):
                    writer.writerow([f"{label}-{group}", f"sha-{label}-{group}", label])
            writer.writerow(["duplicate", "same", 0])
            writer.writerow(["duplicate", "same", 0])
            writer.writerow(["conflict-a", "conflict", 0])
            writer.writerow(["conflict-b", "conflict", 1])
        return path

    def test_conflicts_and_exact_duplicates_are_removed(self) -> None:
        _, rows, stats, conflicts = prepare_rows(self.dataset())

        self.assertEqual(conflicts, ["conflict"])
        self.assertEqual(stats.conflicting_rows_removed, 2)
        self.assertEqual(stats.exact_duplicate_rows_removed, 1)
        self.assertEqual(len(rows), 41)

    def test_split_is_reproducible_group_disjoint_and_stratified(self) -> None:
        _, rows, _, _ = prepare_rows(self.dataset())

        first = grouped_stratified_split(rows, test_fraction=0.2, seed=123)
        second = grouped_stratified_split(rows, test_fraction=0.2, seed=123)

        self.assertEqual(first, second)
        partition_by_sha: dict[str, set[str]] = {}
        test_counts = {0: 0, 1: 0}
        total_counts = {0: 0, 1: 0}
        for row in rows:
            partition_by_sha.setdefault(row.sha1, set()).add(first[row.row_id])
            total_counts[row.label] += 1
            if first[row.row_id] == "test":
                test_counts[row.label] += 1
        self.assertTrue(all(len(partitions) == 1 for partitions in partition_by_sha.values()))
        for label in (0, 1):
            self.assertAlmostEqual(test_counts[label] / total_counts[label], 0.2, delta=0.06)


if __name__ == "__main__":
    unittest.main()
