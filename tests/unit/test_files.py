from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.files import FileStoreError, SessionFileStore, new_session_id


class FileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = SessionFileStore(Path(self.temporary_directory.name), max_bytes=100)
        self.session = new_session_id()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_file_reference_is_session_owned(self) -> None:
        stored = self.store.put_csv(self.session, "input.csv", b"a,b\n1,2\n")
        self.assertEqual(
            self.store.resolve(self.session, stored.reference).path.read_bytes(),
            b"a,b\n1,2\n",
        )
        with self.assertRaisesRegex(FileStoreError, "does not belong"):
            self.store.resolve(new_session_id(), stored.reference)

    def test_path_traversal_non_csv_and_size_are_rejected(self) -> None:
        with self.assertRaises(FileStoreError):
            self.store.put_csv(self.session, "../input.csv", b"a\n1\n")
        with self.assertRaises(FileStoreError):
            self.store.put_csv(self.session, "input.exe", b"a\n1\n")
        with self.assertRaisesRegex(FileStoreError, "exceeds"):
            self.store.put_csv(self.session, "input.csv", b"x" * 101)

    def test_malformed_duplicate_header_and_row_limit_are_rejected(self) -> None:
        with self.assertRaisesRegex(FileStoreError, "duplicate"):
            self.store.put_csv(self.session, "duplicate.csv", b"a,a\n1,2\n")
        with self.assertRaisesRegex(FileStoreError, "does not match"):
            self.store.put_csv(self.session, "malformed.csv", b"a,b\n1\n")
        limited = SessionFileStore(Path(self.temporary_directory.name) / "limited", max_rows=1)
        with self.assertRaisesRegex(FileStoreError, "row limit"):
            limited.put_csv(self.session, "large.csv", b"a\n1\n2\n")

    def test_expired_files_are_removed(self) -> None:
        stored = self.store.put_csv(self.session, "old.csv", b"a\n1\n")
        metadata_path = stored.path.with_suffix(".json")
        metadata = metadata_path.read_text(encoding="utf-8").replace(
            '"created_at": ', '"created_at": -1000000000, "original_created_at": '
        )
        metadata_path.write_text(metadata, encoding="utf-8")

        removed = self.store.purge_expired(60)

        self.assertEqual(removed, 1)
        self.assertFalse(stored.path.exists())


if __name__ == "__main__":
    unittest.main()
