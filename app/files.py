"""Session-owned server-side CSV references."""

from __future__ import annotations

import csv
import io
import json
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

CSV_NAME = re.compile(r"^[^/\\]+\.csv$", re.IGNORECASE)


class FileStoreError(ValueError):
    """Raised for invalid, missing or cross-session file references."""


@dataclass(frozen=True)
class StoredFile:
    reference: str
    original_name: str
    path: Path
    kind: str
    byte_size: int


def new_session_id() -> str:
    return str(uuid.uuid4())


def _uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise FileStoreError(f"Invalid {label}") from exc


class SessionFileStore:
    def __init__(
        self,
        root: Path = Path("instance/session-files"),
        max_bytes: int = 10 * 1024 * 1024,
        max_rows: int = 10_000,
    ) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.max_rows = max_rows

    def _validate_csv(self, content: bytes) -> None:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise FileStoreError("CSV must be UTF-8 encoded") from exc
        try:
            reader = csv.reader(io.StringIO(text), strict=True)
            header = next(reader)
            if not header or any(not name.strip() for name in header):
                raise FileStoreError("CSV header contains an empty name")
            normalized = [name.strip() for name in header]
            if len(normalized) != len(set(normalized)):
                raise FileStoreError("CSV header contains duplicate names")
            row_count = 0
            for row_count, row in enumerate(reader, start=1):
                if row_count > self.max_rows:
                    raise FileStoreError(f"CSV exceeds {self.max_rows} row limit")
                if len(row) != len(header):
                    raise FileStoreError(f"CSV row {row_count} does not match the header")
            if row_count == 0:
                raise FileStoreError("CSV contains no data rows")
        except csv.Error as exc:
            raise FileStoreError(f"CSV is malformed: {exc}") from exc

    def put_csv(
        self,
        session_id: str,
        original_name: str,
        content: bytes,
        *,
        kind: str = "upload",
    ) -> StoredFile:
        session_id = _uuid(session_id, "session ID")
        simple_name = Path(original_name).name
        if not CSV_NAME.fullmatch(simple_name) or simple_name != original_name:
            raise FileStoreError("Only simple .csv filenames are accepted")
        if not content:
            raise FileStoreError("CSV file is empty")
        if len(content) > self.max_bytes:
            raise FileStoreError(f"CSV exceeds {self.max_bytes} byte limit")
        self._validate_csv(content)
        reference = str(uuid.uuid4())
        directory = self.root / session_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{reference}.csv"
        path.write_bytes(content)
        metadata = {
            "reference": reference,
            "original_name": original_name,
            "kind": kind,
            "byte_size": len(content),
            "created_at": time.time(),
        }
        (directory / f"{reference}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return StoredFile(reference, original_name, path, kind, len(content))

    def purge_expired(self, max_age_seconds: float, *, now: float | None = None) -> int:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        current = time.time() if now is None else now
        removed = 0
        if not self.root.exists():
            return removed
        for metadata_path in self.root.glob("*/*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                created_at = float(metadata["created_at"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if current - created_at < max_age_seconds:
                continue
            data_path = metadata_path.with_suffix(".csv")
            data_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            removed += 1
            with suppress(OSError):
                metadata_path.parent.rmdir()
        return removed

    def resolve(self, session_id: str, reference: str) -> StoredFile:
        session_id = _uuid(session_id, "session ID")
        reference = _uuid(reference, "file reference")
        directory = self.root / session_id
        metadata_path = directory / f"{reference}.json"
        path = directory / f"{reference}.csv"
        if not metadata_path.is_file() or not path.is_file():
            raise FileStoreError("File reference is missing or does not belong to this session")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("reference") != reference or path.stat().st_size != metadata.get(
            "byte_size"
        ):
            raise FileStoreError("Stored file metadata mismatch")
        return StoredFile(
            reference=reference,
            original_name=metadata["original_name"],
            path=path,
            kind=metadata["kind"],
            byte_size=metadata["byte_size"],
        )
