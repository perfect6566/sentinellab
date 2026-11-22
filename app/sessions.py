"""Session-isolated, bounded Agent result storage."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any


class SessionResultError(ValueError):
    """Raised when a result is missing, expired, or owned by another session."""


@dataclass(frozen=True)
class StoredResult:
    result_id: str
    session_id: str
    tool: str
    data: dict[str, Any]
    display_data: dict[str, Any]
    created_at: float


class SessionResultStore:
    def __init__(self, *, ttl_seconds: float = 3600.0, max_results_per_session: int = 20) -> None:
        if ttl_seconds <= 0 or max_results_per_session <= 0:
            raise ValueError("Result-store limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_results_per_session = max_results_per_session
        self._results: dict[str, StoredResult] = {}

    @staticmethod
    def _bounded_data(tool: str, data: dict[str, Any]) -> dict[str, Any]:
        if tool == "predict_batch":
            return {
                key: data[key]
                for key in ("summary", "result_file_reference", "model_version")
                if key in data
            }
        return dict(data)

    def purge_expired(self, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        expired = [
            result_id
            for result_id, result in self._results.items()
            if current - result.created_at >= self.ttl_seconds
        ]
        for result_id in expired:
            del self._results[result_id]

    def put(self, session_id: str, tool: str, data: dict[str, Any]) -> StoredResult:
        self.purge_expired()
        owned = sorted(
            (result for result in self._results.values() if result.session_id == session_id),
            key=lambda result: result.created_at,
        )
        while len(owned) >= self.max_results_per_session:
            oldest = owned.pop(0)
            del self._results[oldest.result_id]
        stored = StoredResult(
            result_id=str(uuid.uuid4()),
            session_id=session_id,
            tool=tool,
            data=self._bounded_data(tool, data),
            display_data=dict(data),
            created_at=time.time(),
        )
        self._results[stored.result_id] = stored
        return stored

    def get(self, session_id: str, result_id: str) -> StoredResult:
        self.purge_expired()
        result = self._results.get(result_id)
        if result is None or result.session_id != session_id:
            raise SessionResultError("Result is missing, expired, or unavailable to this session")
        return result
