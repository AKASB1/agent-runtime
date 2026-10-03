"""The ``Store`` interface. Identifiers come from store counters taken inside the transaction that
creates the object: sessions ``s-<n>``, runs ``r-<n>``, messages ``m-<n>``, compactions ``k-<n>``;
every event of a session gets ``seq`` 1, 2, ... in the transaction that appends it."""

from __future__ import annotations

from typing import Any, Protocol


class VersionConflict(Exception):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"state version is {actual}, not {expected}")
        self.expected = expected
        self.actual = actual


class NotFound(KeyError):
    pass


class Store(Protocol):
    def create_session(self, meta: dict[str, Any] | None = None) -> str: ...

    def session_exists(self, session_id: str) -> bool: ...

    def append_event(self, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """Append ``event`` (``type`` message|compaction); returns it with ``seq`` and ``id``."""
        ...

    def events(self, session_id: str, *, upto_seq: int | None = None) -> list[dict[str, Any]]: ...

    def get_state(self, session_id: str) -> tuple[dict[str, Any], int]: ...

    def put_state(self, session_id: str, state: dict[str, Any], expected_version: int) -> int: ...

    def fork(self, session_id: str, at_seq: int) -> str: ...

    def create_run(self, session_id: str | None, record: dict[str, Any]) -> str: ...

    def save_run(self, run_id: str, record: dict[str, Any]) -> None: ...

    def get_run(self, run_id: str) -> dict[str, Any]: ...

    def list_runs(self, session_id: str | None = None) -> list[dict[str, Any]]: ...

    def put_trace(self, run_id: str, jsonl: str) -> None: ...

    def get_trace(self, run_id: str) -> str: ...

    def close(self) -> None: ...


def event_kind_id(kind: str) -> str:
    return {"message": "m", "compaction": "k"}[kind]
