"""In-memory ``Store`` (simulations and tests)."""

from __future__ import annotations

import copy
import threading
from typing import Any

from agent_runtime.store.base import NotFound, VersionConflict, event_kind_id


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._traces: dict[str, str] = {}

    def _next(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}-{n}"

    def create_session(self, meta: dict[str, Any] | None = None) -> str:
        with self._lock:
            sid = self._next("s")
            self._sessions[sid] = {"meta": dict(meta or {}), "events": [], "state": {}, "version": 0}
            return sid

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def _session(self, session_id: str) -> dict[str, Any]:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise NotFound(session_id) from None

    def append_event(self, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            s = self._session(session_id)
            ev = dict(event)
            ev["seq"] = len(s["events"]) + 1
            ev["id"] = self._next(event_kind_id(ev["type"]))
            s["events"].append(ev)
            return ev

    def events(self, session_id: str, *, upto_seq: int | None = None) -> list[dict[str, Any]]:
        evs = self._session(session_id)["events"]
        return list(evs if upto_seq is None else evs[:upto_seq])

    def get_state(self, session_id: str) -> tuple[dict[str, Any], int]:
        s = self._session(session_id)
        return copy.deepcopy(s["state"]), s["version"]

    def put_state(self, session_id: str, state: dict[str, Any], expected_version: int) -> int:
        with self._lock:
            s = self._session(session_id)
            if s["version"] != expected_version:
                raise VersionConflict(expected_version, s["version"])
            s["state"] = copy.deepcopy(state)
            s["version"] += 1
            return s["version"]

    def fork(self, session_id: str, at_seq: int) -> str:
        with self._lock:
            src = self._session(session_id)
            sid = self._next("s")
            evs = [dict(e) for e in src["events"][:at_seq]]
            self._sessions[sid] = {
                "meta": {"forked_from": session_id, "at_seq": at_seq},
                "events": evs,
                "state": copy.deepcopy(src["state"]),
                "version": 0,
            }
            return sid

    def create_run(self, session_id: str | None, record: dict[str, Any]) -> str:
        with self._lock:
            rid = self._next("r")
            self._runs[rid] = {**record, "id": rid, "session_id": session_id}
            return rid

    def save_run(self, run_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            if run_id not in self._runs:
                raise NotFound(run_id)
            self._runs[run_id] = {**self._runs[run_id], **record, "id": run_id}

    def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            return dict(self._runs[run_id])
        except KeyError:
            raise NotFound(run_id) from None

    def list_runs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return [
            dict(r) for r in self._runs.values() if session_id is None or r.get("session_id") == session_id
        ]

    def put_trace(self, run_id: str, jsonl: str) -> None:
        self._traces[run_id] = jsonl

    def get_trace(self, run_id: str) -> str:
        try:
            return self._traces[run_id]
        except KeyError:
            raise NotFound(run_id) from None

    def close(self) -> None:
        pass
