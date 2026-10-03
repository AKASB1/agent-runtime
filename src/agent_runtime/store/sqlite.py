"""SQLite ``Store``: WAL, a busy timeout, ``BEGIN IMMEDIATE`` write transactions, every connection
closed explicitly. Events are only ever inserted."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from agent_runtime.store.base import NotFound, VersionConflict, event_kind_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, meta TEXT NOT NULL, state TEXT NOT NULL, version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  session_id TEXT NOT NULL, seq INTEGER NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
  PRIMARY KEY (session_id, seq));
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, session_id TEXT, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces (run_id TEXT PRIMARY KEY, jsonl TEXT NOT NULL);
"""


class SqliteStore:
    def __init__(self, path: str, *, busy_timeout_ms: int = 10_000) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            path, timeout=busy_timeout_ms / 1000.0, isolation_level=None, check_same_thread=False
        )
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._tx() as c:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    c.execute(statement)

    # -- transactions --------------------------------------------------------------------------
    class _Tx:
        def __init__(self, store: SqliteStore) -> None:
            self.s = store

        def __enter__(self) -> sqlite3.Connection:
            self.s._lock.acquire()
            self.s._conn.execute("BEGIN IMMEDIATE")
            return self.s._conn

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            try:
                self.s._conn.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self.s._lock.release()

    def _tx(self) -> SqliteStore._Tx:
        return SqliteStore._Tx(self)

    def _next(self, c: sqlite3.Connection, prefix: str) -> str:
        row = c.execute("SELECT value FROM counters WHERE name=?", (prefix,)).fetchone()
        n = (row[0] if row else 0) + 1
        c.execute(
            "INSERT INTO counters(name, value) VALUES(?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
            (prefix, n),
        )
        return f"{prefix}-{n}"

    # -- sessions --------------------------------------------------------------------------------
    def create_session(self, meta: dict[str, Any] | None = None) -> str:
        with self._tx() as c:
            sid = self._next(c, "s")
            c.execute(
                "INSERT INTO sessions(id, meta, state, version) VALUES(?, ?, '{}', 0)",
                (sid, json.dumps(meta or {})),
            )
            return sid

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return (
                self._conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is not None
            )

    def append_event(self, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        with self._tx() as c:
            if c.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                raise NotFound(session_id)
            row = c.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id=?", (session_id,)
            ).fetchone()
            ev = dict(event)
            ev["seq"] = row[0] + 1
            ev["id"] = self._next(c, event_kind_id(ev["type"]))
            c.execute(
                "INSERT INTO events(session_id, seq, id, body) VALUES(?, ?, ?, ?)",
                (session_id, ev["seq"], ev["id"], json.dumps(ev, sort_keys=True)),
            )
            return ev

    def events(self, session_id: str, *, upto_seq: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if not self.session_exists(session_id):
                raise NotFound(session_id)
            q = (
                "SELECT body FROM events WHERE session_id=?"
                + (" AND seq<=?" if upto_seq is not None else "")
                + " ORDER BY seq"
            )
            args = (session_id,) if upto_seq is None else (session_id, upto_seq)
            return [json.loads(r[0]) for r in self._conn.execute(q, args).fetchall()]

    def get_state(self, session_id: str) -> tuple[dict[str, Any], int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT state, version FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise NotFound(session_id)
        return json.loads(row[0]), int(row[1])

    def put_state(self, session_id: str, state: dict[str, Any], expected_version: int) -> int:
        with self._tx() as c:
            row = c.execute("SELECT version FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise NotFound(session_id)
            if int(row[0]) != expected_version:
                raise VersionConflict(expected_version, int(row[0]))
            c.execute(
                "UPDATE sessions SET state=?, version=? WHERE id=?",
                (json.dumps(state), expected_version + 1, session_id),
            )
            return expected_version + 1

    def fork(self, session_id: str, at_seq: int) -> str:
        with self._tx() as c:
            row = c.execute("SELECT state FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise NotFound(session_id)
            sid = self._next(c, "s")
            c.execute(
                "INSERT INTO sessions(id, meta, state, version) VALUES(?, ?, ?, 0)",
                (sid, json.dumps({"forked_from": session_id, "at_seq": at_seq}), row[0]),
            )
            c.execute(
                "INSERT INTO events(session_id, seq, id, body) SELECT ?, seq, id, body FROM events WHERE session_id=? AND seq<=?",
                (sid, session_id, at_seq),
            )
            return sid

    # -- runs and traces ---------------------------------------------------------------------------
    def create_run(self, session_id: str | None, record: dict[str, Any]) -> str:
        with self._tx() as c:
            rid = self._next(c, "r")
            rec = {**record, "id": rid, "session_id": session_id}
            c.execute(
                "INSERT INTO runs(id, session_id, record) VALUES(?, ?, ?)", (rid, session_id, json.dumps(rec))
            )
            return rid

    def save_run(self, run_id: str, record: dict[str, Any]) -> None:
        with self._tx() as c:
            row = c.execute("SELECT record FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise NotFound(run_id)
            rec = {**json.loads(row[0]), **record, "id": run_id}
            c.execute("UPDATE runs SET record=? WHERE id=?", (json.dumps(rec), run_id))

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT record FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFound(run_id)
        return json.loads(row[0])

    def list_runs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if session_id is None:
                rows = self._conn.execute("SELECT record FROM runs ORDER BY rowid").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT record FROM runs WHERE session_id=? ORDER BY rowid", (session_id,)
                ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def put_trace(self, run_id: str, jsonl: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO traces(run_id, jsonl) VALUES(?, ?) ON CONFLICT(run_id) DO UPDATE SET jsonl=excluded.jsonl",
                (run_id, jsonl),
            )

    def get_trace(self, run_id: str) -> str:
        with self._lock:
            row = self._conn.execute("SELECT jsonl FROM traces WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFound(run_id)
        return row[0]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]
