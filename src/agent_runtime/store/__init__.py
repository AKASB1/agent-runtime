"""Sessions (append-only event log + versioned state), runs, and traces behind a small ``Store``
interface with two implementations: in memory (simulations) and SQLite (service, persistence).
The interface has no update and no delete for events."""

from agent_runtime.store.base import NotFound, Store, VersionConflict
from agent_runtime.store.memory import MemoryStore
from agent_runtime.store.sqlite import SqliteStore

__all__ = ["MemoryStore", "NotFound", "SqliteStore", "Store", "VersionConflict"]
