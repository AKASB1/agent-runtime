"""In-memory session store; persistence is planned."""
class MemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, list[str]] = {}

    def append(self, session_id: str, text: str) -> None:
        self._items.setdefault(session_id, []).append(text)

    def read(self, session_id: str) -> list[str]:
        return list(self._items.get(session_id, []))
