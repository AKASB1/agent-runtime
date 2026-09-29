"""Retrieval adapter boundary."""
from typing import Protocol

class Retriever(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[str]: ...
