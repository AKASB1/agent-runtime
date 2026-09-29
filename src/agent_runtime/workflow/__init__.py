"""Sequential workflow primitive."""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

@dataclass
class Step:
    name: str
    run: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

async def execute(steps: list[Step], state: dict[str, Any]) -> dict[str, Any]:
    current = dict(state)
    for step in steps:
        current.update(await step.run(current))
    return current
