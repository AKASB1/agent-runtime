"""Workflows of the runtime (registered by name) and the scaffold's sequential ``execute`` helper.

``chat``, ``react``, ``react_parallel``, ``plan_execute``, ``plan_replan`` live in
``workflow.workflows``; ``agent`` and ``optimize`` register themselves from their modules.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    name: str
    run: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def execute(steps: list[Step], state: dict[str, Any]) -> dict[str, Any]:
    """Run steps one after another over a state dictionary (the scaffold's helper, kept)."""
    current = dict(state)
    for step in steps:
        current.update(await step.run(current))
    return current
