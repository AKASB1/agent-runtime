"""Named async tools with explicit side-effect metadata."""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

@dataclass(frozen=True)
class Tool:
    name: str
    call: Callable[[dict[str, Any]], Awaitable[Any]]
    side_effects: bool = False

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError("duplicate tool")
        self._tools[tool.name] = tool

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._tools[name].call(arguments)
