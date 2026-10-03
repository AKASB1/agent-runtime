"""A small MCP server for tests (a test double): ``lookup`` (the twin of a native tool), ``add``,
``fail``, ``slow``, and ``crash``. Run: ``python -m agent_runtime.mcp.testserver``."""

from __future__ import annotations

import asyncio
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agent_runtime.mcp.twin import lookup_value

server = MCPServer("test", log_level="CRITICAL")


@server.tool(description="Look up the value stored under a key.")
def lookup(key: str) -> dict:
    try:
        return lookup_value(key)
    except ValueError as e:
        raise ToolError(str(e)) from None


@server.tool(description="Add two integers.")
def add(a: int, b: int) -> dict:
    return {"sum": a + b}


@server.tool(description="Always fails with the given message.")
def fail(message: str) -> dict:
    raise ToolError(f"FAILED: {message}")


@server.tool(description="Wait for ms milliseconds of real time, then answer.")
async def slow(ms: int) -> dict:
    await asyncio.sleep(ms / 1000.0)  # a test double on the system clock
    return {"slept_ms": ms}


@server.tool(description="Exit the server process in the middle of the call.")
def crash() -> dict:
    os._exit(3)


if __name__ == "__main__":
    server.run("stdio")
