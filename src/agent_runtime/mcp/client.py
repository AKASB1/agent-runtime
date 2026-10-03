"""MCP client over stdio, built on the ``mcp`` SDK (pinned 2.2.0; protocol revision 2026-07-28 with
fallback to the handshake revisions).

The SDK's ``Client`` speaks the protocol; this module owns the lifecycle that TASK requires: the
server's PID is recorded when the SDK spawns it, the client lives in one owner task (the SDK's
anyio scopes must be entered and left in the same task), the shutdown closes stdin and waits up to
5 seconds, and afterwards the process tree is killed by PID if anything survived. Tool errors,
protocol errors, timeouts, and a server that exits become ``ToolError`` (-> ``tool_error``), and
timeouts are applied by the run engine (-> ``tool_timeout``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextvars import ContextVar
from typing import Any

import mcp.client.stdio as _sdk_stdio
from mcp import Client, StdioServerParameters
from pydantic import BaseModel, Field

from agent_runtime.procs import kill_tree, pid_alive
from agent_runtime.tools import TOOL_ERROR_CODES, Capability, SideEffect, Tool, ToolError

SHUTDOWN_GRACE_S = 5.0
# the SDK prefixes tool errors with "Error executing tool <name>: "; our servers put a code first
_CODED = re.compile(r"^(?:Error executing tool [^:]+: )?([A-Z_]+): (.*)$", re.DOTALL)
_PID_SINK: ContextVar[list[int] | None] = ContextVar("agent_runtime_mcp_pid_sink", default=None)
_orig_spawn = _sdk_stdio._create_platform_compatible_process


async def _spawn_recording_pid(*args: Any, **kwargs: Any) -> Any:
    process = await _orig_spawn(*args, **kwargs)
    sink = _PID_SINK.get()
    if sink is not None:
        sink.append(process.pid)
    return process


# Record the PID of every server the SDK spawns for us, and give it 5 s to exit after stdin closes.
_sdk_stdio._create_platform_compatible_process = _spawn_recording_pid
_sdk_stdio.PROCESS_TERMINATION_TIMEOUT = SHUTDOWN_GRACE_S


class McpServerConfig(BaseModel):
    name: str
    command: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    #: capability of every tool of this server (input of the permission layer); unknown effects -> exec
    capability: Capability = "exec"
    #: side effect of every tool of this server; unknown -> write (never retried)
    side_effect: SideEffect = "write"
    timeout_ms: int = 10_000
    start_timeout_ms: int = 60_000


def python_server(name: str, module: str, *args: str, **kw: Any) -> McpServerConfig:
    """Config for a server that is a module of this package, run with the current interpreter.

    The SDK passes only a short list of environment variables to the server, so the location of
    this package is handed over explicitly (computed at run time, never a committed path)."""
    import agent_runtime

    src = os.path.dirname(os.path.dirname(os.path.abspath(agent_runtime.__file__)))
    env = {"PYTHONPATH": src, "PYTHONUTF8": "1", **kw.pop("env", {})}
    return McpServerConfig(name=name, command=[sys.executable, "-m", module, *args], env=env, **kw)


class McpServer:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._pids: list[int] = []
        self._client: Client | None = None
        self._owner: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._listing: list[Any] = []
        self.calls = 0

    @property
    def pid(self) -> int | None:
        return self._pids[0] if self._pids else None

    async def __aenter__(self) -> McpServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def start(self) -> None:
        self._owner = asyncio.ensure_future(self._own())
        try:
            await asyncio.wait_for(self._ready.wait(), self.config.start_timeout_ms / 1000.0)
        except TimeoutError:
            await self.close()
            raise RuntimeError(f"MCP server {self.config.name!r} did not start in time") from None
        if self._error is not None:
            err = self._error
            await self.close()
            raise RuntimeError(f"MCP server {self.config.name!r} failed to start: {err!r}") from None

    async def _own(self) -> None:
        token = _PID_SINK.set(self._pids)
        try:
            params = StdioServerParameters(
                command=self.config.command[0],
                args=self.config.command[1:],
                env=dict(self.config.env) or None,
                cwd=self.config.cwd,
            )
            async with Client(params, mode="auto") as client:
                self._client = client
                listing = await client.list_tools()
                self._listing = list(listing.tools)
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surfaced through start() or calls
            self._error = e
        finally:
            self._client = None
            _PID_SINK.reset(token)
            self._ready.set()

    def tools(self) -> list[Tool]:
        out = []
        for t in self._listing:
            schema = dict(t.input_schema or {"type": "object"})
            out.append(
                Tool(
                    name=f"{self.config.name}.{t.name}",
                    handler=self._handler_for(t.name),
                    input_schema=schema,
                    description=t.description or "",
                    timeout_ms=self.config.timeout_ms,
                    side_effect=self.config.side_effect,
                    capability=self.config.capability,
                    meta={"mcp_server": self.config.name},
                )
            )
        return out

    def _handler_for(self, tool_name: str):
        async def handler(args: dict[str, Any]) -> Any:
            return await self.call(tool_name, args)

        return handler

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        client = self._client
        if client is None or (self._owner is not None and self._owner.done()):
            raise ToolError("UNAVAILABLE", f"MCP server {self.config.name!r} is not running")
        self.calls += 1
        try:
            res = await client.call_tool(tool_name, args)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # protocol errors, closed streams, a server that exited
            if self.pid is not None and not pid_alive(self.pid):
                raise ToolError("UNAVAILABLE", f"MCP server {self.config.name!r} exited") from None
            raise ToolError("FAILED", f"MCP error: {type(e).__name__}: {str(e)[:200]}") from None
        text = "".join(getattr(c, "text", "") for c in res.content or [])
        if res.is_error:
            m = _CODED.match(text)
            if m and m.group(1) in TOOL_ERROR_CODES:
                raise ToolError(m.group(1), m.group(2)[:500])
            raise ToolError("FAILED", text[:500] or "MCP tool reported an error")
        if res.structured_content is not None:
            sc = res.structured_content
            if isinstance(sc, dict) and set(sc) == {"result"}:
                return sc["result"]
            return sc
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"text": text}

    async def close(self) -> None:
        self._stop.set()
        if self._owner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._owner), SHUTDOWN_GRACE_S + 10)
            except (TimeoutError, asyncio.CancelledError):
                self._owner.cancel()
            except BaseException:  # noqa: BLE001
                pass
        # whatever the SDK did, nothing we started may survive: kill the tree by PID
        for pid in self._pids:
            if pid_alive(pid):
                kill_tree(pid)
