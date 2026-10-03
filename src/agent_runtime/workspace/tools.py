"""The workspace tools: ``fs_read``, ``fs_list``, ``fs_write``, ``fs_edit``, ``run_command``,
``http_get``. They run on the system clock (``run_command`` uses threads to read the pipes)."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import threading
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.procs import kill_tree, new_group_kwargs, pid_alive
from agent_runtime.tools import Tool, ToolError
from agent_runtime.workspace.paths import PathRejected, relative, resolve

MAX_READ = 65_536
MAX_STREAM = 65_536
MAX_HTTP = 262_144
DEFAULT_ENV_ALLOW: tuple[str, ...] = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL", "PYTHONUTF8", "PYTHONIOENCODING",
)  # fmt: skip
TRUNCATED = "\n[... output truncated at {n} bytes]"


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadIn(_In):
    path: str
    offset: int = Field(0, ge=0)
    limit: int = Field(MAX_READ, ge=1, le=MAX_READ)


class ListIn(_In):
    path: str = "."


class WriteIn(_In):
    path: str
    content: str
    mode: Literal["create", "overwrite", "append"] = "create"


class EditIn(_In):
    path: str
    old: str = Field(min_length=1)
    new: str


class CommandIn(_In):
    argv: list[str] = Field(min_length=1)
    cwd: str = "."
    timeout_ms: int | None = Field(None, ge=1)


class HttpIn(_In):
    url: str


class _Capped:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.buf = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.cap - len(self.buf)
        if room > 0:
            self.buf += chunk[:room]

    def text(self) -> str:
        s = self.buf.decode("utf-8", errors="replace")
        return s + TRUNCATED.format(n=self.cap) if self.total > self.cap else s


def _pump(stream: Any, sink: _Capped) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            sink.feed(chunk)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


class Workspace:
    """A directory a run is bound to, and the tools that work inside it."""

    def __init__(
        self,
        root: str,
        *,
        clock: Any,
        env_allow: tuple[str, ...] = DEFAULT_ENV_ALLOW,
        timeout_ms: int = 10_000,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.root = os.path.realpath(root)
        self.clock = clock
        self.env_allow = env_allow
        self.timeout_ms = timeout_ms
        self.http_transport = http_transport
        self.pids: list[int] = []

    # -- resolution (also the input of the permission layer) ----------------------------------------
    def resolve(self, path: str) -> str:
        return resolve(self.root, path)

    def resolve_paths(self, args: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {"paths_ok": True}
        for n in names:
            if n not in args:
                continue
            try:
                out[n] = relative(self.root, self.resolve(args[n]))
            except PathRejected as e:
                out["paths_ok"] = False
                out[n] = {"rejected": e.reason}
        return out

    def child_env(self) -> dict[str, str]:
        return {k: os.environ[k] for k in self.env_allow if k in os.environ}

    async def close(self) -> None:
        """Kill every process tree this workspace started that is still alive (by PID)."""
        for pid in self.pids:
            if pid_alive(pid):
                kill_tree(pid)

    # -- tools ------------------------------------------------------------------------------------
    def tools(self) -> list[Tool]:
        def fs(names: tuple[str, ...]):
            return lambda args, ws: self.resolve_paths(args, names)

        def cmd(args: dict[str, Any], ws: Any) -> dict[str, Any]:
            out = self.resolve_paths({"cwd": args.get("cwd", ".")}, ("cwd",))
            out["argv"] = list(args.get("argv") or [])
            return out

        def net(args: dict[str, Any], ws: Any) -> dict[str, Any]:
            try:
                u = httpx.URL(args.get("url", ""))
                port = u.port or (443 if u.scheme == "https" else 80)
                return {"host": f"{u.host}:{port}", "scheme": u.scheme}
            except (httpx.InvalidURL, TypeError):
                return {"host": None, "scheme": None}

        return [
            Tool(
                name="fs_read",
                handler=self._read,
                input_model=ReadIn,
                description="Read a UTF-8 text file of the workspace (offset/limit in bytes).",
                side_effect="read",
                capability="fs_read",
                meta={"resolve": fs(("path",))},
            ),
            Tool(
                name="fs_list",
                handler=self._list,
                input_model=ListIn,
                description="List a directory of the workspace.",
                side_effect="read",
                capability="fs_read",
                meta={"resolve": fs(("path",))},
            ),
            Tool(
                name="fs_write",
                handler=self._write,
                input_model=WriteIn,
                description="Write a file (create, overwrite, or append).",
                side_effect="write",
                capability="fs_write",
                meta={"resolve": fs(("path",))},
            ),
            Tool(
                name="fs_edit",
                handler=self._edit,
                input_model=EditIn,
                description="Replace exactly one occurrence of a string in a file.",
                side_effect="write",
                capability="fs_write",
                meta={"resolve": fs(("path",))},
            ),
            Tool(
                name="run_command",
                handler=self._command,
                input_model=CommandIn,
                description="Run a command given as an argument vector (no shell, no pipes) in the workspace.",
                side_effect="write",
                capability="exec",
                timeout_ms=self.timeout_ms + 5_000,
                meta={"resolve": cmd},
            ),
            Tool(
                name="http_get",
                handler=self._http,
                input_model=HttpIn,
                description="GET a URL (text, redirects not followed).",
                side_effect="read",
                capability="net",
                timeout_ms=self.timeout_ms,
                meta={"resolve": net},
            ),
        ]

    def _path(self, p: str) -> str:
        try:
            return self.resolve(p)
        except PathRejected as e:
            raise ToolError("PERMISSION_DENIED", f"path rejected: {e.reason}") from None

    async def _read(self, a: ReadIn) -> dict[str, Any]:
        path = self._path(a.path)
        if not os.path.isfile(path):
            raise ToolError("NOT_FOUND", f"no file {a.path!r}")
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(a.offset)
            data = f.read(min(a.limit, MAX_READ))
        if b"\x00" in data:
            raise ToolError("FAILED", "binary file refused")
        text = None
        for cut in range(0, 4):
            try:
                text = (data[: len(data) - cut] if cut else data).decode("utf-8")
                data = data[: len(data) - cut] if cut else data
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ToolError("FAILED", "not UTF-8 text: binary file refused")
        return {"content": text, "offset": a.offset, "bytes": len(data), "eof": a.offset + len(data) >= size}

    async def _list(self, a: ListIn) -> dict[str, Any]:
        path = self._path(a.path)
        if not os.path.isdir(path):
            raise ToolError("NOT_FOUND", f"no directory {a.path!r}")
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            entries.append(
                {
                    "name": name,
                    "type": "dir" if os.path.isdir(full) else "file",
                    "size": os.path.getsize(full) if os.path.isfile(full) else None,
                }
            )
        return {"path": relative(self.root, path), "entries": entries}

    async def _write(self, a: WriteIn) -> dict[str, Any]:
        path = self._path(a.path)
        if a.mode == "create" and os.path.exists(path):
            raise ToolError("FAILED", f"{a.path!r} exists (mode create)")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a" if a.mode == "append" else "w", encoding="utf-8", newline="") as f:
            f.write(a.content)
        return {"path": relative(self.root, path), "bytes": len(a.content.encode("utf-8")), "mode": a.mode}

    async def _edit(self, a: EditIn) -> dict[str, Any]:
        path = self._path(a.path)
        if not os.path.isfile(path):
            raise ToolError("NOT_FOUND", f"no file {a.path!r}")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        n = text.count(a.old)
        if n != 1:
            raise ToolError("FAILED", f"the string occurs {n} times (exactly one is required)")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(a.old, a.new, 1))
        return {"path": relative(self.root, path), "replaced": 1}

    async def _command(self, a: CommandIn) -> dict[str, Any]:
        cwd = self._path(a.cwd)
        env = self.child_env()
        exe = shutil.which(a.argv[0], path=env.get("PATH")) or a.argv[0]
        timeout = a.timeout_ms or self.timeout_ms
        try:
            proc = subprocess.Popen(
                [exe, *a.argv[1:]],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                **new_group_kwargs(),
            )
        except OSError as e:
            raise ToolError("FAILED", f"cannot start {a.argv[0]!r}: {e.strerror}") from None
        self.pids.append(proc.pid)
        out, err = _Capped(MAX_STREAM), _Capped(MAX_STREAM)
        threads = [
            threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
        ]
        for t in threads:
            t.start()
        timed_out = False
        try:
            code = await self.clock.wait_for(asyncio.to_thread(proc.wait), timeout)
        except TimeoutError:
            timed_out = True
            kill_tree(proc.pid)
            code = None
            await asyncio.to_thread(proc.wait, 10)
        except asyncio.CancelledError:
            kill_tree(proc.pid)
            raise
        for t in threads:
            await asyncio.to_thread(t.join, 5)
        if timed_out:
            raise ToolError(
                "TIMEOUT", f"killed after {timeout} ms", {"stdout": out.text()[:2000], "pid": proc.pid}
            )
        return {"exit_code": code, "stdout": out.text(), "stderr": err.text(), "pid": proc.pid}

    async def _http(self, a: HttpIn) -> dict[str, Any]:
        try:
            u = httpx.URL(a.url)
        except httpx.InvalidURL:
            raise ToolError("INVALID_ARGUMENTS", "invalid URL") from None
        if u.scheme not in ("http", "https"):
            raise ToolError("INVALID_ARGUMENTS", "only http and https")
        local = u.host in ("127.0.0.1", "localhost", "::1")
        try:
            async with httpx.AsyncClient(
                transport=self.http_transport,
                trust_env=not local and self.http_transport is None,
                follow_redirects=False,
                timeout=self.timeout_ms / 1000.0,
            ) as c:
                async with c.stream("GET", a.url) as r:
                    buf = _Capped(MAX_HTTP)
                    async for chunk in r.aiter_bytes():
                        buf.feed(chunk)
                        if buf.total > MAX_HTTP:
                            break
                    return {
                        "status": r.status_code,
                        "content": buf.text(),
                        "location": r.headers.get("location"),
                    }
        except httpx.TimeoutException:
            raise ToolError("TIMEOUT", "HTTP request timed out") from None
        except httpx.HTTPError as e:
            raise ToolError("UNAVAILABLE", f"HTTP error: {type(e).__name__}") from None
