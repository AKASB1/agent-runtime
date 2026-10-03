"""Typed tool registry.

A tool has a name (``[a-z][a-z0-9_]{0,47}``, MCP tools ``<server>.<tool>``), a description, an input
and an output model (Pydantic; MCP tools carry the JSON Schema their server declares instead), a
timeout, a side effect (``none``, ``read``, ``write``) and a capability (``none``, ``fs_read``,
``fs_write``, ``exec``, ``net``), which is the input of the permission layer. Arguments are
validated before the call and the output after it; a failure is a ``tool_error`` result, never an
exception that ends a run.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from agent_runtime.canon import canonical
from agent_runtime.models.structured import validate_args
from agent_runtime.models.types import ToolSpec

SideEffect = Literal["none", "read", "write"]
Capability = Literal["none", "fs_read", "fs_write", "exec", "net"]
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
QUALIFIED_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}\.[a-z][a-z0-9_]{0,47}$")

#: tool error codes that appear in tool results
TOOL_ERROR_CODES: tuple[str, ...] = (
    "INVALID_ARGUMENTS",
    "UNKNOWN_TOOL",
    "INVALID_OUTPUT",
    "NOT_FOUND",
    "UNAVAILABLE",
    "TIMEOUT",
    "PERMISSION_DENIED",
    "FAILED",
    "SUBAGENT_FAILED",
)


class ToolError(Exception):
    """Raised by a tool handler to return an error result (``code`` from ``TOOL_ERROR_CODES``)."""

    def __init__(self, code: str, message: str = "", details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class Tool:
    name: str
    handler: Callable[[Any], Awaitable[Any]]
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    description: str = ""
    timeout_ms: int = 10_000
    side_effect: SideEffect = "none"
    capability: Capability = "none"
    #: JSON Schema of the arguments when there is no input model (MCP tools)
    input_schema: dict[str, Any] | None = None
    #: argument names that are workspace paths (resolved before the permission decision)
    path_args: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (NAME_RE.match(self.name) or QUALIFIED_RE.match(self.name)):
            raise ValueError(f"invalid tool name {self.name!r}")
        if self.input_model is None and self.input_schema is None:
            raise ValueError("a tool needs an input model or an input schema")

    def schema(self) -> dict[str, Any]:
        if self.input_model is not None:
            return self.input_model.model_json_schema()
        return dict(self.input_schema or {})

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.schema())

    def validate_input(self, args: dict[str, Any]) -> tuple[Any, str | None]:
        """Return (validated value, None) or (None, error message)."""
        if "_invalid_json" in args:
            return None, "arguments are not valid JSON"
        if self.input_model is not None:
            try:
                return self.input_model.model_validate(args), None
            except ValidationError as e:
                err = e.errors()[0]
                where = "/".join(str(p) for p in err.get("loc", ())) or "(root)"
                return None, f"{where}: {err.get('msg', 'invalid')}"
        msg = validate_args(args, self.input_schema or {})
        return (args, None) if msg is None else (None, msg)

    def dump_output(self, out: Any) -> tuple[Any, str | None]:
        """Validate the handler's output; return (JSON-able value, None) or (None, error)."""
        try:
            if self.output_model is not None:
                if isinstance(out, BaseModel):
                    out = out.model_dump(mode="json")
                return self.output_model.model_validate(out).model_dump(mode="json"), None
            if isinstance(out, BaseModel):
                out = out.model_dump(mode="json")
            canonical(out)
            return out, None
        except (ValidationError, TypeError, ValueError) as e:
            return None, f"output failed validation: {str(e).splitlines()[0]}"


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    result: Any = None
    error_code: str | None = None
    error_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "result": self.result}
        err: dict[str, Any] = {"code": self.error_code, "message": self.error_message}
        if self.details:
            err["details"] = self.details
        return {"ok": False, "error": err}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolResult:
        if d.get("ok"):
            return cls(ok=True, result=d.get("result"))
        e = d.get("error") or {}
        return cls(
            ok=False,
            error_code=e.get("code"),
            error_message=e.get("message", ""),
            details=e.get("details") or {},
        )

    def content(self) -> str:
        """The text of the tool message that goes back to the model."""
        return canonical(self.to_dict())

    @classmethod
    def error(cls, code: str, message: str = "", **details: Any) -> ToolResult:
        return cls(ok=False, error_code=code, error_message=message, details=dict(details))


class ToolRegistry:
    """Tools by name, in registration order."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self, allow: list[str] | None = None) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values() if allow is None or t.name in allow]

    def subset(self, names: list[str]) -> ToolRegistry:
        return ToolRegistry([t for t in self._tools.values() if t.name in names])

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        """Validate, call, and validate the output (no permission layer, no trace: a helper)."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError("UNKNOWN_TOOL", f"no tool named {name!r}")
        value, err = tool.validate_input(arguments)
        if err:
            raise ToolError("INVALID_ARGUMENTS", err)
        out, err = tool.dump_output(await tool.handler(value))
        if err:
            raise ToolError("INVALID_OUTPUT", err)
        return out
