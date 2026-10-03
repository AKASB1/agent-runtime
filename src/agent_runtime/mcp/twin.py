"""The ``lookup`` tool shared by a native registration and the MCP test server (equivalence test)."""

from __future__ import annotations

from pydantic import BaseModel

from agent_runtime.tools import Tool, ToolError

TABLE = {"alpha": 1, "beta": 22, "gamma": 333}


def lookup_value(key: str) -> dict:
    if key not in TABLE:
        raise ValueError(f"NOT_FOUND: no key {key!r}")
    return {"key": key, "value": TABLE[key]}


class LookupIn(BaseModel):
    key: str


class LookupOut(BaseModel):
    key: str
    value: int


async def _native(args: LookupIn) -> LookupOut:
    try:
        return LookupOut(**lookup_value(args.key))
    except ValueError as e:
        code, _, msg = str(e).partition(": ")
        raise ToolError(code, msg) from None


def native_lookup_tool(name: str = "lookup") -> Tool:
    return Tool(
        name=name,
        handler=_native,
        input_model=LookupIn,
        output_model=LookupOut,
        description="Look up the value stored under a key.",
        side_effect="read",
    )
