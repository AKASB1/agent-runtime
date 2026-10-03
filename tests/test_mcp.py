"""Check 6: typed registry and the MCP round trip over stdio (default event loop, real subprocess)."""

import asyncio

import pytest
from pydantic import BaseModel

from agent_runtime.clock import SystemClock
from agent_runtime.mcp.client import McpServer, python_server
from agent_runtime.mcp.twin import native_lookup_tool
from agent_runtime.procs import pid_alive
from agent_runtime.runtime.runner import RunSpec, execute
from agent_runtime.store.memory import MemoryStore
from agent_runtime.telemetry.trace import validate_trace
from agent_runtime.tools import Tool, ToolError, ToolRegistry
from helpers import ScriptedEnv, call, u


def test_typed_registry_schema_from_model_and_validation_both_ways():
    class In(BaseModel):
        a: int
        b: str = "x"

    class Out(BaseModel):
        s: str

    async def h(v):
        return {"s": f"{v.a}{v.b}"}

    async def bad(v):
        return {"nope": 1}

    reg = ToolRegistry(
        [
            Tool(name="t", handler=h, input_model=In, output_model=Out),
            Tool(name="bad", handler=bad, input_model=In, output_model=Out),
        ]
    )
    schema = reg.get("t").spec().parameters
    assert schema["properties"]["a"]["type"] == "integer" and schema["required"] == ["a"]
    assert asyncio.run(reg.invoke("t", {"a": 1})) == {"s": "1x"}
    with pytest.raises(ToolError) as ei:
        asyncio.run(reg.invoke("t", {"a": "not int"}))
    assert ei.value.code == "INVALID_ARGUMENTS"
    with pytest.raises(ToolError) as ei:
        asyncio.run(reg.invoke("bad", {"a": 1}))
    assert ei.value.code == "INVALID_OUTPUT"
    with pytest.raises(ValueError):
        Tool(name="Bad-Name", handler=h, input_model=In)
    with pytest.raises(ValueError):
        reg.register(Tool(name="t", handler=h, input_model=In))


def test_mcp_round_trip_list_call_invalid_args_error_and_clean_shutdown_by_pid():
    async def go():
        server = McpServer(
            python_server("test", "agent_runtime.mcp.testserver", capability="none", side_effect="read")
        )
        await server.start()
        try:
            tools = {t.name: t for t in server.tools()}
            assert set(tools) == {"test.lookup", "test.add", "test.fail", "test.slow", "test.crash"}
            assert tools["test.add"].schema()["required"] == ["a", "b"]
            assert await server.call("add", {"a": 2, "b": 40}) == {"sum": 42}
            # arguments are validated against the declared schema before calling
            _, err = tools["test.add"].validate_input({"a": "x", "b": 1})
            assert err is not None
            with pytest.raises(ToolError) as ei:
                await server.call("fail", {"message": "boom"})
            assert (ei.value.code, ei.value.message) == ("FAILED", "boom")
            pid = server.pid
            assert pid_alive(pid)
        finally:
            await server.close()
        assert not pid_alive(pid)

    asyncio.run(go())


def test_mcp_server_killed_in_the_middle_of_a_call_is_a_tool_error():
    async def go():
        server = McpServer(python_server("test", "agent_runtime.mcp.testserver", capability="none"))
        await server.start()
        pid = server.pid
        try:
            with pytest.raises(ToolError) as ei:
                await asyncio.wait_for(server.call("crash", {}), 30)
            assert ei.value.code in ("UNAVAILABLE", "FAILED")
            with pytest.raises(ToolError) as ei:
                await server.call("add", {"a": 1, "b": 2})
            assert ei.value.code in ("UNAVAILABLE", "FAILED")
        finally:
            await server.close()
        assert not pid_alive(pid)

    asyncio.run(go())


def mcp_workflow(use_mcp: bool):
    """The same react workflow, with the native lookup tool or its MCP twin, on the system clock."""
    name = "test.lookup" if use_mcp else "lookup"
    script = [
        {"tool_calls": [call("c1", name, key="beta")], "usage": u()},
        {"tool_calls": [call("c2", name, key="zzz")], "usage": u()},
        {"text": "done", "usage": u()},
    ]

    async def go():
        clock = SystemClock()
        senv = ScriptedEnv({"p": script}, tools=[])
        env = senv(None, clock)
        server = None
        if use_mcp:
            server = McpServer(
                python_server("test", "agent_runtime.mcp.testserver", capability="none", side_effect="read")
            )
            await server.start()
            for t in server.tools():
                env.tools.register(t)
        else:
            env.tools.register(native_lookup_tool())
        store = MemoryStore()
        rs = RunSpec(
            run_id="r-1", workflow="react", input={"task": "t"}, default_target="p/m", clock="system"
        )
        try:
            res = await execute(rs, env, clock, store, store.create_session())
        finally:
            if server:
                await server.close()
        return res, server.pid if server else None

    res, pid = asyncio.run(go())
    validate_trace(res.trace.dicts())
    assert pid is None or not pid_alive(pid)
    return [t["result"] for t in res.trace.dicts() if t["kind"] == "tool"], res


def test_native_tool_and_its_mcp_twin_give_the_same_results():
    native, rn = mcp_workflow(False)
    remote, rm = mcp_workflow(True)
    assert native == remote
    assert native[0] == {"ok": True, "result": {"key": "beta", "value": 22}}
    assert native[1]["error"]["code"] == "NOT_FOUND"
    assert (rn.status, rn.text) == (rm.status, rm.text) == ("succeeded", "done")
