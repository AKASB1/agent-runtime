"""The run engine on the scripted model: trace, replay, the failure table (check 4), budgets and
cancellation (check 8), structured output (check 5), accounting per run (check 3)."""

import asyncio

import pytest

from agent_runtime.clock import run_virtual
from agent_runtime.models.types import ProviderError
from agent_runtime.runtime.engine import Budget, RunConfig
from agent_runtime.runtime.runner import RunSpec, execute, replay_spans
from agent_runtime.store.memory import MemoryStore
from agent_runtime.tools import Tool, ToolError
from agent_runtime.workflow import workflows
from helpers import ScriptedEnv, call, kinds, run_scripted, spec, u
from test_models import MALFORMED_FIXTURES, SCHEMA


def react_script():
    return [
        {"tool_calls": [call("c1", key="beta")], "usage": u(100, 10)},
        {"tool_calls": [call("c2", key="nope")], "usage": u(150, 10)},
        {"text": "beta is 22", "usage": u(200, 5)},
    ]


def test_react_run_trace_accounting_and_replay():
    res = run_scripted(ScriptedEnv({"p": react_script()}))
    assert (res.status, res.text) == ("succeeded", "beta is 22")
    # 100+150+200 in at 1 cu/M + 25 out at 4 cu/M = 450 + 100 ucu
    assert res.meter.cost_ucu == 550
    assert (res.meter.model_calls, res.meter.tool_calls, res.meter.provider_calls) == (3, 2, 3)
    assert res.latency_ms == 300
    tools = kinds(res, "tool")
    assert [t["status"] for t in tools] == ["ok", "error"]
    assert tools[1]["result"]["error"]["code"] == "NOT_FOUND"
    run_span = res.trace.dicts()[0]
    assert run_span["attrs"]["agent_runtime.cost_ucu"] == 550


def test_replay_detects_a_changed_code_path_at_the_first_differing_span(monkeypatch):
    senv = ScriptedEnv({"p": react_script()})
    res = run_scripted(senv)
    spans = res.trace.dicts()
    orig = workflows.Run.run_tool_calls

    async def drops_results(self, calls, origin, *, parallel):
        out = []
        for c in calls:
            out.append(await self.call_tool(c, origin))
        return out  # the patched path forgets to log the tool results

    monkeypatch.setattr(workflows.Run, "run_tool_calls", drops_results)
    rep = replay_spans(spans, lambda s: senv(s, None))
    monkeypatch.setattr(workflows.Run, "run_tool_calls", orig)
    second_model = [s for s in spans if s["kind"] == "model"][1]
    assert rep["divergence"]["span_id"] == second_model["span_id"]
    assert "request differs" in rep["divergence"]["detail"]
    assert rep["provider_calls"] == 0 and rep["tool_calls"] == 0


# ---- failure table ----------------------------------------------------------------------------


def two_providers(a_script, b_script, **kw):
    return ScriptedEnv({"a": a_script, "b": b_script}, specs=[spec(provider="a"), spec(provider="b")], **kw)


def test_server_error_retries_then_falls_back():
    errs = [ProviderError("server_error", status=503)] * 3
    senv = two_providers(errs, [{"text": "from b", "usage": u()}])
    res = run_scripted(senv, "chat", {"message": "hi"}, config=RunConfig(fallback=["b/m"]))
    assert (res.status, res.text) == ("succeeded", "from b")
    assert (res.meter.provider_calls, res.meter.retries, res.meter.fallbacks) == (4, 2, 1)
    models = kinds(res, "model")
    assert [m["attrs"].get("error.type") for m in models] == ["server_error"] * 3 + [None]
    assert [m["attrs"]["agent_runtime.attempt"] for m in models] == [1, 2, 3, 1]
    assert res.meter.cost_ucu == models[-1]["attrs"]["agent_runtime.cost_ucu"]  # errors are not billed


def test_retries_exhausted_without_fallback_fail_provider_unavailable():
    senv = ScriptedEnv({"a": [ProviderError("timeout")] * 3}, specs=[spec(provider="a")])
    res = run_scripted(senv, "chat", {"message": "hi"})
    assert (res.status, res.failure_reason) == ("failed", "provider_unavailable")
    assert res.meter.provider_calls == 3


def test_rate_limited_waits_at_least_retry_after():
    senv = ScriptedEnv(
        {"a": [ProviderError("rate_limited", retry_after_ms=1000), {"text": "ok", "usage": u()}]},
        specs=[spec(provider="a")],
        latency_ms=0,
    )
    res = run_scripted(senv, "chat", {"message": "hi"})
    m = kinds(res, "model")
    assert res.status == "succeeded" and m[1]["start_ms"] - m[0]["end_ms"] >= 1000


def test_timeout_of_a_slow_call_is_retried():
    senv = ScriptedEnv(
        {"a": [{"text": "slow", "usage": u()}, {"text": "fast", "usage": u()}]},
        specs=[spec(provider="a")],
        latency_ms=[20_000, 10],
    )
    res = run_scripted(senv, "chat", {"message": "hi"})
    m = kinds(res, "model")
    assert res.text == "fast" and m[0]["attrs"]["error.type"] == "timeout" and m[0]["end_ms"] == 10_000


def test_bad_request_is_neither_retried_nor_falls_back():
    senv = two_providers([ProviderError("bad_request", status=400)], [{"text": "b", "usage": u()}])
    res = run_scripted(senv, "chat", {"message": "hi"}, config=RunConfig(fallback=["b/m"]))
    assert (res.status, res.failure_reason, res.meter.provider_calls) == ("failed", "bad_request", 1)


def test_auth_falls_back_without_retry():
    senv = two_providers([ProviderError("auth", status=401)], [{"text": "b", "usage": u()}])
    res = run_scripted(senv, "chat", {"message": "hi"}, config=RunConfig(fallback=["b/m"]))
    assert (res.status, res.text, res.meter.provider_calls, res.meter.retries) == ("succeeded", "b", 2, 0)
    senv = ScriptedEnv({"a": [ProviderError("auth", status=403)]}, specs=[spec(provider="a")])
    res = run_scripted(senv, "chat", {"message": "hi"})
    assert (res.status, res.failure_reason) == ("failed", "auth")


def test_context_length_falls_back_only_to_a_larger_window():
    specs = [
        spec(provider="a", window=4000),
        spec(provider="b", window=4000),
        spec(provider="c", window=16000),
    ]
    senv = ScriptedEnv(
        {
            "a": [ProviderError("context_length")],
            "b": [{"text": "b", "usage": u()}],
            "c": [{"text": "c", "usage": u()}],
        },
        specs=specs,
    )
    res = run_scripted(senv, "chat", {"message": "hi"}, config=RunConfig(fallback=["b/m", "c/m"]))
    assert (res.status, res.text) == ("succeeded", "c")
    senv = ScriptedEnv({"a": [ProviderError("context_length")]}, specs=[spec(provider="a")])
    res = run_scripted(senv, "chat", {"message": "hi"})
    assert (res.status, res.failure_reason) == ("failed", "context_length")


def test_input_that_fits_no_window_fails_context_length_without_a_call():
    senv = ScriptedEnv({"a": [{"text": "x", "usage": u()}]}, specs=[spec(provider="a", window=600)])
    res = run_scripted(senv, "chat", {"message": "x" * 4000})
    assert (res.status, res.failure_reason, res.meter.provider_calls) == ("failed", "context_length", 0)


def slow_tool(name="slow_read", side_effect="read", fail_times=10, code="UNAVAILABLE", timeout=False):
    from pydantic import BaseModel

    class In(BaseModel):
        x: int = 0

    state = {"n": 0}

    async def handler(v):
        state["n"] += 1
        if state["n"] <= fail_times:
            if timeout:
                await asyncio.sleep(100)
            raise ToolError(code, "transient")
        return {"n": state["n"]}

    return Tool(name=name, handler=handler, input_model=In, side_effect=side_effect, timeout_ms=1000), state


def test_read_tool_retried_twice_on_unavailable_write_tool_never():
    tool, state = slow_tool(fail_times=2)
    script = [{"tool_calls": [call("c1", "slow_read")], "usage": u()}, {"text": "done", "usage": u()}]
    res = run_scripted(ScriptedEnv({"p": script}, tools=[tool]))
    t = kinds(res, "tool")[0]
    assert t["status"] == "ok" and t["attrs"]["agent_runtime.retry"] == 2 and state["n"] == 3
    wtool, wstate = slow_tool("slow_write", side_effect="write", fail_times=1)
    script = [{"tool_calls": [call("c1", "slow_write")], "usage": u()}, {"text": "done", "usage": u()}]
    res = run_scripted(ScriptedEnv({"p": script}, tools=[wtool]))
    t = kinds(res, "tool")[0]
    assert t["status"] == "error" and t["attrs"]["agent_runtime.retry"] == 0 and wstate["n"] == 1
    assert t["result"]["error"]["code"] == "UNAVAILABLE"


def test_tool_timeout_raises_and_invalid_output_are_tool_results_not_exceptions():
    from pydantic import BaseModel

    class In(BaseModel):
        x: int = 0

    class Out(BaseModel):
        y: int

    async def raises(v):
        raise RuntimeError("kaboom")

    async def bad_output(v):
        return {"y": "not an int"}

    tools = [
        slow_tool("hangs", fail_times=99, timeout=True)[0],
        Tool(name="raises", handler=raises, input_model=In),
        Tool(name="bad_out", handler=bad_output, input_model=In, output_model=Out),
    ]
    script = [
        {"tool_calls": [call("c1", "hangs")], "usage": u()},
        {"tool_calls": [call("c2", "raises")], "usage": u()},
        {"tool_calls": [call("c3", "bad_out")], "usage": u()},
        {"tool_calls": [call("c4", "lookup", key=5)], "usage": u()},
        {"tool_calls": [call("c5", "nonexistent")], "usage": u()},
        {"text": "done", "usage": u()},
    ]
    res = run_scripted(
        ScriptedEnv(
            {"p": script},
            tools=tools + [__import__("agent_runtime.mcp.twin", fromlist=["x"]).native_lookup_tool()],
        )
    )
    assert res.status == "succeeded"
    t = kinds(res, "tool")
    assert [x["result"]["error"]["code"] for x in t] == [
        "TIMEOUT",
        "FAILED",
        "INVALID_OUTPUT",
        "INVALID_ARGUMENTS",
        "UNKNOWN_TOOL",
    ]
    assert t[0]["attrs"]["error.type"] == "tool_timeout" and t[0]["attrs"]["agent_runtime.retry"] == 2
    assert t[1]["attrs"]["error.type"] == "tool_error"


def test_repeated_invalid_tool_calls_use_up_the_repair_budget():
    bad = [{"tool_calls": [call(f"c{i}", "lookup", key=1)], "usage": u()} for i in range(3)]
    res = run_scripted(ScriptedEnv({"p": bad}))
    assert (res.status, res.failure_reason, res.meter.repairs) == ("failed", "invalid_output", 3)


# ---- structured output (check 5) ----------------------------------------------------------------

from agent_runtime.runtime.engine import workflow  # noqa: E402


@workflow("_test_structured")
async def _structured(run, inp):
    from agent_runtime.models.types import Message

    run.append(Message(role="user", content="give json"))
    async with run.step("s"):
        value, _ = await run.call_structured("answer", SCHEMA)
    return value["answer"]


@pytest.mark.parametrize("bad", MALFORMED_FIXTURES)
def test_each_malformed_fixture_is_repaired(bad):
    script = [{"text": bad, "usage": u()}, {"text": '{"answer": "fixed", "n": 1}', "usage": u()}]
    res = run_scripted(ScriptedEnv({"p": script}), "_test_structured", {})
    assert (res.status, res.text, res.meter.repairs) == ("succeeded", "fixed", 1)
    v = kinds(res, "validate")
    assert [x["status"] for x in v] == ["error", "ok"]


def test_structured_output_repair_budget_then_next_model_then_invalid_output():
    bad = [{"text": "nope", "usage": u()}] * 3
    senv = two_providers(bad, [{"text": '{"answer": "b", "n": 2}', "usage": u()}])
    res = run_scripted(
        senv, "_test_structured", {}, router={"type": "fixed", "target": "a/m", "fallback": ["b/m"]}
    )
    assert (res.status, res.text, res.meter.repairs) == ("succeeded", "b", 3)
    senv = ScriptedEnv({"a": bad}, specs=[spec(provider="a")])
    res = run_scripted(senv, "_test_structured", {})
    assert (res.status, res.failure_reason) == ("failed", "invalid_output")


# ---- budgets and cancellation (check 8) ----------------------------------------------------------


def looping_script(n=10):
    return [{"tool_calls": [call(f"c{i}", key="beta")], "usage": u(1000, 100)} for i in range(n)]


@pytest.mark.parametrize(
    "budget,reason_calls",
    [
        (Budget(max_model_calls=3), 3),
        (Budget(max_cost_ucu=2500), 2),  # 1400 ucu per call: the third call does not start
        (Budget(deadline_ms=250), 3),  # calls of 100 ms: starts at 0, 100, 200; none at 300
    ],
)
def test_each_budget_ends_the_run_without_another_call(budget, reason_calls):
    res = run_scripted(ScriptedEnv({"p": looping_script()}), config=RunConfig(budget=budget))
    assert res.status == "budget_exhausted"
    assert res.meter.provider_calls == reason_calls


def test_cancel_in_the_middle_of_a_parallel_step_starts_no_new_call():
    from pydantic import BaseModel

    class In(BaseModel):
        k: int

    started = []

    async def slowpoke(v):
        started.append(v.k)
        await holder["clock"].sleep(1000)
        return {"k": v.k}

    holder = {}
    tool = Tool(name="slowpoke", handler=slowpoke, input_model=In, side_effect="read")
    script = [
        {"tool_calls": [call(f"c{i}", "slowpoke", k=i) for i in range(6)], "usage": u()},
        {"text": "never", "usage": u()},
    ]
    senv = ScriptedEnv({"p": script}, tools=[tool])
    rs = RunSpec(
        run_id="r-c",
        workflow="react_parallel",
        input={"task": "t"},
        config=RunConfig(concurrency=2),
        default_target="p/m",
    )
    store = MemoryStore()
    sid = store.create_session()

    async def main(clock):
        holder["clock"] = clock
        env = senv(rs, clock)
        runs = []
        task = asyncio.ensure_future(execute(rs, env, clock, store, sid, on_start=runs.append))
        await clock.sleep(1500)  # the first two calls finished, the next two are running
        runs[0].cancel()
        res = await task
        await clock.sleep(5000)
        return res, env

    res, env = run_virtual(main)
    from agent_runtime.telemetry.trace import validate_trace

    assert res.status == "cancelled"
    assert started == [0, 1, 2, 3]  # no call started after the cancellation
    assert senv.models["p"].remaining == 1  # the final model call never started
    spans = res.trace.dicts()
    validate_trace(spans)
    assert all(s["end_ms"] is not None for s in spans)
    # spans exist only for calls that started (the others waited for the concurrency limit)
    assert [s["status"] for s in spans if s["kind"] == "tool"] == ["ok", "ok", "cancelled", "cancelled"]


def test_a_rejected_final_answer_fails_the_run_with_verifier_rejected():
    senv = ScriptedEnv({"p": [{"text": "wrong", "usage": u()}]}, verifier=lambda text, attempt: False)
    res = run_scripted(senv, "react", {"task": "t"})
    assert (res.status, res.failure_reason, res.accepted) == ("failed", "verifier_rejected", False)
    assert [s["attrs"]["agent_runtime.verifier.accepted"] for s in kinds(res, "validate")] == [False]
