"""Check 15: subagents (delegate, allowlist, fresh context, capped result, shared meter, share,
remainder, depth limit, nested spans, replay, cancellation, parallel children)."""

import asyncio

from agent_runtime.agents import CAP_MARKER, AgentSpec, delegate_tool
from agent_runtime.clock import run_virtual
from agent_runtime.mcp.twin import native_lookup_tool
from agent_runtime.models import ModelRegistry
from agent_runtime.providers.scripted import response_from_item
from agent_runtime.runtime.engine import Budget, Env, RunConfig
from agent_runtime.runtime.runner import RunSpec, execute, replay_spans
from agent_runtime.store.memory import MemoryStore
from agent_runtime.telemetry.trace import validate_trace
from agent_runtime.tools import ToolRegistry
from helpers import call, u
from helpers import spec as mkspec


class ByTask:
    """Scripted model keyed by the request's task id (the parent is 'root', children 'root/c1', ...)."""

    def __init__(self, scripts, clock=None, latency=100):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests = {}
        self.clock = clock
        self.latency = latency

    async def complete(self, request, spec):
        tid = request.meta["task_id"]
        self.requests.setdefault(tid, []).append(request)
        if self.clock:
            await self.clock.sleep(self.latency)
        return response_from_item(self.scripts[tid].pop(0), request, spec)


CATALOG = {
    "finder": AgentSpec(name="finder", description="looks things up", tools=["lookup"], max_steps=4),
    "writer": AgentSpec(name="writer", tools=["lookup", "secret_write"], max_steps=3, max_model_calls=2),
    "nester": AgentSpec(name="nester", tools=["delegate", "lookup"], max_steps=3),
}


def secret_tool():
    from pydantic import BaseModel

    from agent_runtime.tools import Tool

    class In(BaseModel):
        x: int = 0

    async def h(v):
        return {"done": True}

    return Tool(name="secret_write", handler=h, input_model=In, side_effect="write")


def build(scripts, *, config=None, catalog=None, workflow="react", parent_tools=None):
    holder = {}

    def factory(spec=None, clock=None):
        cat = catalog or CATALOG
        tools = [native_lookup_tool(), delegate_tool(cat)] + ([secret_tool()] if parent_tools else [])
        model = ByTask(scripts, clock)
        holder.setdefault("model", model)
        return Env(
            registry=ModelRegistry([mkspec(provider="p", window=100_000)]),
            adapters={"p": model},
            tools=ToolRegistry(tools),
            agents=cat,
        )

    rs = RunSpec(
        run_id="r-sub",
        workflow=workflow,
        input={"task": "parent task with history"},
        task_id="root",
        config=config or RunConfig(),
        default_target="p/m",
    )
    return factory, rs, holder


def run(scripts, check_replay=True, **kw):
    factory, rs, holder = build(scripts, **kw)
    store = MemoryStore()
    sid = store.create_session()

    async def main(clock):
        return await execute(rs, factory(rs, clock), clock, store, sid)

    res = run_virtual(main)
    spans = res.trace.dicts()
    validate_trace(spans)
    if check_replay:
        rep = replay_spans(spans, lambda s: factory(s, None))
        assert rep["divergence"] is None, rep
        assert rep["provider_calls"] == 0 and rep["tool_calls"] == 0
    return res, spans, holder["model"], store


def test_child_has_a_fresh_context_and_the_parent_sees_only_the_capped_result():
    long_answer = "z" * 3000
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="finder", task="find beta")], "usage": u()},
            {"text": "parent done", "usage": u()},
        ],
        "root/c1": [
            {"tool_calls": [call("k1", "lookup", key="beta")], "usage": u()},
            {"text": long_answer, "usage": u()},
        ],
    }
    res, spans, model, _ = run(scripts)
    assert res.status == "succeeded" and res.meter.subagent_runs == 1
    first_child = model.requests["root/c1"][0]
    contents = [m.content for m in first_child.messages]
    assert contents == [CATALOG["finder"].system_prompt, "find beta"]  # nothing of the parent's history
    assert [t.name for t in first_child.tools] == ["lookup"]
    parent_next = model.requests["root"][1]
    tool_msg = parent_next.messages[-1]
    assert tool_msg.role == "tool" and CAP_MARKER in tool_msg.content and "k1" not in tool_msg.content
    assert len(tool_msg.content) < 1800
    # spans: the child's spans nest under a subagent span under the delegate tool span
    sub = next(s for s in spans if s["kind"] == "subagent")
    assert (
        sub["attrs"]["agent_runtime.agent.name"] == "finder"
        and sub["attrs"]["agent_runtime.agent.depth"] == 1
    )
    by_id = {s["span_id"]: s for s in spans}
    assert by_id[sub["parent_id"]]["kind"] == "tool"
    child_models = [
        s for s in spans if s["kind"] == "model" and s["attrs"]["agent_runtime.context.conv"] == "r-sub/c1"
    ]
    assert len(child_models) == 2
    # one meter: the run's cost covers the child's calls
    assert res.meter.cost_ucu == sum(
        s["attrs"]["agent_runtime.cost_ucu"] for s in spans if s["kind"] == "model"
    )


def test_a_tool_outside_the_childs_allowlist_is_an_error():
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="finder", task="t")], "usage": u()},
            {"text": "ok", "usage": u()},
        ],
        "root/c1": [
            {"tool_calls": [call("x1", "secret_write", x=1)], "usage": u()},
            {"text": "could not", "usage": u()},
        ],
    }
    res, spans, _, _ = run(scripts, parent_tools=True)
    t = next(s for s in spans if s["kind"] == "tool" and s["attrs"]["gen_ai.tool.call.id"] == "x1")
    assert t["result"]["error"]["code"] == "UNKNOWN_TOOL" and "allowlist" in t["result"]["error"]["message"]


def test_the_childs_share_ends_it_budget_exhausted_and_the_parent_gets_subagent_failed():
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="writer", task="t")], "usage": u()},
            {"text": "parent goes on", "usage": u()},
        ],
        "root/c1": [{"tool_calls": [call(f"k{i}", "lookup", key="beta")], "usage": u()} for i in range(5)],
    }
    res, spans, model, _ = run(scripts)
    assert res.status == "succeeded" and res.text == "parent goes on"
    assert len(model.requests["root/c1"]) == 2  # max_model_calls 2 of the child's share
    t = next(s for s in spans if s["kind"] == "tool" and s["attrs"]["gen_ai.tool.call.id"] == "d1")
    assert (
        t["attrs"]["error.type"] == "subagent_failed"
        and t["result"]["error"]["details"]["status"] == "budget_exhausted"
    )


def test_the_parents_remainder_bounds_the_child():
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="finder", task="t")], "usage": u()},
            {"text": "never", "usage": u()},
        ],
        "root/c1": [{"tool_calls": [call(f"k{i}", "lookup", key="beta")], "usage": u()} for i in range(5)],
    }
    res, spans, model, _ = run(scripts, config=RunConfig(budget=Budget(max_model_calls=3)))
    assert len(model.requests["root/c1"]) == 2  # 3 run calls - 1 used by the parent
    assert res.status == "budget_exhausted" and res.meter.provider_calls == 3


def test_depth_limit_a_child_cannot_delegate_by_default_but_can_at_depth_two():
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="nester", task="t")], "usage": u()},
            {"text": "p", "usage": u()},
        ],
        "root/c1": [
            {"tool_calls": [call("d2", "delegate", agent="finder", task="t2")], "usage": u()},
            {"text": "c", "usage": u()},
        ],
        "root/c2": [{"text": "grandchild", "usage": u()}],
    }
    res, spans, model, _ = run(scripts)
    t = next(s for s in spans if s["kind"] == "tool" and s["attrs"]["gen_ai.tool.call.id"] == "d2")
    assert t["result"]["error"]["code"] == "UNKNOWN_TOOL"  # delegate is not in a depth-1 child's tools
    assert [t.name for t in model.requests["root/c1"][0].tools] == ["lookup"]
    scripts["root/c1"] = [
        {"tool_calls": [call("d2", "delegate", agent="finder", task="t2")], "usage": u()},
        {"text": "c", "usage": u()},
    ]
    scripts["root/c1/c2"] = [{"text": "grandchild", "usage": u()}]
    res, spans, model, _ = run(scripts, config=RunConfig(max_depth=2))
    subs = [s for s in spans if s["kind"] == "subagent"]
    assert [s["attrs"]["agent_runtime.agent.depth"] for s in subs] == [1, 2] and res.status == "succeeded"


def test_cancelling_the_parent_while_a_child_runs_stops_both():
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="finder", task="t")], "usage": u()},
            {"text": "never", "usage": u()},
        ],
        "root/c1": [{"tool_calls": [call(f"k{i}", "lookup", key="beta")], "usage": u()} for i in range(4)],
    }
    factory, rs, holder = build(scripts)
    store = MemoryStore()

    async def main(clock):
        runs = []
        env = factory(rs, clock)
        task = asyncio.ensure_future(
            execute(rs, env, clock, store, store.create_session(), on_start=runs.append)
        )
        await clock.sleep(250)  # parent call (100) + first child call (100) done; the second child call runs
        runs[0].cancel()
        res = await task
        await clock.sleep(10_000)
        return res, env.adapters["p"]

    res, model = run_virtual(main)
    assert res.status == "cancelled"
    assert (
        len(model.requests["root/c1"]) == 2 and len(model.requests["root"]) == 1
    )  # no call started afterwards
    validate_trace(res.trace.dicts())


def test_two_children_in_parallel_under_the_concurrency_limit_are_deterministic():
    scripts = {
        "root": [
            {
                "tool_calls": [
                    call("d1", "delegate", agent="finder", task="a"),
                    call("d2", "delegate", agent="finder", task="b"),
                ],
                "usage": u(),
            },
            {"text": "both done", "usage": u()},
        ],
        "root/c1": [
            {"tool_calls": [call("k1", "lookup", key="alpha")], "usage": u()},
            {"text": "A", "usage": u()},
        ],
        "root/c2": [
            {"tool_calls": [call("k2", "lookup", key="beta")], "usage": u()},
            {"text": "B", "usage": u()},
        ],
    }
    traces = []
    for _ in range(2):
        res, spans, _, _ = run(
            {k: list(v) for k, v in scripts.items()},
            workflow="react_parallel",
            config=RunConfig(concurrency=4),
        )
        traces.append(res.trace.jsonl())
        subs = [s for s in spans if s["kind"] == "subagent"]
        assert len(subs) == 2 and subs[0]["start_ms"] == subs[1]["start_ms"]  # they overlap in time
    assert traces[0] == traces[1]
    res1, spans1, _, _ = run(
        {k: list(v) for k, v in scripts.items()}, workflow="react_parallel", config=RunConfig(concurrency=1)
    )
    assert res1.status == "succeeded"


def test_subagent_failed_is_the_tool_result_code():
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="writer", task="t")], "usage": u()},
            {"text": "ok", "usage": u()},
        ],
        "root/c1": [{"tool_calls": [call(f"k{i}", "lookup", key="beta")], "usage": u()} for i in range(5)],
    }
    res, spans, model, _ = run(scripts)
    t = next(s for s in spans if s["kind"] == "tool" and s["attrs"]["gen_ai.tool.call.id"] == "d1")
    assert t["result"]["error"]["code"] == "SUBAGENT_FAILED"
    assert "SUBAGENT_FAILED" in model.requests["root"][1].messages[-1].content
