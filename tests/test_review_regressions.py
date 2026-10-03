"""Regression tests for the findings of Review 1 (each test names its finding)."""

import collections

from agent_runtime.evals.simrun import Job, run_job, sim_env_factory
from agent_runtime.evals.stats import COST, SUCCESS, wtl
from agent_runtime.models.types import ProviderError
from agent_runtime.permissions import PermissionPolicy, decide, exe_name
from agent_runtime.routing import RandomMixRouter, RouteContext
from agent_runtime.runtime.engine import Budget, RunConfig
from agent_runtime.runtime.runner import replay_spans
from agent_runtime.sim.tasks import gen
from agent_runtime.store.memory import MemoryStore
from agent_runtime.telemetry.trace import load_jsonl
from helpers import ScriptedEnv, call, kinds, run_scripted, spec, u


def test_f1_the_cascade_escalates_at_most_along_its_chain():
    out = run_job(
        Job(
            "t",
            "cascade",
            3,
            n_tasks=60,
            router={"type": "cascade", "chain": ["sim-a/small", "sim-a/medium", "sim-a/large"]},
            run_config={"budget": {"deadline_ms": None, "max_model_calls": 200}},
        )
    )
    assert max(r["escalations"] for r in out["rows"]) <= 2
    assert any(r["escalations"] == 2 for r in out["rows"])


def test_f2_child_and_attempt_conversations_do_not_leak_across_runs_of_a_session():
    from agent_runtime.agents import delegate_tool
    from agent_runtime.clock import run_virtual
    from agent_runtime.mcp.twin import native_lookup_tool
    from agent_runtime.models import ModelRegistry
    from agent_runtime.runtime.engine import Env
    from agent_runtime.runtime.runner import RunSpec, execute
    from agent_runtime.tools import ToolRegistry
    from test_subagents import CATALOG, ByTask  # noqa: F401

    store = MemoryStore()
    sid = store.create_session()
    models = []
    for rid in ("r-1", "r-2"):
        scripts = {
            "root": [
                {"tool_calls": [call("d1", "delegate", agent="finder", task=f"find X {rid}")], "usage": u()},
                {"text": "ok", "usage": u()},
            ],
            "root/c1": [{"text": f"child answer {rid}", "usage": u()}],
        }

        async def main(clock, scripts=scripts, rid=rid):
            m = ByTask(scripts, clock)
            models.append(m)
            env = Env(
                registry=ModelRegistry([spec(provider="p", window=100_000)]),
                adapters={"p": m},
                tools=ToolRegistry([native_lookup_tool(), delegate_tool(CATALOG)]),
                agents=CATALOG,
            )
            rs = RunSpec(
                run_id=rid, workflow="react", input={"task": "t"}, task_id="root", default_target="p/m"
            )
            return await execute(rs, env, clock, store, sid)

        assert run_virtual(main).status == "succeeded"
    second_child = models[1].requests["root/c1"][0]
    assert [m.content for m in second_child.messages] == [CATALOG["finder"].system_prompt, "find X r-2"]


def test_f3_the_budget_is_checked_before_the_first_attempt_of_a_fallback_target():
    errs = [ProviderError("server_error")] * 3
    senv = ScriptedEnv(
        {"a": errs, "b": [{"text": "b", "usage": u()}]}, specs=[spec(provider="a"), spec(provider="b")]
    )
    res = run_scripted(
        senv, "chat", {"message": "hi"}, config=RunConfig(fallback=["b/m"], budget=Budget(max_model_calls=3))
    )
    assert res.status == "budget_exhausted" and res.meter.provider_calls == 3


def test_f4_random_mix_does_not_depend_on_the_order_of_its_probabilities():
    a = RandomMixRouter({"x": 0.25, "y": 0.25, "z": 0.5}, seed=1)
    b = RandomMixRouter({"z": 0.5, "y": 0.25, "x": 0.25}, seed=1)

    def ctx(t):
        return RouteContext("act", ("x", "y", "z"), 0, 0, 0, 1, (), None, t, None, {})

    assert [a.choose(ctx(f"t{i}")).target for i in range(200)] == [
        b.choose(ctx(f"t{i}")).target for i in range(200)
    ]
    out = run_job(
        Job(
            "t",
            "mix",
            1,
            n_tasks=15,
            router={
                "type": "random_mix",
                "probs": {"sim-a/small": 0.25, "sim-a/medium": 0.25, "sim-a/large": 0.5},
            },
            keep_traces=True,
        )
    )
    for text in out["traces"].values():
        assert replay_spans(load_jsonl(text), sim_env_factory(out["job"]))["divergence"] is None


def test_f5_a_child_stopped_by_its_share_after_issuing_calls_leaves_a_valid_trace():
    from agent_runtime.agents import AgentSpec
    from test_subagents import CATALOG, run

    cat = {**CATALOG, "finder": AgentSpec(name="finder", tools=["lookup"], max_cost_ucu=50)}
    scripts = {
        "root": [
            {"tool_calls": [call("d1", "delegate", agent="finder", task="t")], "usage": u()},
            {"text": "ok", "usage": u()},
        ],
        "root/c1": [
            {"tool_calls": [call("k1", "lookup", key="beta")], "usage": u()},
            {"text": "never", "usage": u()},
        ],
    }
    res, spans, _, _ = run(scripts, catalog=cat)  # validate_trace and replay run inside
    assert res.status == "succeeded"


def test_f6_replay_of_budget_exhausted_parallel_traces_does_not_diverge():
    for deadline in (4000, 6000, 9000):
        job = Job(
            "t",
            f"rp{deadline}",
            5,
            workflow="react_parallel",
            n_tasks=20,
            shape="wide",
            run_config={"concurrency": 1, "budget": {"deadline_ms": deadline}},
            keep_traces=True,
        )
        out = run_job(job)
        assert any(r["status"] == "budget_exhausted" for r in out["rows"])
        for text in out["traces"].values():
            rep = replay_spans(load_jsonl(text), sim_env_factory(job))
            assert rep["divergence"] is None, rep


def test_f8_replay_detects_a_changed_capability_and_a_tool_dropped_from_the_allowlist():
    from pydantic import BaseModel

    from agent_runtime.tools import Tool

    class In(BaseModel):
        x: int = 0

    async def h(v):
        return {"ok": 1}

    def tools(cap):
        return [Tool(name="writer", handler=h, input_model=In, capability=cap, side_effect="write")]

    script = [{"tool_calls": [call("w1", "writer", x=1)], "usage": u()}, {"text": "done", "usage": u()}]
    senv = ScriptedEnv({"p": script}, tools=lambda: tools("fs_write"))
    res = run_scripted(senv, check_replay=False)
    assert kinds(res, "permission")
    changed = ScriptedEnv({"p": script}, tools=lambda: tools("none"))
    rep = replay_spans(res.trace.dicts(), lambda s: changed(s, None))
    assert rep["divergence"] is not None and "permission" in rep["divergence"]["detail"]


def test_f9_a_trailing_dot_does_not_get_past_deny_exec():
    assert exe_name("cmd.exe.") == "cmd" and exe_name("C:/Windows/System32/CMD.EXE. ") == "cmd"
    d = decide(
        PermissionPolicy(approval="auto"), "exec", {"paths_ok": True, "argv": ["cmd.exe.", "/c", "dir"]}
    )
    assert (d.decision, d.rule, d.final) == ("deny", "deny_exec", True)


def test_f10_context_length_never_falls_back_to_a_window_that_is_not_larger():
    specs = [
        spec(provider="m", window=16000),
        spec(provider="l", window=32000),
        spec(provider="s", window=4000),
    ]
    senv = ScriptedEnv(
        {
            "m": [ProviderError("context_length")],
            "l": [ProviderError("server_error")] * 3,
            "s": [{"text": "s", "usage": u()}],
        },
        specs=specs,
    )
    res = run_scripted(
        senv, "chat", {"message": "hi"}, config=RunConfig(fallback=["l/m", "s/m"]), default_target="m/m"
    )
    assert res.status == "failed" and "s" not in {
        m["attrs"]["gen_ai.provider.name"] for m in kinds(res, "model")
    }


def test_f11_wide_tasks_cover_4_to_10_steps():
    c = collections.Counter(len(t.steps) for t in gen(101, 2000, "mixed", "wide"))
    assert set(c) == set(range(4, 11))


def test_f12_win_tie_loss_follows_dominance():
    ref = {1: {"success": 0.5, "cost_ucu": 100.0}}
    assert wtl(ref, {1: {"success": 0.6, "cost_ucu": 100.0}}, SUCCESS, [COST]) == {
        "win": 1,
        "tie": 0,
        "loss": 0,
    }
    assert wtl(ref, {1: {"success": 0.6, "cost_ucu": 150.0}}, SUCCESS, [COST]) == {
        "win": 0,
        "tie": 1,
        "loss": 0,
    }  # trade-off
    assert wtl(ref, {1: {"success": 0.5, "cost_ucu": 150.0}}, SUCCESS, [COST]) == {
        "win": 0,
        "tie": 0,
        "loss": 1,
    }  # dominated
    assert wtl(ref, {1: {"success": 0.4, "cost_ucu": 50.0}}, SUCCESS, [COST]) == {
        "win": 0,
        "tie": 0,
        "loss": 1,
    }
