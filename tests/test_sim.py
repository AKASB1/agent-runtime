"""The simulator and the runner: world and generator, the simulated model, determinism (check 2),
and the replay of a seeded batch of 200 runs (check 1)."""

import collections
import math

import pytest

from agent_runtime.evals.experiments import run_jobs
from agent_runtime.evals.simrun import Job, run_job, sim_env_factory
from agent_runtime.planning.plan import validate_plan
from agent_runtime.runtime.runner import replay_spans
from agent_runtime.sim.tasks import gen, perturb, tier_of
from agent_runtime.sim.world import WorldSession, make_world, pure_call
from agent_runtime.telemetry.trace import load_jsonl, validate_trace
from agent_runtime.tools import ToolError, ToolRegistry


def test_world_is_seeded_and_sized():
    w = make_world(1)
    assert (len(w.customers), len(w.products), len(w.orders)) == (200, 600, 2000)
    assert make_world(1) is w
    assert w.customers["C001"] == make_world(1).customers["C001"]
    with pytest.raises(ToolError) as ei:
        pure_call(w, "get_customer", {"customer_id": "C999"})
    assert ei.value.code == "NOT_FOUND"


@pytest.mark.parametrize("shape", ["chain", "wide"])
def test_generator_tiers_gold_plans_and_answers(shape):
    tasks = gen(5, 400, "mixed", shape)
    assert [t.id for t in tasks] == [t.id for t in gen(5, 400, "mixed", shape)]  # reproducible
    tools = ToolRegistry(WorldSession(make_world(1), None, 5, "r").tools())
    counts = collections.Counter(t.tier for t in tasks)
    for t in tasks:
        assert t.tier == tier_of(len(t.steps))
        plan = {
            "steps": [
                {"id": s.id, "tool": s.tool, "args": s.args, "depends_on": list(s.depends_on)}
                for s in t.steps
            ]
        }
        validate_plan(plan, tools)
        assert perturb(t.answer) != t.answer
        if shape == "chain":
            assert 1 <= len(t.steps) <= 8
        else:
            assert 4 <= len(t.steps) <= 10
    if shape == "chain":
        assert abs(counts["easy"] / 400 - 0.5) < 0.08 and abs(counts["hard"] / 400 - 0.15) < 0.06
    reports = sum(t.report for t in tasks) / len(tasks)
    assert 0.12 < reports < 0.28
    # the hint is the tier plus noise of sigma 0.6
    resid = [t.hint - t.tier_index for t in tasks]
    sd = math.sqrt(sum(r * r for r in resid) / len(resid))
    assert 0.5 < sd < 0.7


def test_gold_answer_is_what_the_gold_plan_computes():
    w = make_world(1)
    for t in gen(9, 60, "hard_heavy", "chain"):
        results = {s.id: pure_call(w, s.tool, s.args) for s in t.steps if s.tool != "send_report"}
        last = [s for s in t.steps if s.tool != "send_report"][-1]
        r = results[last.id]
        if last.tool == "get_customer":
            assert r["country"] == t.answer
        elif last.tool == "get_product":
            assert f"{r['price']:.2f} {r['currency']}" == t.answer
        elif last.tool == "calc":
            assert f"{r['value']:.2f} USD" == t.answer
        else:
            assert t.answer.startswith(f"{r['amount']:.2f}")


def test_a_sim_job_is_deterministic_and_fast_enough():
    a = run_job(Job("t", "x", 3, n_tasks=60, keep_traces=True))
    b = run_job(Job("t", "x", 3, n_tasks=60, keep_traces=True))
    assert a["rows"] == b["rows"] and a["traces"] == b["traces"]


def test_determinism_one_worker_vs_four_and_independence_of_concurrency():
    jobs = [
        Job("t", f"c{i}", s, workflow=wf, n_tasks=40, router={"type": "fixed", "target": "sim-a/small"})
        for i, wf in enumerate(["react", "plan_replan"])
        for s in (1, 2, 3)
    ]
    one = run_jobs(jobs, 1)
    four = run_jobs(jobs, 4)
    assert [r["rows"] for r in one] == [r["rows"] for r in four]
    # a task's calls, outcomes, and own latency do not depend on which other tasks run with it:
    # the same tasks as Poisson arrivals with many in flight give the same per-task rows
    seq = run_job(Job("t", "x", 4, n_tasks=30))["rows"]
    conc = run_job(Job("t", "x", 4, arrivals={"rate_per_s": 5.0, "duration_s": 10, "max_in_flight": 1000}))[
        "rows"
    ][:30]
    keys = ("success", "cost_ucu", "latency_ms", "model_calls", "tool_calls", "status")
    by_id = {r["task_id"]: r for r in conc}
    for r in seq:
        assert {k: r[k] for k in keys} == {k: by_id[r["task_id"]][k] for k in keys}


def replay_batch_jobs():
    jobs = []
    modes = ["react", "react_parallel", "plan_execute", "plan_replan"]
    for i, mode in enumerate(modes):
        jobs.append(
            Job(
                "rb",
                f"{mode}-faults",
                7 + i,
                workflow=mode,
                n_tasks=21,
                providers=("sim-a", "sim-b"),
                faults={"sim-a": "flaky"},
                run_config={"fallback": ["sim-b/medium"]},
                keep_traces=True,
            )
        )
        jobs.append(
            Job(
                "rb",
                f"{mode}-ctx",
                11 + i,
                workflow=mode,
                n_tasks=21,
                result_tokens=1500,
                router={"type": "fixed", "target": "sim-a/small", "fallback": ["sim-a/medium"]},
                run_config={"context": {"policy": ["window", "elide", "summarize", "elide"][i]}},
                keep_traces=True,
            )
        )
    jobs.append(
        Job(
            "rb",
            "cascade",
            21,
            n_tasks=32,
            router={"type": "cascade", "chain": ["sim-a/small", "sim-a/medium", "sim-a/large"]},
            run_config={"budget": {"deadline_ms": 30000}},
            keep_traces=True,
        )
    )
    return jobs


def test_replay_of_a_seeded_batch_of_200_runs():
    jobs = replay_batch_jobs()
    results = run_jobs(jobs, 4)
    n = 0
    compacted = 0
    for r in results:
        factory = sim_env_factory(r["job"])
        for rid, text in sorted(r["traces"].items()):
            spans = load_jsonl(text)
            validate_trace(spans)
            rep = replay_spans(spans, factory)
            assert rep["divergence"] is None, (rid, rep)
            assert rep["provider_calls"] == 0 and rep["tool_calls"] == 0
            compacted += any(s["kind"] == "context" for s in spans)
            n += 1
    assert n == 200
    assert compacted > 30
