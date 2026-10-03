"""Tier 2 item 5: the delegate planner mode of E6 (branches, children, replay, keep_delegate)."""

import statistics

from agent_runtime.evals.simrun import Job, run_job, sim_env_factory
from agent_runtime.planning.plan import PlanStep, plan_branches, plan_tail
from agent_runtime.runtime.runner import replay_spans
from agent_runtime.sim.tasks import gen
from agent_runtime.telemetry.trace import load_jsonl, validate_trace


def S(i, *deps):
    return PlanStep(i, "x", {}, tuple(deps))


def test_tail_and_branches_of_hand_made_plans():
    # two independent branches joined by a calc step and a convert after it
    plan = [S("a1"), S("a2", "a1"), S("b1"), S("b2", "b1"), S("j", "a2", "b2"), S("t", "j")]
    assert plan_tail(plan) == ["j", "t"]
    assert plan_branches(plan) == [["a1", "a2"], ["b1", "b2"]]
    # a shared root makes one branch
    shared = [S("r"), S("p1", "r"), S("p2", "r"), S("j", "p1", "p2")]
    assert plan_branches(shared) == [["r", "p1", "p2"]]
    # no join step: no tail, one branch per component
    chain = [S("x1"), S("x2", "x1")]
    assert plan_tail(chain) == [] and plan_branches(chain) == [["x1", "x2"]]


def test_wide_multi_order_tasks_have_independent_branches():
    tasks = [t for t in gen(7, 200, "mixed", "wide") if len(t.steps) >= 5]
    counts = [len(plan_branches(list(t.steps))) for t in tasks]
    assert max(counts) >= 3 and min(counts) >= 1


def delegate_job(keep=0.95, n=40, seed=3):
    return Job(
        "t",
        f"delegate{keep}",
        seed,
        workflow="delegate_plan",
        n_tasks=n,
        shape="wide",
        result_tokens=1500,
        router={"type": "fixed", "target": "sim-a/medium"},
        keep_delegate=keep,
        keep_traces=True,
    )


def test_children_run_one_per_branch_and_the_runs_replay():
    job = delegate_job()
    out = run_job(job)
    tasks = {t.id: t for t in gen(3, 40, "mixed", "wide")}
    checked = 0
    for rid, text in sorted(out["traces"].items()):
        spans = load_jsonl(text)
        validate_trace(spans)
        subs = [s for s in spans if s["kind"] == "subagent"]
        plans = [s for s in spans if s["kind"] == "plan"]
        if plans and spans[0]["status"] == "ok":
            steps = [
                PlanStep(st["id"], st["tool"], {}, tuple(st["depends_on"]))
                for st in plans[0]["attrs"]["agent_runtime.plan.steps"]
            ]
            assert len(subs) == len(plan_branches(steps))
            assert all(s["attrs"]["agent_runtime.agent.depth"] == 1 for s in subs)
            checked += 1
        rep = replay_spans(spans, sim_env_factory(job))
        assert rep["divergence"] is None, (rid, rep)
    assert checked > 10 and tasks


def test_lost_branch_results_make_the_parent_answer_wrong():
    lost = run_job(delegate_job(keep=0.0, n=40))["rows"]
    kept = run_job(delegate_job(keep=1.0, n=40))["rows"]
    assert statistics.mean(r["success"] for r in lost) == 0.0
    assert statistics.mean(r["success"] for r in kept) > 0.1
    # the parent's own context stays small: the children's tool results never enter it
    react = run_job(
        Job(
            "t",
            "react",
            3,
            workflow="react",
            n_tasks=40,
            shape="wide",
            result_tokens=1500,
            router={"type": "fixed", "target": "sim-a/medium"},
        )
    )["rows"]
    assert statistics.mean(r["peak_parent_context_tokens"] for r in kept) < statistics.mean(
        r["peak_parent_context_tokens"] for r in react
    )
