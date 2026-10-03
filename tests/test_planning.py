"""Check 10: plan validation, list scheduling (Graham's bound), replanning, write safety."""

import pytest
from pydantic import BaseModel

from agent_runtime.canon import canonical
from agent_runtime.clock import run_virtual
from agent_runtime.models.structured import StructuredOutputError
from agent_runtime.planning.plan import PlanStep, critical_path, run_dag, validate_plan, width
from agent_runtime.rand import below, stream
from agent_runtime.sim.world import WorldSession, make_world
from agent_runtime.tools import Tool, ToolError, ToolRegistry
from helpers import ScriptedEnv, kinds, run_scripted, u

TOOLS = ToolRegistry(WorldSession(make_world(1), None, 1, "r").tools())


def plan(*steps):
    return {"steps": [{"id": i, "tool": t, "args": a, "depends_on": list(d)} for i, t, a, d in steps]}


GOOD = plan(
    ("s1", "get_order", {"order_id": "O0001"}, []), ("s2", "get_product", {"product_id": "P001"}, ["s1"])
)


@pytest.mark.parametrize(
    "bad,cls",
    [
        (
            plan(
                ("s1", "get_order", {"order_id": "O1"}, ["s2"]),
                ("s2", "get_order", {"order_id": "O2"}, ["s1"]),
            ),
            "cycle",
        ),
        (plan(("s1", "get_order", {"order_id": "O1"}, ["s1"])), "cycle"),
        (plan(("s1", "fly_to_moon", {}, [])), "unknown_tool"),
        (plan(("s1", "get_order", {"order_id": 7}, [])), "invalid_args"),
        (plan(("s1", "get_order", {"order_id": "O1"}, ["s9"])), "unresolved_dependency"),
        (plan(*[(f"s{i}", "calc", {"op": "sum", "values": [1]}, []) for i in range(21)]), "plan_too_long"),
        (
            plan(
                ("s1", "calc", {"op": "sum", "values": [1]}, []),
                ("s1", "calc", {"op": "sum", "values": [2]}, []),
            ),
            "duplicate_step",
        ),
        ({"steps": []}, "plan_invalid"),
    ],
)
def test_the_validator_rejects_each_class_of_invalid_plan(bad, cls):
    with pytest.raises(StructuredOutputError) as ei:
        validate_plan(bad, TOOLS)
    assert str(ei.value).startswith(cls)


def test_valid_plan_and_dependencies_on_completed_steps():
    assert [s.id for s in validate_plan(GOOD, TOOLS)] == ["s1", "s2"]
    p = plan(("s3", "calc", {"op": "sum", "values": [1]}, ["s1"]))
    assert validate_plan(p, TOOLS, completed={"s1"})[0].depends_on == ("s1",)


def random_dag(rng, n):
    steps = []
    for i in range(n):
        deps = tuple(f"s{j}" for j in range(i) if rng.random() < 0.25)
        steps.append(PlanStep(f"s{i}", "x", {}, deps))
    durs = {s.id: 1 + below(rng, 9) * 100 for s in steps}
    return steps, durs


def schedule(steps, durs, k):
    started: list[str] = []

    async def main(clock):
        async def run_step(s):
            started.append(s.id)
            await clock.sleep(durs[s.id])
            return True

        await run_dag(steps, k, run_step, stop_on_failure=False)
        return clock.now_ms()

    return run_virtual(main), started


def test_list_scheduling_respects_grahams_bound_and_meets_the_critical_path_at_full_width():
    rng = stream(42, "dags")
    for trial in range(60):
        steps, durs = random_dag(rng, 4 + below(rng, 14))
        cp = critical_path(steps, durs)
        w = sum(durs.values())
        for k in (1, 2, 3, 4, 8):
            makespan, started = schedule(steps, durs, k)
            assert sorted(started) == sorted(s.id for s in steps) and len(started) == len(set(started))
            assert makespan <= w / k + (1 - 1 / k) * cp + 1e-9, (trial, k)
            assert makespan >= max(cp, w / k) - 1e-9
        # k at least the width of the DAG: the makespan is the critical path
        makespan, _ = schedule(steps, durs, len(steps))
        assert makespan == cp
    layered = [
        PlanStep("a", "x"),
        PlanStep("b", "x", {}, ("a",)),
        PlanStep("c", "x", {}, ("a",)),
        PlanStep("d", "x", {}, ("b", "c")),
    ]
    assert width(layered) == 2
    assert schedule(layered, {"a": 100, "b": 300, "c": 200, "d": 100}, 2)[0] == 500


def test_scheduler_prefers_the_longest_remaining_path_and_never_repeats():
    steps = [
        PlanStep("a", "x"),
        PlanStep("b", "x"),
        PlanStep("c", "x", {}, ("b",)),
        PlanStep("d", "x", {}, ("c",)),
    ]
    _, started = schedule(steps, {"a": 100, "b": 100, "c": 100, "d": 100}, 1)
    # b first (longest remaining path), then c; a and d tie on remaining length -> step id order
    assert started == ["b", "c", "a", "d"]
    steps2 = [PlanStep("a", "x"), PlanStep("b", "x", {}, ("a",)), PlanStep("c", "x")]
    seen = []

    async def main(clock):
        async def run_step(s):
            seen.append(s.id)
            await clock.sleep(10)
            return s.id != "a"

        return await run_dag(steps2, 2, run_step, stop_on_failure=False)

    results, skipped = run_virtual(main)
    assert seen == ["a", "c"] and skipped == ["b"] and results == {"a": False, "c": True}


class ReportIn(BaseModel):
    to: str


def write_world():
    writes = []

    async def send(v):
        writes.append(v.to)
        return {"sent": len(writes)}

    state = {"fail": 1}

    class GetIn(BaseModel):
        key: str

    async def get(v):
        if v.key == "flaky" and state["fail"] > 0:
            state["fail"] -= 1
            raise ToolError("NOT_FOUND", "flaky key")
        return {"key": v.key}

    tools = [
        Tool(name="send_report", handler=send, input_model=ReportIn, side_effect="write"),
        Tool(name="get", handler=get, input_model=GetIn, side_effect="read"),
    ]
    return tools, writes


def plan_text(*steps):
    return canonical(plan(*steps))


def test_a_replan_after_a_failed_step_completes_the_run_and_never_repeats_the_write():
    tools, writes = write_world()
    script = [
        {
            "text": plan_text(
                ("s1", "send_report", {"to": "x"}, []),
                ("s2", "get", {"key": "flaky"}, []),
                ("s3", "get", {"key": "z"}, ["s2"]),
            ),
            "usage": u(),
        },
        # the replanner returns the write again (already done) and the failed step
        {
            "text": plan_text(
                ("s1", "send_report", {"to": "x"}, []),
                ("s2b", "get", {"key": "flaky"}, []),
                ("s3", "get", {"key": "z"}, ["s2b"]),
            ),
            "usage": u(),
        },
        {"text": "answer", "usage": u()},
    ]
    res = run_scripted(ScriptedEnv({"p": script}, tools=lambda: tools), "plan_replan", {"task": "t"})
    assert (res.status, res.text, res.meter.replans) == ("succeeded", "answer", 1)
    assert writes == ["x"]  # the completed write was not run again
    # s2 starts first (longer remaining path), s1 alongside; the replan reruns only the failed branch
    assert [t["attrs"]["gen_ai.tool.call.id"] for t in kinds(res, "tool")] == [
        "p1#s2",
        "p1#s1",
        "p2#s2b",
        "p2#s3",
    ]


def test_a_naive_replanner_that_repeats_a_write_is_caught_by_the_worlds_duplicate_detector():
    from agent_runtime.evals.simrun import Job, task_row
    from agent_runtime.sim.tasks import gen

    task = next(t for t in gen(3, 200) if t.report)
    report = next(s for s in task.steps if s.tool == "send_report")
    ws = WorldSession(make_world(1), None, 3, "r")

    async def naive(clock):
        ws.clock = clock
        tool = next(t for t in ws.tools() if t.name == "send_report")
        v = tool.input_model.model_validate(report.args)
        await tool.handler(v)
        await tool.handler(v)  # a replanner without the completed-step guard runs the write again

    run_virtual(naive)

    class FakeRes:
        text = task.answer
        accepted = True
        status = "succeeded"
        failure_reason = None
        latency_ms = 0

        class meter:  # noqa: N801
            cost_ucu = model_calls = tool_calls = provider_calls = retries = fallbacks = repairs = 0
            replans = escalations = peak_context_tokens = compactions = repeat_reads = 0
            cache_read_tokens = input_tokens = subagent_runs = peak_parent_context_tokens = 0
            by_model: dict = {}

        class trace:  # noqa: N801
            @staticmethod
            def jsonl():
                return ""

    row = task_row(Job("t", "x", 3), task, FakeRes(), ws, None)
    assert row["duplicate_writes"] == 1 and row["success"] == 0
