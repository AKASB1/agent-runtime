"""The part of the input a policy may not drop decides whether a target is offered (found in E5:
summaries and stubs that no policy removes must count, or a step fails with context_length instead
of moving to the larger window)."""

import statistics

from agent_runtime.context.derive import SimCounter, derive_messages
from agent_runtime.evals.simrun import Job, run_job
from test_context import make_log

C = SimCounter(3000)


def test_the_floor_of_each_policy():
    events = make_log(3)  # 500 + 3 x 3000; the last exchange is in progress
    d = {
        p: derive_messages(events, p, 100_000, C).irreducible
        for p in ("none", "window", "elide", "summarize")
    }
    assert d["none"] == 500 + 3 * 3000
    assert d["window"] == 500 + 3000
    assert d["summarize"] == 500 + 3000 + 100
    assert d["elide"] == 500 + 3000 + 2 * 20


def test_a_step_moves_to_the_larger_window_instead_of_failing():
    rows = {}
    for policy in ("summarize", "elide", "window"):
        job = Job(
            "t",
            policy,
            2,
            n_tasks=40,
            result_tokens=3000,
            router={"type": "fixed", "target": "sim-a/small", "fallback": ["sim-a/medium"]},
            run_config={"context": {"policy": policy, "keep_last": 1, "compact_to": 0.4}},
            summarizer="sim-a/small",
        )
        rows[policy] = run_job(job)["rows"]
    for policy, rs in rows.items():
        assert sum(r["context_length_failures"] for r in rs) == 0, policy
    assert statistics.mean(r["medium_calls"] for r in rows["summarize"]) > 0
