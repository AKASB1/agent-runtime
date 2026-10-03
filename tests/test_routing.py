"""Routers on hand-made contexts, and a test that a router's context carries no hidden information."""

import dataclasses

from agent_runtime.evals.simrun import Job, run_job
from agent_runtime.routing import (
    AttemptOutcome,
    CascadeRouter,
    FixedRouter,
    RandomMixRouter,
    RouteContext,
    ThresholdRouter,
    make_router,
)
from agent_runtime.runtime import engine
from agent_runtime.sim.tasks import gen

S, MD, L = "sim-a/small", "sim-a/medium", "sim-a/large"
SPECS = {
    S: {"latency_base_ms": 200, "latency_per_token_ms": 8},
    MD: {"latency_base_ms": 500, "latency_per_token_ms": 15},
    L: {"latency_base_ms": 1200, "latency_per_token_ms": 30},
}


def ctx(**kw):
    base = dict(
        step_kind="act",
        candidates=(S, MD, L),
        spent_cost_ucu=0,
        elapsed_ms=0,
        calls=0,
        attempt=1,
        history=(),
        hint=1.0,
        task_id="t0001",
        deadline_ms=30_000,
        specs=SPECS,
    )
    base.update(kw)
    return RouteContext(**base)


def test_fixed_router_and_its_fallback_list():
    r = FixedRouter(S, fallback=[MD])
    assert r.choose(ctx()).target == S
    assert r.choose(ctx(candidates=(MD, L))).target == MD
    assert r.choose(ctx(candidates=(L,))) is None
    assert not r.escalate(ctx())


def test_random_mix_is_a_deterministic_per_task_draw_with_the_given_probabilities():
    r = RandomMixRouter({S: 0.5, MD: 0.25, L: 0.25}, seed=1)
    picks = [r.choose(ctx(task_id=f"t{i}")).target for i in range(4000)]
    assert abs(picks.count(S) / 4000 - 0.5) < 0.03 and abs(picks.count(L) / 4000 - 0.25) < 0.03
    again = RandomMixRouter({S: 0.5, MD: 0.25, L: 0.25}, seed=1)
    assert [again.choose(ctx(task_id=f"t{i}")).target for i in range(50)] == picks[:50]
    assert all(
        r.choose(ctx(task_id="t7")).target == r.choose(ctx(task_id="t7", calls=5)).target for _ in range(3)
    )


def test_threshold_router_uses_only_the_hint():
    r = ThresholdRouter(1.5, 2.5, [S, MD, L])
    assert [r.choose(ctx(hint=h)).target for h in (0.2, 1.49, 1.5, 2.49, 2.5, 9)] == [S, S, MD, MD, L, L]


def test_cascade_escalates_along_its_chain_and_stops_when_the_deadline_cannot_be_met():
    r = CascadeRouter([S, MD, L])
    assert r.choose(ctx(attempt=1)).target == S
    assert r.choose(ctx(attempt=2)).target == MD
    assert r.choose(ctx(attempt=3)).target == L
    h = (AttemptOutcome(1, S, "succeeded", False, 4),)
    assert r.escalate(ctx(attempt=1, history=h, elapsed_ms=1000))
    # medium median call = 500 + 15*100 = 2000 ms; 4 calls = 8000 ms: 23000 + 8000 > 30000
    assert not r.escalate(ctx(attempt=1, history=h, elapsed_ms=23_000))
    assert not r.escalate(ctx(attempt=3, history=h))
    assert CascadeRouter([S, L], budget_aware=False).escalate(ctx(attempt=1, history=h, elapsed_ms=29_999))


def test_make_router_builds_each_type():
    assert make_router({"type": "fixed", "target": S}, seed=1, default_target=MD).name == "fixed"
    assert make_router({}, seed=1, default_target=MD).choose(ctx()).target == MD
    assert make_router({"type": "cascade", "chain": [S, L]}, seed=1, default_target=MD).name == "cascade"


def test_a_router_never_sees_the_true_tier_or_an_outcome_that_has_not_happened(monkeypatch):
    allowed = {
        "step_kind",
        "candidates",
        "spent_cost_ucu",
        "elapsed_ms",
        "calls",
        "attempt",
        "history",
        "hint",
        "task_id",
        "deadline_ms",
        "specs",
    }
    assert {f.name for f in dataclasses.fields(RouteContext)} == allowed
    seen: list[RouteContext] = []
    orig = engine.Run.route_context

    def spy(self, role, cands):
        c = orig(self, role, cands)
        seen.append(c)
        return c

    monkeypatch.setattr(engine.Run, "route_context", spy)
    job = Job(
        "t",
        "spy",
        2,
        n_tasks=40,
        router={"type": "cascade", "chain": [S, MD, L]},
        run_config={"budget": {"deadline_ms": 30000}},
    )
    rows = run_job(job)["rows"]
    tasks = {t.id: t for t in gen(2, 40)}
    assert seen
    for c in seen:
        t = tasks[c.task_id]
        assert c.hint == t.hint and c.hint != t.tier_index  # the noisy hint, never the tier
        for spec in c.specs.values():
            assert set(spec) == {
                "price_in_ucu",
                "price_out_ucu",
                "context_tokens",
                "latency_base_ms",
                "latency_per_token_ms",
            }
        # history holds only finished attempts, and only what the runtime knew (accepted), not success
        assert all(h.attempt < c.attempt or c.step_kind == "escalate" for h in c.history)
        for h in c.history:
            assert set(dataclasses.asdict(h)) == {"attempt", "target", "status", "accepted", "model_calls"}
    assert any(r["escalations"] for r in rows)
