"""Plan validation and list scheduling.

A plan is a list of steps (``id``, ``tool``, ``args``, ``depends_on``), validated before it runs:
acyclic, tools exist (and are allowed), arguments valid, dependencies resolve, at most 20 steps. An
invalid plan is an ``invalid_output`` and goes through the repair budget. Arguments are literal
values (there are no references to earlier results in this plan format).

The scheduler is list scheduling: whenever fewer than ``k`` steps run, it starts the ready step
with the longest remaining path (in steps; ties by step id). It never repeats a completed step.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.models.structured import StructuredOutputError

MAX_PLAN_STEPS = 20
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": MAX_PLAN_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "tool", "args", "depends_on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PlanStep:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


def validate_plan(
    value: Any, tools: Any, *, allow: list[str] | None = None, completed: set[str] | None = None
) -> list[PlanStep]:
    """Raise ``StructuredOutputError`` (class of the problem first) or return the steps."""
    completed = completed or set()
    raw = value.get("steps") if isinstance(value, dict) else None
    if not isinstance(raw, list):
        raise StructuredOutputError("plan_invalid: no steps list")
    if not raw:
        raise StructuredOutputError("plan_invalid: empty plan")
    if len(raw) > MAX_PLAN_STEPS:
        raise StructuredOutputError(f"plan_too_long: {len(raw)} steps (at most {MAX_PLAN_STEPS})")
    steps: list[PlanStep] = []
    ids: set[str] = set()
    for s in raw:
        sid = s["id"]
        if sid in ids:
            raise StructuredOutputError(f"duplicate_step: {sid}")
        ids.add(sid)
        tool = tools.get(s["tool"]) if (allow is None or s["tool"] in allow) else None
        if tool is None:
            raise StructuredOutputError(f"unknown_tool: step {sid} names {s['tool']!r}")
        _, err = tool.validate_input(s["args"])
        if err is not None:
            raise StructuredOutputError(f"invalid_args: step {sid}: {err}")
        steps.append(PlanStep(sid, s["tool"], dict(s["args"]), tuple(s["depends_on"])))
    for st in steps:
        for d in st.depends_on:
            if d not in ids and d not in completed:
                raise StructuredOutputError(f"unresolved_dependency: step {st.id} depends on {d}")
            if d == st.id:
                raise StructuredOutputError(f"cycle: step {st.id} depends on itself")
    _topo(steps)
    return steps


def _topo(steps: list[PlanStep]) -> list[str]:
    ids = {s.id for s in steps}
    indeg = {s.id: sum(1 for d in s.depends_on if d in ids) for s in steps}
    children: dict[str, list[str]] = {s.id: [] for s in steps}
    for s in steps:
        for d in s.depends_on:
            if d in ids:
                children[d].append(s.id)
    ready = sorted(i for i, n in indeg.items() if n == 0)
    order = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for c in sorted(children[i]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort()
    if len(order) != len(steps):
        raise StructuredOutputError("cycle: the plan's dependencies are cyclic")
    return order


def longest_remaining(steps: list[PlanStep], weights: dict[str, float] | None = None) -> dict[str, float]:
    """Length of the longest path from each step to a sink, the step included."""
    ids = {s.id for s in steps}
    children: dict[str, list[str]] = {s.id: [] for s in steps}
    for s in steps:
        for d in s.depends_on:
            if d in ids:
                children[d].append(s.id)
    w = weights or {}
    memo: dict[str, float] = {}
    for sid in reversed(_topo(steps)):
        memo[sid] = w.get(sid, 1.0) + max((memo[c] for c in children[sid]), default=0.0)
    return memo


def plan_tail(steps: list[Any]) -> list[str]:
    """The join step (the first step, in plan order, with two or more dependencies) and every step
    that depends on it, directly or not; empty for a plan without a join step."""
    join = next((s.id for s in steps if len(s.depends_on) >= 2), None)
    if join is None:
        return []
    tail = {join}
    changed = True
    while changed:
        changed = False
        for s in steps:
            if s.id not in tail and any(d in tail for d in s.depends_on):
                tail.add(s.id)
                changed = True
    return [s.id for s in steps if s.id in tail]


def plan_branches(steps: list[Any]) -> list[list[str]]:
    """The connected components of the plan without its tail, in the order of their first step."""
    tail = set(plan_tail(steps))
    rest = [s for s in steps if s.id not in tail]
    ids = {s.id for s in rest}
    root = {s.id: s.id for s in rest}

    def find(x: str) -> str:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    order = {s.id: i for i, s in enumerate(steps)}
    for s in rest:
        for d in s.depends_on:
            if d in ids:
                a, b = find(s.id), find(d)
                if a != b:
                    keep, drop = (a, b) if order[a] < order[b] else (b, a)
                    root[drop] = keep
    groups: dict[str, list[str]] = {}
    for s in rest:
        groups.setdefault(find(s.id), []).append(s.id)
    return sorted(groups.values(), key=lambda g: order[g[0]])


def critical_path(steps: list[PlanStep], durations: dict[str, float]) -> float:
    lr = longest_remaining(steps, durations)
    return max(lr.values(), default=0.0)


def width(steps: list[PlanStep]) -> int:
    """Maximum number of steps at the same depth level (a lower bound of the antichain width that
    equals it for the layered plans the generator makes)."""
    ids = {s.id for s in steps}
    depth: dict[str, int] = {}
    for sid in _topo(steps):
        st = next(s for s in steps if s.id == sid)
        depth[sid] = 1 + max((depth[d] for d in st.depends_on if d in ids), default=0)
    counts: dict[int, int] = {}
    for d in depth.values():
        counts[d] = counts.get(d, 0) + 1
    return max(counts.values(), default=0)


async def run_dag(
    steps: list[PlanStep],
    k: int,
    run_step: Callable[[PlanStep], Awaitable[bool]],
    *,
    stop_on_failure: bool,
    done: set[str] | None = None,
) -> tuple[dict[str, bool], list[str]]:
    """List-schedule ``steps`` with at most ``k`` running. Returns (ok per started step, skipped).

    ``stop_on_failure``: after a failure no new step starts (in-flight steps finish); otherwise the
    dependents of a failed step are skipped and everything else runs. ``done`` are steps completed
    before (by an earlier plan); they count as finished dependencies.
    """
    done = set(done or ())
    ids = {s.id for s in steps}
    prio = longest_remaining(steps)
    by_id = {s.id: s for s in steps}
    results: dict[str, bool] = {}
    failed: set[str] = set()
    skipped: list[str] = []
    pending = set(ids)
    running: dict[asyncio.Task, str] = {}
    stop = False

    def blocked(s: PlanStep) -> bool:
        return any(d in failed or d in skipped for d in s.depends_on)

    try:
        while pending or running:
            if not stop:
                for sid in sorted(pending):
                    s = by_id[sid]
                    if blocked(s):
                        pending.discard(sid)
                        skipped.append(sid)
                changed = True
                while changed:
                    changed = False
                    for sid in sorted(pending):
                        if blocked(by_id[sid]):
                            pending.discard(sid)
                            skipped.append(sid)
                            changed = True
                ready = [
                    sid
                    for sid in pending
                    if all(
                        (d in results and results[d]) or (d in done and d not in ids)
                        for d in by_id[sid].depends_on
                    )
                ]
                ready.sort(key=lambda i: (-prio[i], i))
                for sid in ready:
                    if len(running) >= k:
                        break
                    pending.discard(sid)
                    running[asyncio.ensure_future(run_step(by_id[sid]))] = sid
            if not running:
                if pending and not stop:
                    # nothing can start: the remaining steps wait on failed or skipped ones
                    skipped.extend(sorted(pending))
                    pending.clear()
                break
            finished, _ = await asyncio.wait(list(running), return_when=asyncio.FIRST_COMPLETED)
            for t in sorted(finished, key=lambda t: running[t]):
                sid = running.pop(t)
                ok = t.result()
                results[sid] = ok
                if not ok:
                    failed.add(sid)
                    if stop_on_failure:
                        stop = True
            if stop and not running:
                skipped.extend(sorted(pending))
                pending.clear()
    except BaseException:
        for t in running:
            t.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    return results, skipped
