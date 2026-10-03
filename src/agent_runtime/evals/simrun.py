"""Running simulated tasks through the runtime: one job = (experiment, configuration, seed) on one
virtual-time loop. Tasks run one after another (E1, E3) or as Poisson arrivals with a cap on runs in
flight (E2). A task's calls, outcomes, and own latency depend only on (seed, task, configuration)."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.agents import delegate_tool
from agent_runtime.canon import digest
from agent_runtime.clock import run_virtual, wall_ms
from agent_runtime.context.derive import SimCounter
from agent_runtime.models.registry import ModelRegistry
from agent_runtime.planning.plan import PlanStep, critical_path
from agent_runtime.rand import stream
from agent_runtime.runtime.engine import Env, RunConfig, RunResult
from agent_runtime.runtime.runner import RunSpec, execute
from agent_runtime.sim.model import FAULT_PROFILES, SimProvider, sim_specs
from agent_runtime.sim.tasks import Task, gen
from agent_runtime.sim.world import WorldSession, make_world
from agent_runtime.store.memory import MemoryStore
from agent_runtime.store.sqlite import SqliteStore
from agent_runtime.tools import ToolRegistry

TASK_COLUMNS = [
    "experiment", "config", "seed", "task_id", "tier", "hint", "success", "accepted", "status", "failure_reason",
    "cost_ucu", "latency_ms", "model_calls", "tool_calls", "provider_calls", "retries", "fallbacks", "repairs",
    "replans", "escalations", "peak_context_tokens", "peak_parent_context_tokens", "compactions", "repeat_reads", "cache_read_tokens",
    "input_tokens", "subagent_runs", "medium_calls", "context_length_failures", "duplicate_writes", "wrong_writes", "bound_gap", "arrival_ms", "trace_sha256",
]  # fmt: skip


@dataclass
class Job:
    experiment: str
    config: str
    seed: int
    workflow: str = "react"
    router: dict[str, Any] = field(default_factory=lambda: {"type": "fixed", "target": "sim-a/medium"})
    run_config: dict[str, Any] = field(default_factory=dict)
    n_tasks: int = 100
    mix: str = "mixed"
    shape: str = "chain"
    hint_sigma: float = 0.6
    result_tokens: int = 250
    recall: float = 0.7
    keep: float = 0.9
    cache: bool = False
    tool_fail_rate: float = 0.0
    providers: tuple[str, ...] = ("sim-a",)
    faults: dict[str, str] = field(default_factory=dict)  # provider -> profile name
    arrivals: dict[str, Any] | None = None  # {"rate_per_s", "duration_s", "max_in_flight"}
    keep_traces: bool = False
    store_path: str | None = None  # SQLite file for the session logs (tests); memory otherwise
    summarizer: str | None = (
        None  # the target that writes summaries (policy summarize); default: the call's target
    )
    keep_delegate: float = 0.95  # E6: probability that a child's final text carries its branch's results


def run_job(job: Job) -> dict[str, Any]:
    """Run one job; returns ``{"rows": [...], "traces": {...}, "wall_ms": ...}`` (picklable)."""
    t0 = wall_ms()
    rows, traces = run_virtual(lambda clock: _run_job(job, clock))
    return {"job": job, "rows": rows, "traces": traces, "wall_ms": round(wall_ms() - t0, 1)}


def _tasks_for(job: Job) -> tuple[list[Task], list[int] | None]:
    if job.arrivals is None:
        return gen(job.seed, job.n_tasks, job.mix, job.shape, hint_sigma=job.hint_sigma), None
    a = job.arrivals
    rng = stream(job.seed, "arrivals")
    times, t = [], 0.0
    while True:
        t += -math.log(1.0 - rng.random()) / a["rate_per_s"]
        if t >= a["duration_s"]:
            break
        times.append(int(round(t * 1000)))
    return gen(job.seed, len(times), job.mix, job.shape, hint_sigma=job.hint_sigma), times


async def _run_job(job: Job, clock: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tasks, arrivals = _tasks_for(job)
    by_id = {t.id: t for t in tasks}
    world = make_world(1)
    providers = {
        p: SimProvider(
            p,
            by_id,
            seed=job.seed,
            clock=clock,
            result_tokens=job.result_tokens,
            faults=FAULT_PROFILES[job.faults.get(p, "none")],
            keep=job.keep,
            cache=job.cache,
            keep_delegate=job.keep_delegate,
        )
        for p in job.providers
    }
    specs = sim_specs(job.providers)
    store = SqliteStore(job.store_path) if job.store_path else MemoryStore()
    rows: list[dict[str, Any]] = []
    traces: dict[str, str] = {}

    async def one(task: Task, arrival: int | None) -> None:
        run_id = f"r-{job.experiment}.{job.config}.{job.seed}.{task.id}"
        ws = WorldSession(world, clock, job.seed, run_id, fail_rate=job.tool_fail_rate)
        agents: dict[str, Any] = {}
        env = Env(
            registry=ModelRegistry(list(specs)),
            adapters=dict(providers),
            tools=ToolRegistry(
                ws.tools() + ([delegate_tool(agents)] if job.workflow == "delegate_plan" else [])
            ),
            agents=agents,
            counters={"*": SimCounter(job.result_tokens)},
            verifier=_verifier(task, job),
            system_prompt="You are an agent with tools for customers, orders, products, currencies, and reports.",
            extra={"summarizer": job.summarizer} if job.summarizer else {},
        )
        spec = RunSpec(
            run_id=run_id,
            workflow=job.workflow,
            input={"task": task.text},
            config=RunConfig(seed=job.seed, **job.run_config),
            router=job.router,
            default_target="sim-a/medium",
            task_id=task.id,
            hint=task.hint,
            env={"name": "sim", "job": job.experiment},
        )
        sid = store.create_session({"task": task.id})
        res = await execute(spec, env, clock, store, sid)
        row = task_row(job, task, res, ws, arrival)
        rows.append(row)
        if job.keep_traces:
            traces[run_id] = res.trace.jsonl()

    if arrivals is None:
        for task in tasks:
            await one(task, None)
    else:
        sem = asyncio.Semaphore(job.arrivals["max_in_flight"])

        async def arrive(task: Task, at: int) -> None:
            await clock.sleep(at - clock.now_ms())
            async with sem:
                await one(task, at)

        await asyncio.gather(*(arrive(t, at) for t, at in zip(tasks, arrivals, strict=True)))
    store.close()
    rows.sort(key=lambda r: r["task_id"])
    return rows, traces


def sim_env_factory(job: Job):
    """An env factory for replaying the runs of ``job``: same registry, tools, counters, verifier."""
    tasks, _ = _tasks_for(job)
    by_id = {t.id: t for t in tasks}
    world = make_world(1)

    def factory(spec: RunSpec) -> Env:
        task = by_id[spec.task_id]
        ws = WorldSession(world, None, job.seed, spec.run_id, fail_rate=job.tool_fail_rate)
        agents: dict[str, Any] = {}
        return Env(
            registry=ModelRegistry(sim_specs(job.providers)),
            adapters={p: None for p in job.providers},
            tools=ToolRegistry(
                ws.tools() + ([delegate_tool(agents)] if job.workflow == "delegate_plan" else [])
            ),
            agents=agents,
            counters={"*": SimCounter(job.result_tokens)},
            verifier=_verifier(task, job),
            system_prompt="You are an agent with tools for customers, orders, products, currencies, and reports.",
            extra={"summarizer": job.summarizer} if job.summarizer else {},
        )

    return factory


def _verifier(task: Task, job: Job):
    def verify(text: str, attempt: int) -> bool:
        if text == task.answer:
            return True
        return stream(job.seed, f"verifier/{task.id}/{attempt}").random() >= job.recall

    return verify


def bound_gap(res: RunResult, k: int) -> float | None:
    """Tool-phase duration / max(CP, W/k) of the executed plan (plan modes only, else NaN)."""
    spans = res.trace.dicts()
    plans = [s for s in spans if s["kind"] == "plan"]
    tools = [s for s in spans if s["kind"] == "tool" and s["attrs"].get("agent_runtime.tool.side_effect")]
    if not plans or not tools:
        return None
    dur: dict[str, float] = {}
    deps: dict[str, tuple[str, ...]] = {}
    for p in plans:
        for st in p["attrs"]["agent_runtime.plan.steps"]:
            deps[st["id"]] = tuple(st["depends_on"])
    for t in tools:
        sid = t["attrs"]["gen_ai.tool.call.id"].split("#", 1)[1]
        dur[sid] = dur.get(sid, 0) + (t["end_ms"] - t["start_ms"])
    steps = [PlanStep(i, "x", {}, tuple(d for d in deps.get(i, ()) if d in dur)) for i in sorted(dur)]
    cp = critical_path(steps, dur)
    w = sum(dur.values())
    phase = max(t["end_ms"] for t in tools) - min(t["start_ms"] for t in tools)
    denom = max(cp, w / k)
    return round(phase / denom, 6) if denom > 0 else None


def task_row(job: Job, task: Task, res: RunResult, ws: WorldSession, arrival: int | None) -> dict[str, Any]:
    gold_write = next((s.args for s in task.steps if s.tool == "send_report"), None)
    seen: list[str] = []
    dup = 0
    for w in ws.writes:
        key = digest(w)
        if key in seen:
            dup += 1
        seen.append(key)
    wrong = sum(1 for w in ws.writes if gold_write is None or w != gold_write)
    success = res.text == task.answer and dup == 0 and wrong == 0
    m = res.meter
    k = int(job.run_config.get("concurrency", 4))
    return {
        "experiment": job.experiment,
        "config": job.config,
        "seed": job.seed,
        "task_id": task.id,
        "tier": task.tier,
        "hint": task.hint,
        "success": int(success),
        "accepted": int(res.accepted),
        "status": res.status,
        "failure_reason": res.failure_reason or "",
        "cost_ucu": m.cost_ucu,
        "latency_ms": res.latency_ms,
        "model_calls": m.model_calls,
        "tool_calls": m.tool_calls,
        "provider_calls": m.provider_calls,
        "retries": m.retries,
        "fallbacks": m.fallbacks,
        "repairs": m.repairs,
        "replans": m.replans,
        "escalations": m.escalations,
        "peak_context_tokens": m.peak_context_tokens,
        "peak_parent_context_tokens": m.peak_parent_context_tokens,
        "compactions": m.compactions,
        "repeat_reads": m.repeat_reads,
        "cache_read_tokens": m.cache_read_tokens,
        "input_tokens": m.input_tokens + m.cache_read_tokens,
        "subagent_runs": m.subagent_runs,
        "medium_calls": m.by_model.get("sim-a/medium", {}).get("calls", 0),
        "context_length_failures": int(res.failure_reason == "context_length"),
        "duplicate_writes": dup,
        "wrong_writes": wrong,
        "bound_gap": bound_gap(res, k) if job.workflow.startswith("plan") else None,
        "arrival_ms": arrival if arrival is not None else -1,
        "trace_sha256": digest(res.trace.jsonl()),
    }
