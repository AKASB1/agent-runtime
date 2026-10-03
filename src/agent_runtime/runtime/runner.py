"""Building and executing runs from a serialisable ``RunSpec``, and replaying them from a trace.

The run span records the spec (``agent_runtime.config``), so a replay needs only the trace, an
environment factory (the same registry and tool list, with stubs that fail if a real provider or
tool is called), and, for a session run, the session's log before the run (``start_seq``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

import agent_runtime.workflow.workflows  # noqa: F401  (registers the workflows)
from agent_runtime.clock import run_virtual
from agent_runtime.routing.routers import make_router
from agent_runtime.runtime.engine import Env, Run, RunConfig, RunResult
from agent_runtime.runtime.replay import ReplayDivergence, Replayer, compare_spans
from agent_runtime.store.memory import MemoryStore
from agent_runtime.telemetry.trace import TraceInvalid, validate_trace


class RunSpec(BaseModel):
    run_id: str
    workflow: str
    input: dict[str, Any] = Field(default_factory=dict)
    config: RunConfig = Field(default_factory=RunConfig)
    router: dict[str, Any] = Field(default_factory=dict)
    default_target: str = ""
    task_id: str | None = None
    hint: float | None = None
    env: dict[str, Any] = Field(default_factory=dict)  # name and parameters for the env factory
    start_seq: int = 0
    clock: str = "virtual"


async def execute(
    spec: RunSpec,
    env: Env,
    clock: Any,
    store: Any,
    session_id: str,
    *,
    replayer: Replayer | None = None,
    on_start: Callable[[Run], None] | None = None,
) -> RunResult:
    router = make_router(spec.router, seed=spec.config.seed, default_target=spec.default_target)
    run = Run(
        run_id=spec.run_id,
        env=env,
        config=spec.config,
        clock=clock,
        store=store,
        session_id=session_id,
        workflow=spec.workflow,
        input=spec.input,
        router=router,
        task_id=spec.task_id,
        hint=spec.hint,
        replayer=replayer,
        record_config=spec.model_dump(mode="json"),
    )
    if on_start is not None:
        on_start(run)
    return await run.execute()


class GuardAdapter:
    """Stands in for every provider during replay; a call is a bug."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: Any, spec: Any) -> Any:
        self.calls += 1
        raise AssertionError("replay called a provider")


def replay_spans(
    spans: list[dict[str, Any]],
    env_factory: Callable[[RunSpec], Env],
    *,
    prior_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay a recorded trace on a virtual clock. Returns ``{"divergence": None | {...}, ...}``."""
    run_span = spans[0]
    spec = RunSpec.model_validate(run_span["attrs"]["agent_runtime.config"])
    replayer = Replayer(spans)
    env = env_factory(spec)
    guard = GuardAdapter()
    env.adapters = {p: guard for p in env.adapters}
    tool_calls = {"n": 0}
    for t in env.tools.tools():
        orig = t.handler

        async def guarded(value: Any, _orig: Any = orig) -> Any:
            tool_calls["n"] += 1
            raise AssertionError("replay called a tool")

        t.handler = guarded
    store = MemoryStore()
    sid = store.create_session({"replay_of": run_span["trace_id"]})
    for ev in prior_events or []:
        store.append_event(sid, {k: v for k, v in ev.items() if k not in ("seq", "id")})

    async def main(clock: Any) -> RunResult:
        return await execute(spec, env, clock, store, sid, replayer=replayer)

    divergence = None
    result = None
    try:
        result = run_virtual(main)
        replayed = result.trace.dicts()
        compare_spans(spans, replayed, compare_times=spec.clock == "virtual")
        if result.trace.dicts()[0]["attrs"].get("agent_runtime.result_sha256") != run_span["attrs"].get(
            "agent_runtime.result_sha256"
        ):
            raise ReplayDivergence(run_span["span_id"], "result differs")
        validate_trace(replayed)
    except ReplayDivergence as e:
        divergence = e.report()
    except TraceInvalid as e:
        divergence = {"span_id": None, "detail": f"the replayed trace is invalid: {e}"}
    return {
        "divergence": divergence,
        "provider_calls": guard.calls,
        "tool_calls": tool_calls["n"],
        "replayed_model_calls": replayer.replayed_model_calls,
        "replayed_tool_calls": replayer.replayed_tool_calls,
        "status": result.status if result else None,
    }
