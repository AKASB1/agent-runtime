"""Shared test helpers: a scripted environment and a one-call runner on the virtual clock."""

from __future__ import annotations

from typing import Any

from agent_runtime.clock import run_virtual
from agent_runtime.mcp.twin import native_lookup_tool
from agent_runtime.models import ModelRegistry, ModelSpec
from agent_runtime.providers.scripted import ScriptedModel
from agent_runtime.runtime.engine import Env, RunConfig
from agent_runtime.runtime.runner import RunSpec, execute, replay_spans
from agent_runtime.store.memory import MemoryStore
from agent_runtime.telemetry.trace import validate_trace
from agent_runtime.tools import ToolRegistry


def spec(id: str = "m", provider: str = "p", window: int = 8000, **kw: Any) -> ModelSpec:
    base = dict(
        id=id,
        provider=provider,
        wire="scripted",
        price_in_ucu=1_000_000,
        price_out_ucu=4_000_000,
        context_tokens=window,
    )
    base.update(kw)
    return ModelSpec(**base)


def call(cid: str, name: str = "lookup", **args: Any) -> dict:
    return {"id": cid, "name": name, "arguments": args}


def u(i: int = 100, o: int = 10) -> dict:
    return {"input_tokens": i, "output_tokens": o}


class ScriptedEnv:
    """Build an Env from per-provider scripts; the same object rebuilds it for replay."""

    def __init__(
        self,
        scripts: dict[str, list],
        specs: list[ModelSpec] | None = None,
        tools: list | None = None,
        latency_ms: int | list[int] = 100,
        **env_kw: Any,
    ) -> None:
        self.scripts = scripts
        self.specs = specs or [spec(provider=p) for p in scripts]
        self.tools = tools
        self.latency_ms = latency_ms
        self.env_kw = env_kw
        self.models: dict[str, ScriptedModel] = {}

    def __call__(self, run_spec: RunSpec | None = None, clock: Any = None) -> Env:
        models = {
            p: ScriptedModel(list(s), clock=clock, latency_ms=self.latency_ms)
            for p, s in self.scripts.items()
        }
        if clock is not None or not self.models:  # a replay (no clock) keeps the live run's models
            self.models = models
        tools = (
            self.tools()
            if callable(self.tools)
            else (self.tools if self.tools is not None else [native_lookup_tool()])
        )
        return Env(
            registry=ModelRegistry(list(self.specs)),
            adapters=dict(models),
            tools=ToolRegistry(list(tools)),
            **self.env_kw,
        )


def run_scripted(
    senv: ScriptedEnv,
    workflow: str = "react",
    inp: dict | None = None,
    *,
    router: dict | None = None,
    config: RunConfig | None = None,
    default_target: str | None = None,
    store: Any = None,
    session_id: str | None = None,
    on_start=None,
    check_replay: bool = True,
    run_id: str = "r-1",
):  # noqa: ANN001
    store = store or MemoryStore()
    sid = session_id or store.create_session()
    rs = RunSpec(
        run_id=run_id,
        workflow=workflow,
        input=inp or {"task": "what is beta?"},
        config=config or RunConfig(),
        router=router or {},
        default_target=default_target or senv.specs[0].target,
    )
    holder: dict[str, Any] = {}

    async def main(clock: Any):
        env = senv(rs, clock)
        holder["env"] = env
        return await execute(rs, env, clock, store, sid, on_start=on_start)

    res = run_virtual(main)
    spans = res.trace.dicts()
    validate_trace(spans)
    if check_replay and res.status != "cancelled":
        rep = replay_spans(spans, lambda s: senv(s, None))
        assert rep["divergence"] is None, rep
        assert rep["provider_calls"] == 0 and rep["tool_calls"] == 0
    res.extra["env"] = holder["env"]
    res.extra["store"] = store
    res.extra["session_id"] = sid
    return res


def kinds(res: Any, kind: str) -> list[dict]:
    return [s for s in res.trace.dicts() if s["kind"] == kind]
