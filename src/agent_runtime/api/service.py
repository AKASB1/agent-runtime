"""The service behind the HTTP API: builds environments for runs, executes them in background tasks
on the server's event loop, persists sessions, runs, and traces in SQLite, and keeps usage totals.

The models behind the service are the simulated ones (on the system clock, optionally scaled), the
scripted targets (``local/echo``, ``scripted/demo-script`` for agent runs, ``scripted/opt-script`` for
optimize runs), and, by configuration, an OpenAI-compatible base URL exposed as ``openai/<model>``.
Logs are structured JSON without payloads or secrets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import agent_runtime.optimization.workflow  # noqa: F401  (registers optimize)
import agent_runtime.workflow.workflows  # noqa: F401  (registers the other workflows)
from agent_runtime.agents import AgentSpec, delegate_tool
from agent_runtime.clock import SystemClock
from agent_runtime.context.derive import POLICIES, EstimatorCounter, SimCounter, derive_messages
from agent_runtime.models import ModelRegistry, ModelSpec
from agent_runtime.optimization.e4 import OPT_SPEC, load_card, load_script, opt_server, scripted_model
from agent_runtime.permissions.policy import PermissionPolicy
from agent_runtime.providers.openai_chat import OpenAIChatAdapter, OpenAIChatConfig
from agent_runtime.providers.scripted import ScriptedModel
from agent_runtime.routing.routers import make_router  # noqa: F401
from agent_runtime.runtime.engine import WORKFLOWS, Env, Run, RunConfig, RunResult
from agent_runtime.runtime.runner import RunSpec, execute, replay_spans
from agent_runtime.sim.model import SimProvider, sim_specs
from agent_runtime.sim.tasks import gen
from agent_runtime.sim.world import WorldSession, make_world
from agent_runtime.store.sqlite import SqliteStore
from agent_runtime.telemetry.trace import load_jsonl
from agent_runtime.tools import ToolRegistry
from agent_runtime.workspace import Workspace

log = logging.getLogger("agent_runtime.api")

SANDBOX_ORDER = ["read_only", "workspace_write"]
APPROVAL_ORDER = ["never", "ask", "auto"]
# the documented workflows; delegate_plan is registered for E6 only (it needs a simulated task)
API_WORKFLOWS = ("chat", "react", "react_parallel", "plan_execute", "plan_replan", "optimize", "agent")
WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SCRIPT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ECHO = ModelSpec(
    id="echo",
    provider="local",
    wire="scripted",
    price_in_ucu=0,
    price_out_ucu=0,
    context_tokens=32_000,
    priced=False,
)
DEMO = ModelSpec(
    id="demo-script",
    provider="scripted",
    wire="scripted",
    price_in_ucu=1_000_000,
    price_out_ucu=4_000_000,
    context_tokens=32_000,
)
DEFAULT_CATALOG = {
    "lookup": AgentSpec(
        name="lookup", description="reads files and answers", tools=["fs_read", "fs_list"], max_steps=4
    )
}
DEFAULT_TARGET = {"chat": "local/echo", "optimize": OPT_SPEC.target, "agent": DEMO.target}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details or {}


class Settings(BaseModel):
    db: str = "tmp/agent-runtime.db"
    workspace_root: str = "tmp/workspaces"
    examples_dir: str = "examples"
    time_scale: float = 1.0
    max_sandbox: str = "workspace_write"
    max_approval: str = "never"
    allow_exec: list[list[str]] = Field(default_factory=list)
    allow_hosts: list[str] = Field(default_factory=list)
    openai_base_url: str | None = None
    openai_models: list[str] = Field(default_factory=lambda: ["echo"])
    openai_dialect: str = "openai"


class EchoModel:
    """``local/echo``: answers with the last user message (a test double for chat sessions)."""

    async def complete(self, request: Any, spec: Any) -> Any:
        from agent_runtime.providers.scripted import response_from_item

        last = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        return response_from_item({"text": f"echo: {last}"}, request, spec)


class Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        Path(settings.db).parent.mkdir(parents=True, exist_ok=True)
        Path(settings.workspace_root).mkdir(parents=True, exist_ok=True)
        self.store = SqliteStore(settings.db)
        self.clock = SystemClock(settings.time_scale)
        self.runs: dict[str, Run] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.usage: dict[str, dict[str, int]] = {}
        self._mcp = None
        self._mcp_lock = asyncio.Lock()
        self.openai = None
        if settings.openai_base_url:
            local = any(h in settings.openai_base_url for h in ("127.0.0.1", "localhost"))
            self.openai = OpenAIChatAdapter(
                OpenAIChatConfig(
                    base_url=settings.openai_base_url, dialect=settings.openai_dialect, trust_env=not local
                )
            )

    # -- registry and tools -------------------------------------------------------------------------
    def registry(self) -> ModelRegistry:
        specs = sim_specs(("sim-a", "sim-b")) + [ECHO, DEMO, OPT_SPEC]
        if self.openai is not None:
            specs += [
                ModelSpec(
                    id=m,
                    provider="openai",
                    wire="openai_chat",
                    dialect=self.settings.openai_dialect,
                    priced=False,
                    context_tokens=32_000,
                )
                for m in self.settings.openai_models
            ]
        return ModelRegistry(specs)

    def tool_catalog(self) -> list[dict[str, Any]]:
        ws = Workspace(self.settings.workspace_root, clock=self.clock)
        world = WorldSession(make_world(1), self.clock, 1, "catalog")
        out = []
        for group, tools in (
            ("world", world.tools()),
            ("workspace", ws.tools()),
            ("runtime", [delegate_tool(DEFAULT_CATALOG)]),
        ):
            for t in tools:
                out.append(
                    {
                        "name": t.name,
                        "group": group,
                        "description": t.description,
                        "side_effect": t.side_effect,
                        "capability": t.capability,
                        "parameters": t.schema(),
                    }
                )
        out.append(
            {
                "name": "opt.*",
                "group": "mcp",
                "description": "solve, verify, check_report of the worked example (MCP server)",
                "side_effect": "none",
                "capability": "none",
                "parameters": {},
            }
        )
        return out

    async def mcp(self):
        async with self._mcp_lock:
            if self._mcp is None:
                server = opt_server()
                await server.start()
                self._mcp = server
            return self._mcp

    # -- configuration checks --------------------------------------------------------------------------
    def policy_for(self, cfg: dict[str, Any]) -> PermissionPolicy:
        s = self.settings
        sandbox = cfg.get("sandbox", s.max_sandbox)
        approval = cfg.get("approval", s.max_approval)
        if sandbox not in SANDBOX_ORDER or approval not in APPROVAL_ORDER:
            raise ApiError(400, "BAD_REQUEST", "unknown sandbox or approval mode")
        if SANDBOX_ORDER.index(sandbox) > SANDBOX_ORDER.index(s.max_sandbox):
            raise ApiError(403, "CEILING_EXCEEDED", f"sandbox {sandbox} exceeds the ceiling {s.max_sandbox}")
        if APPROVAL_ORDER.index(approval) > APPROVAL_ORDER.index(s.max_approval):
            raise ApiError(
                403, "CEILING_EXCEEDED", f"approval {approval} exceeds the ceiling {s.max_approval}"
            )
        allow_exec = cfg.get("allow_exec", s.allow_exec)
        for p in allow_exec:
            if not any(len(c) <= len(p) and p[: len(c)] == c for c in s.allow_exec):
                raise ApiError(
                    403, "CEILING_EXCEEDED", f"allow_exec {p} is not within the service's --allow-exec"
                )
        allow_hosts = cfg.get("allow_hosts", s.allow_hosts)
        for h in allow_hosts:
            if h not in s.allow_hosts:
                raise ApiError(
                    403, "CEILING_EXCEEDED", f"allow_hosts {h} is not within the service's --allow-hosts"
                )
        return PermissionPolicy(
            sandbox=sandbox, approval=approval, allow_exec=allow_exec, allow_hosts=allow_hosts
        )

    def workspace_dir(self, name: str) -> str:
        if not isinstance(name, str) or not WORKSPACE_NAME.match(name):
            raise ApiError(
                400,
                "BAD_REQUEST",
                "workspace must be a name ([A-Za-z0-9_-], at most 64 characters), never a path",
            )
        path = os.path.join(self.settings.workspace_root, name)
        os.makedirs(path, exist_ok=True)
        return path

    # -- environments ----------------------------------------------------------------------------------
    def build_env(self, spec: RunSpec, *, live: bool) -> Env:
        wf = spec.workflow
        p = spec.env
        reg = self.registry()
        adapters: dict[str, Any] = {"local": EchoModel()}
        if self.openai is not None:
            adapters["openai"] = self.openai
        counters: dict[str, Any] = {}
        tools: list = []
        env_kw: dict[str, Any] = {}
        if wf in ("react", "react_parallel", "plan_execute", "plan_replan"):
            task = gen(p["task_seed"], p["task_index"] + 1, p.get("mix", "mixed"), p.get("shape", "chain"))[
                p["task_index"]
            ]
            sim = SimProvider("sim-a", {task.id: task}, seed=spec.config.seed, clock=self.clock)
            adapters.update(
                {
                    "sim-a": sim,
                    "sim-b": SimProvider("sim-b", {task.id: task}, seed=spec.config.seed, clock=self.clock),
                }
            )
            tools = WorldSession(make_world(1), self.clock, spec.config.seed, spec.run_id).tools()
            counters = {t: SimCounter(250) for t in reg.targets() if t.startswith("sim-")}
        elif wf == "optimize":
            adapters["scripted"] = scripted_model(load_script(self._root(), p["problem"]), self.clock)
        elif wf == "agent":
            ws = Workspace(
                self.workspace_dir(p["workspace"])
                if live
                else os.path.join(self.settings.workspace_root, p["workspace"]),
                clock=self.clock,
            )
            agents = {k: v for k, v in DEFAULT_CATALOG.items() if k in p.get("agents", list(DEFAULT_CATALOG))}
            tools = ws.tools() + ([delegate_tool(agents)] if agents else [])
            script = Path(self.settings.examples_dir) / "agent" / "scripts" / f"{p['script']}.json"
            adapters["scripted"] = ScriptedModel.from_file(script, clock=self.clock)
            env_kw.update(policy=PermissionPolicy.model_validate(p["policy"]), workspace=ws, agents=agents)
        return Env(registry=reg, adapters=adapters, tools=ToolRegistry(tools), counters=counters, **env_kw)

    def _root(self) -> Path:
        return Path(self.settings.examples_dir).resolve().parent

    async def add_mcp_tools(self, env: Env) -> None:
        server = await self.mcp()
        for t in server.tools():
            env.tools.register(t)

    def make_spec(self, run_id: str, session_id: str, body: dict[str, Any]) -> RunSpec:
        wf = body.get("workflow")
        if wf not in API_WORKFLOWS or wf not in WORKFLOWS:
            raise ApiError(400, "UNKNOWN_WORKFLOW", f"unknown workflow {wf!r}")
        inp = dict(body.get("input") or {})
        cfg = dict(body.get("config") or {})
        seed = int(cfg.get("seed", 1))
        rc: dict[str, Any] = {"seed": seed}
        for k in ("retry", "budget", "fallback", "context", "concurrency", "timeout_ms"):
            if k in cfg:
                rc[k] = cfg[k]
        try:
            config = RunConfig.model_validate(rc)
        except Exception as e:  # pydantic validation
            raise ApiError(422, "VALIDATION_ERROR", f"invalid config: {str(e).splitlines()[0]}") from None
        if config.context.policy not in POLICIES:
            raise ApiError(422, "VALIDATION_ERROR", f"unknown context policy {config.context.policy!r}")
        router = cfg.get("router") or {}
        if isinstance(router, str):
            router = {"type": "fixed", "target": router}
        env: dict[str, Any] = {"name": wf}
        task_id = None
        hint = None
        if wf in ("react", "react_parallel", "plan_execute", "plan_replan"):
            st = inp.get("sim_task") or {}
            env.update(
                task_seed=int(st.get("seed", seed)),
                task_index=int(st.get("index", 0)),
                mix=st.get("mix", "mixed"),
                shape=st.get("shape", "chain"),
            )
            task = gen(env["task_seed"], env["task_index"] + 1, env["mix"], env["shape"])[env["task_index"]]
            task_id, hint = task.id, task.hint
            inp = {"task": task.text}
            router = router or {"type": "fixed", "target": "sim-a/medium"}
        elif wf == "optimize":
            pid = inp.get("problem")
            try:
                card = load_card(self._root(), pid)
            except (OSError, TypeError):
                raise ApiError(400, "BAD_REQUEST", f"unknown problem {pid!r}") from None
            env["problem"] = pid
            inp = {"card": card}
        elif wf == "agent":
            policy = self.policy_for(cfg)
            script = cfg.get("script", "demo")
            if (
                not SCRIPT_NAME.match(str(script))
                or not (Path(self.settings.examples_dir) / "agent" / "scripts" / f"{script}.json").exists()
            ):
                raise ApiError(400, "BAD_REQUEST", f"unknown script {script!r}")
            env.update(
                workspace=cfg.get("workspace", "default"),
                script=script,
                policy=policy.model_dump(),
                agents=cfg.get("agents", list(DEFAULT_CATALOG)),
            )
            self.workspace_dir(env["workspace"])  # validates the name
        elif wf == "chat":
            if "message" not in inp:
                raise ApiError(422, "VALIDATION_ERROR", "chat needs input.message")
        default_target = (
            router.get("target")
            if isinstance(router, dict) and router.get("target")
            else DEFAULT_TARGET.get(wf, "sim-a/medium")
        )
        if default_target not in self.registry():
            raise ApiError(400, "BAD_REQUEST", f"unknown target {default_target!r}")
        return RunSpec(
            run_id=run_id,
            workflow=wf,
            input=inp,
            config=config,
            router=router,
            default_target=default_target,
            task_id=task_id,
            hint=hint,
            env=env,
            start_seq=len(self.store.events(session_id)),
            clock="system",
        )

    # -- runs ------------------------------------------------------------------------------------------
    async def start_run(self, session_id: str | None, body: dict[str, Any]) -> str:
        if session_id is not None and not self.store.session_exists(session_id):
            raise ApiError(404, "NOT_FOUND", f"no session {session_id}")
        own_session = session_id is None
        sid = session_id or self.store.create_session({"implicit": True})
        run_id = self.store.create_run(sid, {"workflow": body.get("workflow"), "status": "running"})
        try:
            spec = self.make_spec(run_id, sid, body)
        except ApiError:
            self.store.save_run(run_id, {"status": "failed", "failure_reason": "bad_request"})
            raise
        self.store.save_run(
            run_id,
            {"spec": spec.model_dump(mode="json"), "own_session": own_session, "start_seq": spec.start_seq},
        )
        env = self.build_env(spec, live=True)
        if spec.workflow == "optimize":
            await self.add_mcp_tools(env)
        log.info(
            json.dumps({"event": "run_start", "run_id": run_id, "session_id": sid, "workflow": spec.workflow})
        )

        def on_start(run: Run) -> None:
            self.runs[run_id] = run

        async def go() -> None:
            try:
                res = await execute(spec, env, self.clock, self.store, sid, on_start=on_start)
                self._finish(run_id, res)
            except Exception as e:  # noqa: BLE001 - recorded as a failed run, never a crash of the service
                log.error(json.dumps({"event": "run_error", "run_id": run_id, "error": type(e).__name__}))
                self.store.save_run(
                    run_id, {"status": "failed", "failure_reason": "step_failed", "error": type(e).__name__}
                )
            finally:
                if env.workspace is not None:
                    await env.workspace.close()
                self.runs.pop(run_id, None)

        self.tasks[run_id] = asyncio.ensure_future(go())
        return run_id

    def _finish(self, run_id: str, res: RunResult) -> None:
        m = res.meter
        for target, u in m.by_model.items():
            tot = self.usage.setdefault(
                target,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cost_ucu": 0},
            )
            for k in tot:
                tot[k] += u.get(k, 0)
        rec = {
            "status": res.status,
            "failure_reason": res.failure_reason,
            "result": res.text,
            "accepted": res.accepted,
            "usage": m.to_dict(),
            "cost_ucu": m.cost_ucu,
            "cost_estimated": m.estimated,
            "latency_ms": res.latency_ms,
            "latency_s": round(res.latency_ms / 1000.0, 3),
        }
        self.store.put_trace(run_id, res.trace.jsonl())
        self.store.save_run(run_id, rec)
        log.info(
            json.dumps(
                {
                    "event": "run_end",
                    "run_id": run_id,
                    "status": res.status,
                    "cost_ucu": m.cost_ucu,
                    "model_calls": m.model_calls,
                }
            )
        )

    async def wait(self, run_id: str) -> None:
        t = self.tasks.get(run_id)
        if t is not None:
            await asyncio.shield(t)

    def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            rec = self.store.get_run(run_id)
        except KeyError:
            raise ApiError(404, "NOT_FOUND", f"no run {run_id}") from None
        return {k: v for k, v in rec.items() if k not in ("spec",)}

    def trace(self, run_id: str) -> str:
        self.get_run(run_id)
        live = self.runs.get(run_id)
        if live is not None:
            return live.trace.jsonl()
        try:
            return self.store.get_trace(run_id)
        except KeyError:
            raise ApiError(404, "NOT_FOUND", f"no trace for {run_id}") from None

    def cancel(self, run_id: str) -> dict[str, Any]:
        rec = self.get_run(run_id)
        run = self.runs.get(run_id)
        if run is None:
            return {"id": run_id, "status": rec.get("status"), "cancelled": False}
        run.cancel()
        return {"id": run_id, "status": "cancelling", "cancelled": True}

    async def replay(self, run_id: str) -> dict[str, Any]:
        rec = self.store.get_run(run_id) if run_id else None
        if rec is None:
            raise ApiError(404, "NOT_FOUND", f"no run {run_id}")
        if rec.get("status") == "running":
            raise ApiError(409, "RUN_NOT_FINISHED", "the run is still running")
        if rec.get("status") == "cancelled":
            raise ApiError(409, "NOT_REPLAYABLE", "a cancelled run has no complete trace to replay")
        spans = load_jsonl(self.trace(run_id))
        prior = self.store.events(rec["session_id"], upto_seq=rec.get("start_seq", 0))

        def factory(spec: RunSpec) -> Env:
            return self.build_env(spec, live=False)

        out = await asyncio.to_thread(replay_spans, spans, factory, prior_events=prior)
        return out

    def context(self, session_id: str, policy: str, window: int) -> dict[str, Any]:
        if not self.store.session_exists(session_id):
            raise ApiError(404, "NOT_FOUND", f"no session {session_id}")
        if policy not in POLICIES:
            raise ApiError(422, "VALIDATION_ERROR", f"unknown policy {policy!r}")
        if window <= 0:
            raise ApiError(422, "VALIDATION_ERROR", "window must be positive")
        events = [e for e in self.store.events(session_id) if e.get("conv", "main") == "main"]
        d = derive_messages(events, policy, window, EstimatorCounter())
        messages = d.messages
        if d.proposal is not None and policy != "summarize":
            # dry run: apply the proposed compaction without logging it
            ev = {**d.proposal.to_event(), "seq": (events[-1]["seq"] if events else 0) + 1, "conv": "main"}
            d2 = derive_messages(events + [ev], policy, window, EstimatorCounter())
            messages, tokens = d2.messages, d2.tokens
        else:
            tokens = d.tokens
        return {
            "session_id": session_id,
            "policy": policy,
            "window": window,
            "tokens": tokens,
            "fits": tokens <= window,
            "compaction_due": d.proposal is not None,
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
        }

    async def aclose(self) -> None:
        for t in list(self.tasks.values()):
            if not t.done():
                t.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        if self._mcp is not None:
            await self._mcp.close()
            self._mcp = None
        if self.openai is not None:
            await self.openai.aclose()
        self.store.close()


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            '{"time_ms": %(created).3f, "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}'
        )
    )
    root = logging.getLogger("agent_runtime")
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def serve(args: Any) -> int:
    import uvicorn

    from agent_runtime.api.app import create_app

    settings = Settings(
        db=args.db,
        workspace_root=args.workspace_root,
        time_scale=args.time_scale,
        max_sandbox=args.max_sandbox,
        max_approval=args.max_approval,
        allow_exec=[json.loads(x) for x in args.allow_exec],
        allow_hosts=list(args.allow_hosts),
        openai_base_url=args.openai_base_url,
        openai_models=[m.strip() for m in args.openai_models.split(",") if m.strip()],
        openai_dialect=args.openai_dialect,
        examples_dir=getattr(args, "examples_dir", "examples"),
    )
    configure_logging()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((args.host, args.port))
    port = sock.getsockname()[1]
    if getattr(args, "port_file", None):
        Path(args.port_file).write_text(str(port), encoding="utf-8")
    log.info(json.dumps({"event": "listening", "host": args.host, "port": port}))
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    asyncio.run(server.serve(sockets=[sock]))
    return 0
