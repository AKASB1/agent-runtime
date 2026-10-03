"""The run engine.

A ``Run`` owns one trace, one meter, the budget, the cancellation flag, and the session log of the
conversation(s) it works on. Workflows use three calls: ``call_model`` (route -> derive the input
from the log -> compact if due -> call with retry and fallback -> bill -> log the answer),
``call_structured`` (``call_model`` plus validation and the bounded repair loop), and ``call_tool``
(validate -> permission -> execute with timeout and read-only retries -> validate the output).
Every model call, tool call, routing decision, permission decision, compaction, subagent, and
validation is a span.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, Field

from agent_runtime.canon import canonical, digest
from agent_runtime.context.derive import (
    POLICIES,
    SUMMARY_PREFIX,
    EstimatorCounter,
    TokenCounter,
    apply_log,
    derive_messages,
    summary_request_messages,
)
from agent_runtime.models.accounting import cost_ucu, estimate_tokens
from agent_runtime.models.registry import ModelRegistry, ModelSpec
from agent_runtime.models.structured import StructuredOutputError, parse_structured
from agent_runtime.models.types import (
    Message,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)
from agent_runtime.permissions.policy import PermissionPolicy, decide, settle
from agent_runtime.rand import stream
from agent_runtime.routing.routers import AttemptOutcome, Decision, RouteContext, Router
from agent_runtime.telemetry.trace import Span, Trace
from agent_runtime.tools import Tool, ToolError, ToolRegistry, ToolResult

# ---------------------------------------------------------------------------------------------
# configuration


class Budget(BaseModel):
    deadline_ms: int | None = None
    max_cost_ucu: int | None = None
    max_model_calls: int | None = 40


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_base_ms: int = 200
    backoff_cap_ms: int = 5000


class ContextConfig(BaseModel):
    policy: str = "none"
    compact_at: float = 0.75
    compact_to: float = 0.5
    keep_last: int = 2


class RunConfig(BaseModel):
    seed: int = 1
    timeout_ms: int = 10_000
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fallback: list[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    context: ContextConfig = Field(default_factory=ContextConfig)
    concurrency: int = 4
    repair_budget: int = 2
    tool_retries: int = 2
    max_output_tokens: int = 512
    capture_payloads: bool = True
    max_replans: int = 2
    max_depth: int = 1


# ---------------------------------------------------------------------------------------------
# run endings


class RunAbort(Exception):
    status = "failed"

    def __init__(self, reason: str | None = None, message: str = "") -> None:
        super().__init__(message or reason or self.status)
        self.reason = reason


class StepFailed(RunAbort):
    status = "failed"


class BudgetExhausted(RunAbort):
    status = "budget_exhausted"


class Cancelled(RunAbort):
    status = "cancelled"


@dataclass
class Meter:
    cost_ucu: int = 0
    model_calls: int = 0
    provider_calls: int = 0
    tool_calls: int = 0
    tool_retries: int = 0
    retries: int = 0
    fallbacks: int = 0
    repairs: int = 0
    replans: int = 0
    escalations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False
    peak_context_tokens: int = 0
    peak_parent_context_tokens: int = 0
    compactions: int = 0
    repeat_reads: int = 0
    subagent_runs: int = 0
    skill_catalog_tokens: int = 0
    skill_body_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def bill(self, target: str, usage: Usage, cost: int) -> None:
        self.cost_ucu += cost
        self.model_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.estimated = self.estimated or usage.estimated
        m = self.by_model.setdefault(
            target, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cost_ucu": 0}
        )
        m["calls"] += 1
        m["input_tokens"] += usage.input_tokens
        m["output_tokens"] += usage.output_tokens
        m["cache_read_tokens"] += usage.cache_read_tokens
        m["cost_ucu"] += cost

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "by_model"}
        d["by_model"] = {k: dict(v) for k, v in sorted(self.by_model.items())}
        return d


# ---------------------------------------------------------------------------------------------
# environment


Approver = Callable[[str, dict[str, Any], str], Awaitable[bool] | bool]


@dataclass
class Env:
    registry: ModelRegistry
    adapters: dict[str, Any]
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    approver: Approver | None = None
    workspace: Any = None
    #: run-time verifier of a final answer (True = accept); None = no verifier
    verifier: Callable[[str, int], bool] | None = None
    counters: dict[str, TokenCounter] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    #: a SkillRegistry: its catalog goes into the system prompt, bodies load through load_skill
    skills: Any = None
    system_prompt: str = "You are a careful assistant that uses tools to answer."
    #: provider name -> callable building the request meta seen by that provider
    extra: dict[str, Any] = field(default_factory=dict)

    def counter_for(self, target: str) -> TokenCounter:
        return self.counters.get(target) or self.counters.get("*") or _ESTIMATOR


_ESTIMATOR = EstimatorCounter()


@dataclass(frozen=True)
class _Limit:
    max_cost_ucu: int | None
    max_model_calls: int | None
    base_cost: int
    base_calls: int


@dataclass(frozen=True)
class Frame:
    conv: str
    task_id: str
    path: str
    span: Span
    depth: int = 0
    tools_allow: tuple[str, ...] | None = None
    limits: tuple[_Limit, ...] = ()
    attempt: int = 1
    agent: str | None = None
    router: Any = None


_FRAME: ContextVar[Frame | None] = ContextVar("agent_runtime_frame", default=None)
#: the model span of the last successful call in this context (the origin of its tool calls)
LAST_MODEL_SPAN: ContextVar[str | None] = ContextVar("agent_runtime_last_model_span", default=None)


@dataclass
class RunResult:
    run_id: str
    status: str
    failure_reason: str | None
    text: str
    accepted: bool
    meter: Meter
    trace: Trace
    latency_ms: int
    session_id: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "result": self.text,
            "accepted": self.accepted,
            "latency_ms": self.latency_ms,
            "usage": self.meter.to_dict(),
            "session_id": self.session_id,
            **self.extra,
        }


Workflow = Callable[["Run", dict[str, Any]], Awaitable[str]]
WORKFLOWS: dict[str, Workflow] = {}


def workflow(name: str) -> Callable[[Workflow], Workflow]:
    def deco(fn: Workflow) -> Workflow:
        WORKFLOWS[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------------------------


class Run:
    def __init__(
        self,
        *,
        run_id: str,
        env: Env,
        config: RunConfig,
        clock: Any,
        store: Any,
        session_id: str,
        workflow: str,
        input: dict[str, Any],
        router: Router,
        task_id: str | None = None,
        hint: float | None = None,
        replayer: Any = None,
        record_config: dict[str, Any] | None = None,
    ) -> None:
        if config.context.policy not in POLICIES:
            raise ValueError(f"unknown context policy {config.context.policy!r}")
        self.run_id = run_id
        self.env = env
        self.config = config
        self.clock = clock
        self.store = store
        self.session_id = session_id
        self.workflow = workflow
        self.input = input
        self.router = router
        self.task_id = task_id or run_id
        self.hint = hint
        self.replayer = replayer
        self.record_config = record_config if record_config is not None else {}
        self.trace = Trace(run_id, clock, capture_payloads=config.capture_payloads)
        self.meter = Meter()
        self.cancelled = False
        self._t0 = clock.now_ms()
        self._ordinals: dict[tuple[str, str], int] = {}
        self._sim_ordinals: dict[tuple[str, int], int] = {}
        self._retry_ordinal = 0
        self._call_ordinal = 0
        self._sem = asyncio.Semaphore(max(1, config.concurrency))
        self._main: asyncio.Task | None = None
        self._events_cache: list[dict[str, Any]] | None = None
        self._seen_calls: dict[str, set[str]] = {}
        self.history: list[AttemptOutcome] = []
        self.accepted = True
        self.extra: dict[str, Any] = {}
        self.last_target: str | None = None

    # -- frames and spans ---------------------------------------------------------------------
    @property
    def frame(self) -> Frame:
        f = _FRAME.get()
        assert f is not None, "not inside a run"
        return f

    def elapsed_ms(self) -> int:
        return self.clock.now_ms() - self._t0

    def _key(self, kind: str) -> tuple[str, str, int]:
        path = self.frame.path
        k = (path, kind)
        n = self._ordinals.get(k, 0) + 1
        self._ordinals[k] = n
        return (path, kind, n)

    @contextlib.asynccontextmanager
    async def step(
        self, name: str, kind: str = "step", attrs: dict[str, Any] | None = None, **frame_updates: Any
    ):
        f = self.frame
        path = f"{f.path}/{name}" if f.path else name
        span = self.trace.start(kind, name, f.span, {"agent_runtime.step.path": path, **(attrs or {})})
        token = _FRAME.set(replace(f, path=path, span=span, **frame_updates))
        status = "ok"
        try:
            yield span
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except RunAbort as e:
            status = e.status
            raise
        except BaseException:
            status = "error"
            raise
        finally:
            _FRAME.reset(token)
            self.trace.end(span, status)

    # -- the session log ----------------------------------------------------------------------
    def events(self, conv: str | None = None) -> list[dict[str, Any]]:
        conv = conv or self.frame.conv
        if self._events_cache is None:
            self._events_cache = self.store.events(self.session_id)
        return [e for e in self._events_cache if e.get("conv", "main") == conv]

    def append(self, message: Message, conv: str | None = None) -> dict[str, Any]:
        conv = conv or self.frame.conv
        ev = self.store.append_event(
            self.session_id,
            {
                "type": "message",
                "run_id": self.run_id,
                "conv": conv,
                "message": message.model_dump(mode="json", exclude_none=True),
            },
        )
        if self._events_cache is not None:
            self._events_cache.append(ev)
        return ev

    def _append_compaction(self, event: dict[str, Any], conv: str) -> dict[str, Any]:
        ev = self.store.append_event(self.session_id, {**event, "run_id": self.run_id, "conv": conv})
        if self._events_cache is not None:
            self._events_cache.append(ev)
        return ev

    # -- budgets and cancellation -----------------------------------------------------------------
    def check_live(self, *, model_call: bool = False) -> None:
        if self.cancelled:
            raise Cancelled("cancelled")
        b = self.config.budget
        if b.deadline_ms is not None and self.elapsed_ms() >= b.deadline_ms:
            raise BudgetExhausted("deadline")
        if b.max_cost_ucu is not None and self.meter.cost_ucu >= b.max_cost_ucu:
            raise BudgetExhausted("max_cost")
        if model_call and b.max_model_calls is not None and self.meter.provider_calls >= b.max_model_calls:
            raise BudgetExhausted("max_model_calls")
        for lim in self.frame.limits:
            if lim.max_cost_ucu is not None and self.meter.cost_ucu - lim.base_cost >= lim.max_cost_ucu:
                raise BudgetExhausted("child_max_cost")
            if (
                model_call
                and lim.max_model_calls is not None
                and self.meter.provider_calls - lim.base_calls >= lim.max_model_calls
            ):
                raise BudgetExhausted("child_max_model_calls")

    def cancel(self) -> None:
        self.cancelled = True
        if self._main is not None and not self._main.done():
            self._main.cancel()

    # -- model calls -------------------------------------------------------------------------------
    def _spec_public(self, spec: ModelSpec) -> dict[str, Any]:
        return {
            "price_in_ucu": spec.price_in_ucu,
            "price_out_ucu": spec.price_out_ucu,
            "context_tokens": spec.context_tokens,
            "latency_base_ms": spec.latency_base_ms,
            "latency_per_token_ms": spec.latency_per_token_ms,
        }

    def _candidates(
        self, events: list[dict[str, Any]], needs: set[str], max_out: int, exclude: set[str]
    ) -> tuple[list[str], bool]:
        cands = []
        window_excluded = False
        policy = self.config.context.policy
        for spec in self.env.registry.specs():
            if spec.provider not in self.env.adapters or spec.target in exclude:
                continue
            if not needs <= set(spec.capabilities):
                continue
            budget = spec.context_tokens - max_out
            d = derive_messages(events, policy, budget, self.env.counter_for(spec.target), **self._ctx_kw())
            if d.irreducible > budget:
                window_excluded = True
                continue
            cands.append(spec.target)
        return cands, window_excluded

    def _ctx_kw(self) -> dict[str, Any]:
        c = self.config.context
        return {"compact_at": c.compact_at, "compact_to": c.compact_to, "keep_last": c.keep_last}

    def route_context(self, role: str, cands: list[str]) -> RouteContext:
        f = self.frame
        return RouteContext(
            step_kind=role,
            candidates=tuple(cands),
            spent_cost_ucu=self.meter.cost_ucu,
            elapsed_ms=self.elapsed_ms(),
            calls=self.meter.model_calls,
            attempt=f.attempt,
            history=tuple(self.history),
            hint=self.hint,
            task_id=f.task_id,
            deadline_ms=self.config.budget.deadline_ms,
            specs={t: self._spec_public(self.env.registry.get(t)) for t in cands},
        )

    def _route(self, role: str, cands: list[str]) -> Decision | None:
        router = self.frame.router or self.router
        ctx = self.route_context(role, cands)
        span = self.trace.start(
            "route", "route", self.frame.span, {"agent_runtime.step.path": self.frame.path}
        )
        d = router.choose(ctx) if cands else None
        if d is None or d.target is None:
            self.trace.end(
                span,
                "error",
                **{
                    "agent_runtime.route.policy": router.name,
                    "agent_runtime.route.chosen": None,
                    "agent_runtime.route.reason": "no candidate",
                    "agent_runtime.route.candidates": list(cands),
                },
            )
            return None
        self.trace.end(
            span,
            "ok",
            **{
                "agent_runtime.route.policy": router.name,
                "agent_runtime.route.chosen": d.target,
                "agent_runtime.route.reason": d.reason,
                "agent_runtime.route.candidates": list(cands),
            },
        )
        return d

    async def call_model(
        self,
        role: str,
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        exclude: set[str] | None = None,
        target: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """One model call whose input is derived from the conversation's log; the answer is logged."""
        self.check_live(model_call=True)
        max_out = max_output_tokens or self.config.max_output_tokens
        needs = ({"tools"} if tools else set()) | ({"json_schema"} if response_schema else set())
        events = self.events()
        cands, window_excluded = self._candidates(events, needs, max_out, exclude or set())
        if target is not None:
            cands = [t for t in cands if t == target]
        decision = self._route(role, cands)
        if decision is None:
            raise StepFailed(
                "context_length" if window_excluded else "provider_unavailable", "no target is offered"
            )
        chain = [decision.target] + [t for t in self.config.fallback if t != decision.target and t in cands]
        last_reason = "provider_unavailable"
        too_small = 0  # the largest window that failed with context_length in this step
        for pos, tgt in enumerate(chain):
            spec = self.env.registry.get(tgt)
            if pos > 0:
                if spec.context_tokens <= too_small:
                    continue  # context_length falls back only to a target with a larger window
                self.meter.fallbacks += 1
            try:
                return await self._call_target(
                    spec, role, tools or [], response_schema, max_out, fallback=pos > 0, extra_meta=meta
                )
            except _TargetFailed as tf:
                last_reason = tf.reason
                if tf.reason == "context_length":
                    too_small = max(too_small, spec.context_tokens)
                if tf.reason == "bad_request":
                    raise StepFailed("bad_request", tf.message) from None
                continue
        raise StepFailed(last_reason, "every target failed")

    async def _call_target(
        self,
        spec: ModelSpec,
        role: str,
        tools: list[ToolSpec],
        schema: dict | None,
        max_out: int,
        *,
        fallback: bool,
        extra_meta: dict[str, Any] | None = None,
    ) -> ModelResponse:
        f = self.frame
        policy = self.config.context.policy
        budget = spec.context_tokens - max_out
        counter = self.env.counter_for(spec.target)
        events = self.events()
        d = derive_messages(events, policy, budget, counter, **self._ctx_kw())
        if d.proposal is not None:
            await self._compact(d.proposal, spec, budget, counter)
            events = self.events()
            d = derive_messages(events, policy, budget, counter, **self._ctx_kw())
        if d.tokens > budget:
            raise _TargetFailed(
                "context_length", f"input of {d.tokens} tokens exceeds the budget {budget} of {spec.target}"
            )
        self.meter.peak_context_tokens = max(self.meter.peak_context_tokens, d.tokens)
        if f.depth == 0:
            self.meter.peak_parent_context_tokens = max(self.meter.peak_parent_context_tokens, d.tokens)
        sim_key = (f.conv, f.attempt)
        ordinal = self._sim_ordinals.get(sim_key, 0) + 1
        meta = {
            "task_id": f.task_id,
            "role": role,
            "hint": self.hint,
            "attempt": f.attempt,
            "ordinal": ordinal,
            "run_id": self.run_id,
            **(extra_meta or {}),
        }
        req = ModelRequest(
            model=spec.target,
            messages=d.messages,
            tools=tools,
            response_schema=schema,
            max_output_tokens=max_out,
            timeout_ms=self.config.timeout_ms,
            meta=meta,
        )
        ctx_attrs = {
            "agent_runtime.context.tokens": d.tokens,
            "agent_runtime.context.window": spec.context_tokens,
            "agent_runtime.context.budget": budget,
            "agent_runtime.context.policy": policy,
            "agent_runtime.context.conv": f.conv,
            "agent_runtime.context.log_seq": d.last_seq,
            "agent_runtime.context.messages_sha256": digest(
                [m.model_dump(mode="json", exclude_none=True) for m in d.messages]
            ),
        }
        resp = await self._attempts(spec, req, role, ctx_attrs, fallback=fallback)
        self._sim_ordinals[sim_key] = ordinal
        self.append(
            Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls, reasoning=resp.reasoning)
        )
        return resp

    async def _attempts(
        self, spec: ModelSpec, req: ModelRequest, role: str, ctx_attrs: dict, *, fallback: bool
    ) -> ModelResponse:
        f = self.frame
        adapter = self.env.adapters[spec.provider]
        rc = self.config.retry
        for attempt in range(1, rc.max_attempts + 1):
            self.check_live(
                model_call=True
            )  # before every provider attempt (fallbacks and after a summary too)
            self._call_ordinal += 1
            req = req.model_copy(update={"meta": {**req.meta, "call_ordinal": self._call_ordinal}})
            key = self._key("model")
            self.meter.provider_calls += 1
            payload = req.model_dump(mode="json", exclude_none=True)
            span = self.trace.start(
                "model",
                "chat",
                f.span,
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": spec.provider,
                    "gen_ai.request.model": spec.id,
                    "agent_runtime.attempt": attempt,
                    "agent_runtime.role": role,
                    "agent_runtime.step.path": f.path,
                    "agent_runtime.ordinal": key[2],
                    "agent_runtime.fallback": fallback,
                    **ctx_attrs,
                },
            )
            self.trace.set_payload(span, "request", payload)
            err: ProviderError | None = None
            resp: ModelResponse | None = None
            try:
                if self.replayer is not None:
                    resp, err = await self.replayer.model(self, key, span, payload)
                else:
                    resp = await self.clock.wait_for(adapter.complete(req, spec), req.timeout_ms)
            except TimeoutError:
                err = ProviderError("timeout", f"no answer within {req.timeout_ms} ms")
            except ProviderError as e:
                err = e
            except asyncio.CancelledError:
                self.trace.end(span, "cancelled", **{"error.type": "cancelled"})
                raise
            if resp is not None:
                cost = cost_ucu(resp.usage, spec)
                self.meter.bill(spec.target, resp.usage, cost)
                u = resp.usage
                self.trace.set_payload(span, "response", resp.model_dump(mode="json", exclude_none=True))
                self.trace.end(
                    span,
                    "ok",
                    **{
                        "gen_ai.usage.input_tokens": u.total_input_tokens,
                        "gen_ai.usage.output_tokens": u.output_tokens,
                        "gen_ai.usage.cache_read.input_tokens": u.cache_read_tokens,
                        "gen_ai.usage.cache_write.input_tokens": u.cache_write_tokens,
                        "gen_ai.response.finish_reasons": [resp.finish_reason],
                        "agent_runtime.usage.estimated": u.estimated,
                        "agent_runtime.cost_ucu": cost,
                    },
                )
                LAST_MODEL_SPAN.set(span.span_id)
                self.last_target = spec.target
                return resp
            assert err is not None
            self.trace.set_payload(span, "response", {"error": err.to_dict()})
            self.trace.end(span, "error", **{"error.type": err.kind, "agent_runtime.cost_ucu": 0})
            if err.kind in ("timeout", "rate_limited", "server_error"):
                if attempt < rc.max_attempts:
                    self.meter.retries += 1
                    await self.clock.sleep(self._backoff(attempt, err))
                    continue
                raise _TargetFailed("provider_unavailable", err.message)
            if err.kind == "auth":
                raise _TargetFailed("auth", err.message)
            if err.kind == "context_length":
                raise _TargetFailed("context_length", err.message)
            raise _TargetFailed("bad_request", err.message)
        raise _TargetFailed("provider_unavailable", "attempts used up")

    def _backoff(self, attempt: int, err: ProviderError) -> int:
        rc = self.config.retry
        self._retry_ordinal += 1
        u = stream(self.config.seed, f"retry/{self.run_id}/{self._retry_ordinal}").random()
        delay = int(u * min(rc.backoff_cap_ms, rc.backoff_base_ms * (2 ** (attempt - 1))))
        if err.kind == "rate_limited" and err.retry_after_ms:
            delay = max(delay, int(err.retry_after_ms))
        return delay

    async def _compact(self, proposal: Any, spec: ModelSpec, budget: int, counter: TokenCounter) -> None:
        f = self.frame
        span = self.trace.start(
            "context",
            "compact",
            f.span,
            {
                "agent_runtime.context.policy": proposal.policy,
                "agent_runtime.step.path": f.path,
                "agent_runtime.context.conv": f.conv,
            },
        )
        try:
            if proposal.policy == "summarize":
                summary = await self._summarize(proposal.replacement["summarize"], spec)
                proposal.replacement["summary"] = summary
            ev = self._append_compaction(proposal.to_event(), f.conv)
            after = derive_messages(
                self.events(), self.config.context.policy, budget, counter, **self._ctx_kw()
            )
            self.meter.compactions += 1
            self.trace.end(
                span,
                "ok",
                **{
                    "agent_runtime.context.tokens_before": proposal.tokens_before,
                    "agent_runtime.context.tokens_after": after.tokens,
                    "agent_runtime.context.budget": budget,
                    "agent_runtime.context.covers": [proposal.covers_from, proposal.covers_to],
                    "agent_runtime.context.compaction_id": ev["id"],
                },
            )
        except BaseException:
            self.trace.end(span, "error")
            raise

    async def _summarize(self, seqs: list[int], spec: ModelSpec) -> str:
        """One billed and traced model call with role ``summarize``; its input is derived from the log."""
        f = self.frame
        msgs = summary_request_messages(self.events(), seqs)
        target = self.env.extra.get("summarizer") or spec.target
        sspec = self.env.registry.get(target)
        key = (f.conv, -f.attempt)
        ordinal = self._sim_ordinals.get(key, 0) + 1
        self._sim_ordinals[key] = ordinal
        req = ModelRequest(
            model=sspec.target,
            messages=msgs,
            max_output_tokens=200,
            timeout_ms=self.config.timeout_ms,
            meta={
                "task_id": f.task_id,
                "role": "summarize",
                "hint": self.hint,
                "attempt": f.attempt,
                "ordinal": f"s{ordinal}",
                "run_id": self.run_id,
            },
        )
        ctx_attrs = {
            "agent_runtime.context.tokens": 0,
            "agent_runtime.context.window": sspec.context_tokens,
            "agent_runtime.context.conv": f.conv,
            "agent_runtime.context.summarizes": seqs,
            "agent_runtime.context.log_seq": self.events()[-1]["seq"] if self.events() else 0,
        }
        try:
            resp = await self._attempts(sspec, req, "summarize", ctx_attrs, fallback=False)
        except _TargetFailed as tf:
            raise StepFailed(tf.reason, "summarize call failed") from None
        text = resp.text.strip()
        return text if text.startswith(SUMMARY_PREFIX) else f"{SUMMARY_PREFIX} {text}"

    async def call_structured(
        self,
        role: str,
        schema: dict[str, Any],
        *,
        validate: Callable[[Any], None] | None = None,
        max_output_tokens: int | None = None,
        repair_note: str = "Reply again with one JSON value that matches the schema.",
        error_prefix: str | None = None,
    ) -> tuple[Any, ModelResponse]:
        """A model call with structured output: validation, bounded repair, then the next model.
        ``error_prefix`` (a diagnostic code) is put in front of every validation message."""
        exclude: set[str] = set()
        repairs = 0
        while True:
            resp = await self.call_model(
                role, response_schema=schema, max_output_tokens=max_output_tokens, exclude=exclude
            )
            try:
                value = parse_structured(resp.text, schema)
                if validate is not None:
                    validate(value)
                self._validate_span(role, True, None)
                return value, resp
            except StructuredOutputError as err:
                e = f"{error_prefix}: {err}" if error_prefix else str(err)
                self._validate_span(role, False, e)
                self.meter.repairs += 1
                if repairs < self.config.repair_budget:
                    repairs += 1
                    self.append(
                        Message(role="user", content=f"Your last output was invalid: {e}. {repair_note}")
                    )
                    continue
                exclude.add(resp_target(resp))
                repairs = 0
                self.append(Message(role="user", content=f"Your last output was invalid: {e}. {repair_note}"))
                cands, _ = self._candidates(
                    self.events(),
                    {"json_schema"},
                    max_output_tokens or self.config.max_output_tokens,
                    exclude,
                )
                router = self.frame.router or self.router
                if not cands or router.choose(self.route_context(role, cands)) is None:
                    raise StepFailed("invalid_output", e) from None

    def _validate_span(self, what: str, ok: bool, message: str | None) -> None:
        f = self.frame
        s = self.trace.start("validate", what, f.span, {"agent_runtime.step.path": f.path})
        self.trace.end(
            s, "ok" if ok else "error", **({"agent_runtime.validation.error": message} if message else {})
        )

    # -- tool calls ----------------------------------------------------------------------------
    def tool_specs(self) -> list[ToolSpec]:
        allow = self.frame.tools_allow
        return self.env.tools.specs(None if allow is None else list(allow))

    async def call_tool(self, call: ToolCall, origin: Span | str) -> ToolResult:
        """Validate -> permission -> execute (timeout, read-only retries) -> validate the output."""
        self.check_live()
        f = self.frame
        origin_id = origin.span_id if isinstance(origin, Span) else origin
        key = self._key("tool")
        allow = f.tools_allow
        tool = self.env.tools.get(call.name) if (allow is None or call.name in allow) else None
        base_attrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": call.name,
            "gen_ai.tool.call.id": call.id,
            "agent_runtime.origin_span": origin_id,
            "agent_runtime.step.path": f.path,
            "agent_runtime.ordinal": key[2],
        }
        is_delegate = tool is not None and tool.meta.get("runtime") == "delegate"
        if self.replayer is not None and not is_delegate:
            return await self.replayer.tool(self, key, call, tool, base_attrs)
        seen = self._seen_calls.setdefault(f.conv, set())
        sig = canonical([call.name, call.arguments])
        if sig in seen:
            self.meter.repeat_reads += 1
        if tool is None:
            why = (
                "not in this agent's allowlist"
                if (allow is not None and call.name in self.env.tools)
                else "unknown tool"
            )
            return self._tool_span_only(
                base_attrs, call, ToolResult.error("UNKNOWN_TOOL", f"{call.name!r}: {why}")
            )
        value, err = tool.validate_input(call.arguments)
        if err is not None:
            return self._tool_span_only(base_attrs, call, ToolResult.error("INVALID_ARGUMENTS", err))
        # permission layer: every call takes this path; capability "none" is allowed without a span
        if tool.capability != "none":
            resolved = (
                tool.meta["resolve"](call.arguments, self.env.workspace)
                if "resolve" in tool.meta
                else {"paths_ok": True}
            )
            d = decide(self.env.policy, tool.capability, resolved)
            pspan = self.trace.start(
                "permission",
                "permission",
                f.span,
                {
                    "agent_runtime.permission.capability": tool.capability,
                    "agent_runtime.permission.resolved": resolved,
                    "agent_runtime.permission.args_sha256": digest(call.arguments),
                    "agent_runtime.step.path": f.path,
                    "agent_runtime.origin_span": origin_id,
                    "gen_ai.tool.call.id": call.id,
                    "gen_ai.tool.name": call.name,
                },
            )
            if d.final:
                final, by = "deny", None
            else:
                final, by = await settle(self.env.policy, d, self.env.approver, call.name, resolved)
            self.trace.end(
                pspan,
                "ok",
                **{
                    "agent_runtime.permission.decision": final,
                    "agent_runtime.permission.rule": d.rule,
                    "agent_runtime.permission.approved_by": by,
                },
            )
            if final != "allow":
                return ToolResult.error("PERMISSION_DENIED", f"denied by rule {d.rule}", rule=d.rule)
        seen.add(sig)
        return await self._execute(tool, value, call, base_attrs)

    def _tool_span_only(self, attrs: dict, call: ToolCall, result: ToolResult) -> ToolResult:
        s = self.trace.start("tool", call.name, self.frame.span, dict(attrs))
        self.trace.set_payload(s, "args", call.arguments)
        self.trace.set_payload(s, "result", result.to_dict())
        self.trace.end(
            s, "error", **{"error.type": "tool_error", "agent_runtime.tool.error_code": result.error_code}
        )
        return result

    async def _execute(self, tool: Tool, value: Any, call: ToolCall, attrs: dict) -> ToolResult:
        f = self.frame
        if tool.meta.get("runtime") == "delegate":
            return await self._delegate(tool, value, call, attrs)
        tries = 1 + (self.config.tool_retries if tool.side_effect in ("read", "none") else 0)
        result: ToolResult | None = None
        retries = 0
        await self._sem.acquire()
        try:
            self.check_live()
        except RunAbort:
            self._sem.release()
            raise
        span = self.trace.start(
            "tool", call.name, f.span, {**attrs, "agent_runtime.tool.side_effect": tool.side_effect}
        )
        self.trace.set_payload(span, "args", call.arguments)
        try:
            # the run's concurrency limit is held for the whole call (retries included)
            for i in range(tries):
                if i > 0:
                    self.check_live()
                self.meter.tool_calls += 1 if i == 0 else 0
                try:
                    out = await self.clock.wait_for(tool.handler(value), tool.timeout_ms)
                    dumped, err = tool.dump_output(out)
                    result = (
                        ToolResult(ok=True, result=dumped)
                        if err is None
                        else ToolResult.error("INVALID_OUTPUT", err)
                    )
                except TimeoutError:
                    result = ToolResult.error("TIMEOUT", f"no result within {tool.timeout_ms} ms")
                except ToolError as e:
                    result = ToolResult.error(e.code, e.message, **e.details)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # a tool that raises: the error is the tool result
                    result = ToolResult.error("FAILED", f"{type(e).__name__}: {str(e)[:200]}")
                if result.ok or result.error_code not in ("TIMEOUT", "UNAVAILABLE") or i == tries - 1:
                    break
                retries += 1
                self.meter.tool_retries += 1
        except asyncio.CancelledError:
            self.trace.set_payload(
                span, "result", {"ok": False, "error": {"code": "CANCELLED", "message": "cancelled"}}
            )
            self.trace.end(span, "cancelled", **{"error.type": "cancelled"})
            raise
        except RunAbort as e:
            self.trace.set_payload(
                span, "result", {"ok": False, "error": {"code": "ABORTED", "message": e.status}}
            )
            self.trace.end(span, e.status)
            raise
        finally:
            self._sem.release()
        assert result is not None
        if result.ok and tool.meta.get("skill_loader"):
            self.meter.skill_body_tokens += estimate_tokens(str(result.result.get("body", "")))
        self.trace.set_payload(span, "result", result.to_dict())
        extra = {"agent_runtime.retry": retries}
        if not result.ok:
            extra["error.type"] = "tool_timeout" if result.error_code == "TIMEOUT" else "tool_error"
            extra["agent_runtime.tool.error_code"] = result.error_code
        self.trace.end(span, "ok" if result.ok else "error", **extra)
        return result

    async def _delegate(self, tool: Tool, value: Any, call: ToolCall, attrs: dict) -> ToolResult:
        """Run a child agent (fresh context, allowlist, budget share, depth limit) inside this run."""

        f = self.frame
        span = self.trace.start(
            "tool", call.name, f.span, {**attrs, "agent_runtime.tool.side_effect": tool.side_effect}
        )
        self.trace.set_payload(span, "args", call.arguments)
        self.meter.tool_calls += 1
        spec = self.env.agents.get(value.agent)
        depth = f.depth + 1
        if spec is None:
            result = ToolResult.error("INVALID_ARGUMENTS", f"no agent named {value.agent!r}")
        elif depth > min(self.config.max_depth, 3):
            result = ToolResult.error(
                "SUBAGENT_FAILED", f"depth limit {self.config.max_depth} reached", status="failed"
            )
        else:
            result = await self._run_child(spec, value.task, depth, span)
        self.trace.set_payload(span, "result", result.to_dict())
        extra: dict[str, Any] = {"agent_runtime.retry": 0}
        if not result.ok:
            extra["error.type"] = (
                "subagent_failed" if result.error_code == "SUBAGENT_FAILED" else "tool_error"
            )
            extra["agent_runtime.tool.error_code"] = result.error_code
        self.trace.end(span, "ok" if result.ok else "error", **extra)
        return result

    async def _run_child(self, spec: Any, task: str, depth: int, tool_span: Span) -> ToolResult:
        from agent_runtime.agents import cap_text
        from agent_runtime.routing.routers import FixedRouter

        f = self.frame
        self._children = getattr(self, "_children", 0) + 1
        n = self._children
        parent_allow = f.tools_allow
        allow = tuple(
            t
            for t in spec.tools
            if t in self.env.tools
            and (parent_allow is None or t in parent_allow)
            and (t != "delegate" or depth < min(self.config.max_depth, 3))
        )
        limit = _Limit(
            spec.max_cost_ucu, spec.max_model_calls, self.meter.cost_ucu, self.meter.provider_calls
        )
        sub = self.trace.start(
            "subagent",
            spec.name,
            tool_span,
            {
                "agent_runtime.agent.name": spec.name,
                "agent_runtime.agent.depth": depth,
                "agent_runtime.step.path": f"{f.path}/c{n}",
            },
        )
        frame = replace(
            f,
            conv=f"{f.conv}/c{n}"
            if f.conv != "main"
            else f"{self.run_id}/c{n}",  # a child's conversation is its run's
            task_id=f"{f.task_id}/c{n}",
            path=f"{f.path}/c{n}",
            span=sub,
            depth=depth,
            tools_allow=allow,
            limits=(*f.limits, limit),
            attempt=1,
            agent=spec.name,
            router=FixedRouter(spec.target, spec.fallback) if spec.target else f.router,
        )
        token = _FRAME.set(frame)
        self.meter.subagent_runs += 1
        status, reason, text, steps = "succeeded", None, "", 0
        try:
            self.append(Message(role="system", content=spec.system_prompt))
            self.append(Message(role="user", content=task))
            while True:
                if steps >= spec.max_steps:
                    status, reason = "failed", "max_steps"
                    break
                steps += 1
                async with self.step(f"turn{steps}"):
                    resp = await self.call_model("act", tools=self.tool_specs())
                    if not resp.tool_calls:
                        text = resp.text
                        break
                    await self.run_tool_calls(resp.tool_calls, LAST_MODEL_SPAN.get(), parallel=False)
        except BudgetExhausted as e:
            status, reason = "budget_exhausted", e.reason
        except StepFailed as e:
            status, reason = "failed", e.reason
        finally:
            _FRAME.reset(token)
            self.trace.end(
                sub,
                "ok" if status == "succeeded" else status,
                **{
                    "agent_runtime.agent.status": status,
                    "agent_runtime.agent.steps": steps,
                },
            )
        if status != "succeeded":
            return ToolResult.error(
                "SUBAGENT_FAILED",
                f"child {spec.name} ended {status}: {reason}",
                status=status,
                reason=reason,
                steps=steps,
            )
        return ToolResult(ok=True, result={"status": status, "text": cap_text(text), "steps": steps})

    async def run_tool_calls(
        self, calls: list[ToolCall], origin: Span, *, parallel: bool
    ) -> list[ToolResult]:
        """Run the calls of one model turn (in parallel under the run's limit, or one by one) and log
        each result as a tool message, in call order."""
        if parallel and len(calls) > 1:
            tasks = [asyncio.ensure_future(self.call_tool(c, origin)) for c in calls]
            try:
                results = list(await asyncio.gather(*tasks))
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        else:
            results = []
            for c in calls:
                results.append(await self.call_tool(c, origin))
        for c, r in zip(calls, results, strict=True):
            self.append(Message(role="tool", tool_call_id=c.id, content=r.content()))
        return results

    # -- execution ------------------------------------------------------------------------------
    async def execute(self) -> RunResult:
        wf = WORKFLOWS[self.workflow]
        run_span = self.trace.start(
            "run",
            self.workflow,
            None,
            {
                "agent_runtime.workflow": self.workflow,
                "agent_runtime.config": self.record_config,
                "agent_runtime.input": self.input,
                "agent_runtime.session_id": self.session_id,
            },
        )
        token = _FRAME.set(Frame(conv="main", task_id=self.task_id, path="", span=run_span))
        status, reason, text = "succeeded", None, ""
        try:
            self._main = asyncio.ensure_future(wf(self, self.input))
            text = await self._main
            if not self.accepted:
                status, reason = "failed", self.extra.get("failure_reason", "verifier_rejected")
        except asyncio.CancelledError:
            status, reason = "cancelled", None
            if not self.cancelled:
                raise
        except RunAbort as e:
            status = e.status
            reason = e.reason if e.status == "failed" else None
        finally:
            _FRAME.reset(token)
        self.trace.close_open(status if status != "succeeded" else "error", keep=run_span)
        m = self.meter
        self.trace.end(
            run_span,
            "ok" if status == "succeeded" else status,
            **{
                "agent_runtime.status": status,
                "agent_runtime.failure_reason": reason,
                "agent_runtime.cost_ucu": m.cost_ucu,
                "agent_runtime.model_calls": m.model_calls,
                "agent_runtime.provider_calls": m.provider_calls,
                "agent_runtime.tool_calls": m.tool_calls,
                "gen_ai.usage.input_tokens": m.input_tokens + m.cache_read_tokens + m.cache_write_tokens,
                "gen_ai.usage.output_tokens": m.output_tokens,
                "agent_runtime.result_sha256": digest(text),
            },
        )
        accepted = status == "succeeded" and self.accepted
        return RunResult(
            run_id=self.run_id,
            status=status,
            failure_reason=reason,
            text=text,
            accepted=accepted,
            meter=m,
            trace=self.trace,
            latency_ms=self.elapsed_ms(),
            session_id=self.session_id,
            extra=dict(self.extra),
        )


class _TargetFailed(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.message = message


def resp_target(resp: ModelResponse) -> str:
    return f"{resp.provider}/{resp.model}"


def conversation_messages(run: Run, conv: str | None = None) -> list[Message]:
    return [m for _, m in apply_log(run.events(conv))]
