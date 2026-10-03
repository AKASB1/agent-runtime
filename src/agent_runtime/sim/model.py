"""The simulated model and provider.

A simulated call derives its action from the task's gold plan and from the conversation in its
request (the calls it issued and the results it got back), and draws from its own stream
``sim/<provider>/<model>/<task_id>/<attempt>/<ordinal>``: it is a pure function of its request and
the seed (the prefix cache, off by default, is the one piece of per-run state). Provider faults draw
from ``fault/<provider>/<run_id>/<ordinal>``. All numbers are assumptions (``docs/contracts.md``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.canon import canonical, digest
from agent_runtime.context.derive import STUB_PREFIX, SUMMARY_PREFIX, SimCounter
from agent_runtime.models.registry import ModelSpec
from agent_runtime.models.types import Message, ModelRequest, ModelResponse, ProviderError, ToolCall, Usage
from agent_runtime.rand import below, lognormal, stream
from agent_runtime.sim.tasks import GoldStep, Task, perturb

#: reference models: price in/out (ucu per 1M tokens), latency base and per output token (ms),
#: P(correct call) for easy/medium/hard, context window (tokens)
SIM_MODELS: dict[str, dict[str, Any]] = {
    "small": {
        "price_in": 150_000,
        "price_out": 600_000,
        "base_ms": 200,
        "per_token_ms": 8,
        "p": (0.92, 0.55, 0.15),
        "context": 4096,
    },
    "medium": {
        "price_in": 1_000_000,
        "price_out": 4_000_000,
        "base_ms": 500,
        "per_token_ms": 15,
        "p": (0.98, 0.85, 0.50),
        "context": 16384,
    },
    "large": {
        "price_in": 5_000_000,
        "price_out": 20_000_000,
        "base_ms": 1200,
        "per_token_ms": 30,
        "p": (0.995, 0.96, 0.82),
        "context": 32768,
    },
}
LATENCY_SIGMA = 0.2
OUT_TOOL_CALL, OUT_ANSWER, OUT_SUMMARY = 80, 120, 100
P_INVALID, P_NOT_FOUND = 0.3, 0.3  # of a wrong call; the rest (0.4) is silent


def sim_specs(providers: tuple[str, ...] = ("sim-a",), *, cache_price_pct: int = 10) -> list[ModelSpec]:
    out = []
    for prov in providers:
        for name, m in SIM_MODELS.items():
            out.append(
                ModelSpec(
                    id=name,
                    provider=prov,
                    wire="sim",
                    price_in_ucu=m["price_in"],
                    price_out_ucu=m["price_out"],
                    price_cache_read_ucu=m["price_in"] * cache_price_pct // 100,
                    context_tokens=m["context"],
                    latency_base_ms=m["base_ms"],
                    latency_per_token_ms=m["per_token_ms"],
                )
            )
    return out


@dataclass
class FaultProfile:
    p_5xx: float = 0.0
    p_429: float = 0.0
    p_timeout: float = 0.0
    retry_after_ms: int = 1000
    brownout: tuple[int, int] | None = None  # [start, end) in ms of the experiment clock
    error_latency_ms: int = 100


FAULT_PROFILES: dict[str, FaultProfile] = {
    "none": FaultProfile(),
    "calm": FaultProfile(p_5xx=0.005),
    "flaky": FaultProfile(p_5xx=0.06, p_429=0.03, p_timeout=0.02),
    "brownout": FaultProfile(p_5xx=0.005, brownout=(150_000, 270_000)),
}


# -- reading the conversation --------------------------------------------------------------------


def step_of(call_id: str) -> str | None:
    return call_id.rsplit("#", 1)[1] if "#" in call_id else None


@dataclass
class Seen:
    """What the request shows about the gold steps."""

    args: dict[str, dict[str, Any]] = field(default_factory=dict)  # step -> args of its latest ok call
    vis: dict[str, str] = field(default_factory=dict)  # step -> full | stub | summary
    issued: set[str] = field(default_factory=set)  # steps with any call in the input
    delegated: bool = False  # the input holds results of delegated branches (E6)


def read_conversation(messages: list[Message]) -> Seen:
    calls: dict[str, ToolCall] = {}
    seen = Seen()
    for m in messages:
        if m.role == "assistant":
            for c in m.tool_calls:
                calls[c.id] = c
                st = step_of(c.id)
                if st:
                    seen.issued.add(st)
            if m.content.startswith(SUMMARY_PREFIX):
                for fact in _summary_ids(m.content):
                    cid, _, raw_args = fact.partition("|")
                    st = step_of(cid)
                    if st and seen.vis.get(st) != "full":
                        seen.vis[st] = "summary"
                        seen.args[st] = json.loads(raw_args) if raw_args else {}
        elif m.role == "tool" and m.tool_call_id:
            st = step_of(m.tool_call_id)
            c = calls.get(m.tool_call_id)
            if st is None and c is not None and c.name == "delegate":
                _read_delegated(m.content, seen)
                continue
            if st is None or c is None:
                continue
            if m.content.startswith(STUB_PREFIX):
                if seen.vis.get(st) != "full":
                    seen.vis[st] = "stub"
                    seen.args[st] = dict(c.arguments)
                continue
            try:
                ok = bool(json.loads(m.content).get("ok"))
            except (json.JSONDecodeError, AttributeError):
                ok = False
            if ok:
                seen.vis[st] = "full"
                seen.args[st] = dict(c.arguments)
    return seen


def _read_delegated(content: str, seen: Seen) -> None:
    """A delegate result: the child's final text carries its branch's facts (``cid|args``), or not."""
    seen.delegated = True
    try:
        res = json.loads(content)
    except json.JSONDecodeError:
        return
    text = (res.get("result") or {}).get("text", "") if res.get("ok") else ""
    for fact in _summary_ids(text):
        cid, _, raw_args = fact.partition("|")
        st = step_of(cid)
        if st and seen.vis.get(st) != "full":
            seen.vis[st] = "delegated"
            seen.args[st] = json.loads(raw_args) if raw_args else {}


def _summary_ids(text: str) -> list[str]:
    if "facts:" not in text:
        return []
    body = text.split("facts:", 1)[1].strip()
    return [x.strip() for x in body.split(";") if x.strip()]


# -- the provider -----------------------------------------------------------------------------------


class SimProvider:
    def __init__(
        self,
        provider: str,
        tasks: dict[str, Task],
        *,
        seed: int,
        clock: Any,
        result_tokens: int = 250,
        faults: FaultProfile | None = None,
        keep: float = 0.9,
        cache: bool = False,
        recall: float = 0.7,
        keep_delegate: float = 0.95,
    ) -> None:
        self.provider = provider
        self.tasks = tasks
        self.seed = seed
        self.clock = clock
        self.counter = SimCounter(result_tokens)
        self.faults = faults or FaultProfile()
        self.keep = keep
        self.cache = cache
        self.keep_delegate = keep_delegate
        self._prev: dict[tuple[str, str], list[str]] = {}

    def _task(self, task_id: str) -> Task:
        """The task of a call; a child (``<task>/c<n>``) works on the gold steps of branch n."""
        base_id, _, rest = task_id.partition("/c")
        base = self.tasks[base_id]
        if not rest:
            return base
        n = int(rest.split("/c")[-1])
        from agent_runtime.planning.plan import plan_branches

        groups = plan_branches(list(base.steps))
        ids = set(groups[n - 1]) if 0 < n <= len(groups) else set()
        steps = tuple(
            GoldStep(s.id, s.tool, s.args, tuple(d for d in s.depends_on if d in ids))
            for s in base.steps
            if s.id in ids
        )
        return Task(
            task_id, base.text, base.tier, steps, base.answer, base.hint, False, base.shape, {"child": True}
        )

    async def complete(self, request: ModelRequest, spec: ModelSpec) -> ModelResponse:
        meta = request.meta
        run_id = meta.get("run_id", "")
        f = self.faults
        if f.brownout is not None and f.brownout[0] <= self.clock.now_ms() < f.brownout[1]:
            await self.clock.sleep(f.error_latency_ms)
            raise ProviderError("server_error", "brownout", status=503)
        if f.p_5xx or f.p_429 or f.p_timeout:
            u = stream(self.seed, f"fault/{self.provider}/{run_id}/{meta.get('call_ordinal', 0)}").random()
            if u < f.p_5xx:
                await self.clock.sleep(f.error_latency_ms)
                raise ProviderError("server_error", "simulated 5xx", status=500)
            if u < f.p_5xx + f.p_429:
                await self.clock.sleep(f.error_latency_ms)
                raise ProviderError(
                    "rate_limited", "simulated 429", status=429, retry_after_ms=f.retry_after_ms
                )
            if u < f.p_5xx + f.p_429 + f.p_timeout:
                await self.clock.sleep(request.timeout_ms + 1000)
                raise ProviderError("timeout", "simulated timeout")
        tokens_in = self.counter.base + sum(self.counter.message_tokens(m) for m in request.messages)
        if tokens_in > spec.context_tokens:
            raise ProviderError(
                "context_length", f"{tokens_in} tokens exceed the window {spec.context_tokens}", status=400
            )
        task = self._task(meta["task_id"])
        rng = stream(
            self.seed,
            f"sim/{self.provider}/{spec.id}/{meta['task_id']}/{meta.get('attempt', 1)}/{meta.get('ordinal', 1)}",
        )
        p = SIM_MODELS[spec.id]["p"][task.tier_index - 1]
        role = meta.get("role", "act")
        text, calls, out = "", [], OUT_ANSWER
        if role in ("act", "act_parallel"):
            text, calls = self._act(task, request, rng, p, parallel=bool(request.meta.get("parallel")))
            out = OUT_TOOL_CALL * len(calls) if calls else OUT_ANSWER
        elif role in ("plan", "replan"):
            plan = self._plan(task, request, rng, p, replan=role == "replan")
            text = canonical({"steps": plan})
            out = 60 + 40 * len(plan)
        elif role == "answer":
            text = self._answer(task, read_conversation(request.messages), request, rng, p)
        elif role == "summarize":
            text = self._summarize(request)
            out = OUT_SUMMARY
        else:
            text = perturb(task.answer) if rng.random() >= p else task.answer
        cache_read = self._cache_read(run_id, spec.target, request.messages) if self.cache else 0
        usage = Usage(input_tokens=tokens_in - cache_read, output_tokens=out, cache_read_tokens=cache_read)
        m = SIM_MODELS[spec.id]
        await self.clock.sleep(
            int(round((m["base_ms"] + m["per_token_ms"] * out) * lognormal(rng, LATENCY_SIGMA)))
        )
        return ModelResponse(
            text=text,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=usage,
            model=spec.id,
            provider=self.provider,
        )

    # -- prefix cache ------------------------------------------------------------------------------
    def _cache_read(self, run_id: str, target: str, messages: list[Message]) -> int:
        digests = [digest(m.model_dump(mode="json", exclude_none=True)) for m in messages]
        prev = self._prev.get((run_id, target), [])
        self._prev[(run_id, target)] = digests
        n = 0
        for a, b in zip(prev, digests, strict=False):
            if a != b:
                break
            n += 1
        if n == 0:
            return 0
        return self.counter.base + sum(self.counter.message_tokens(m) for m in messages[:n])

    # -- actions -----------------------------------------------------------------------------------
    def _draw_call(
        self, task: Task, sid: str, rng: Any, p: float, forced_wrong: bool = False
    ) -> tuple[str, dict[str, Any]]:
        """(tool name, arguments) of a call for gold step ``sid``: correct with probability p."""
        g = task.step(sid)
        if not forced_wrong and rng.random() < p:
            return g.tool, dict(g.args)
        v = 0.99 if forced_wrong else rng.random()
        if v < P_INVALID:
            if rng.random() < 0.5:
                return g.tool + "_v2", dict(g.args)  # unknown tool
            bad = dict(g.args)
            key = sorted(bad)[0]
            bad[key] = [bad[key]]  # wrong type: validation catches it
            return g.tool, bad
        if v < P_INVALID + P_NOT_FOUND:
            return g.tool, _nonexistent(g.tool, g.args)
        return g.tool, _plausible(g.tool, g.args, rng)

    def _act(
        self, task: Task, request: ModelRequest, rng: Any, p: float, *, parallel: bool
    ) -> tuple[str, list[ToolCall]]:
        seen = read_conversation(request.messages)
        ordinal = request.meta.get("ordinal", 1)
        done = self._completed(task, seen)
        remaining = [s for s in task.steps if s.id not in done]
        if not remaining:
            repeats = self._repeat_calls(task, seen, task.sinks(), rng, p, ordinal)
            if repeats:
                return "", repeats if parallel else repeats[:1]
            if task.meta.get("child"):
                return self._child_answer(task, seen, rng), []
            return self._answer(task, seen, request, rng, p, done=done), []
        ready = [s for s in remaining if all(d in done for d in s.depends_on)]
        if not parallel:
            ready = ready[:1]
        # a stubbed dependency is read again first
        repeats = self._repeat_calls(
            task, seen, sorted({d for s in ready for d in s.depends_on}), rng, p, ordinal
        )
        if repeats:
            return "", repeats if parallel else repeats[:1]
        calls = []
        for k, s in enumerate(ready):
            # a dependency the window dropped, or a wrong one, makes the call wrong and silent
            forced = any(
                (seen.vis.get(d) is None and d in done)
                or (d in seen.args and seen.args[d] != task.step(d).args)
                for d in s.depends_on
            )
            if not forced:
                for d in s.depends_on:
                    if seen.vis.get(d) == "summary" and rng.random() >= self.keep:
                        forced = True
            name, args = self._draw_call(task, s.id, rng, p, forced_wrong=forced)
            calls.append(ToolCall(id=f"c{ordinal}.{k + 1}#{s.id}", name=name, arguments=args))
        return "", calls

    def _repeat_calls(
        self, task: Task, seen: Seen, deps: list[str], rng: Any, p: float, ordinal: int
    ) -> list[ToolCall]:
        calls = []
        for k, d in enumerate(dd for dd in deps if seen.vis.get(dd) == "stub"):
            name, args = self._draw_call(task, d, rng, p)
            calls.append(ToolCall(id=f"c{ordinal}.r{k + 1}#{d}", name=name, arguments=args))
        return calls

    def _child_answer(self, task: Task, seen: Seen, rng: Any) -> str:
        """A child's final text: its branch's facts with probability ``keep_delegate``."""
        if rng.random() >= self.keep_delegate:
            return "[delegated] the results of this branch were lost"
        facts = [f"c#{s.id}|{canonical(seen.args[s.id])}" for s in task.steps if s.id in seen.args]
        return "[delegated] facts: " + "; ".join(facts)

    def _completed(self, task: Task, seen: Seen) -> set[str]:
        done = {sid for sid in seen.vis}
        # ancestors of completed steps were completed too (the window may have dropped them)
        changed = True
        while changed:
            changed = False
            for s in task.steps:
                if s.id in done:
                    for d in s.depends_on:
                        if d not in done:
                            done.add(d)
                            changed = True
        return done

    def _answer(
        self, task: Task, seen: Seen, request: ModelRequest, rng: Any, p: float, done: set[str] | None = None
    ) -> str:
        done = self._completed(task, seen) if done is None else done
        correct = all(s.id in done for s in task.steps)
        if seen.delegated and any(s.id not in seen.vis for s in task.steps):
            correct = False  # a branch whose results did not come back
        if correct:
            for s in task.steps:
                if s.id in seen.args and seen.args[s.id] != s.args:
                    correct = False  # a silent wrong call earlier
        for sid in task.sinks():
            vis = seen.vis.get(sid)
            if vis is None:
                correct = False
            elif vis == "summary" and rng.random() >= self.keep:
                correct = False
        if rng.random() >= p:
            correct = False
        return task.answer if correct else perturb(task.answer)

    def _plan(
        self, task: Task, request: ModelRequest, rng: Any, p: float, *, replan: bool
    ) -> list[dict[str, Any]]:
        seen = read_conversation(request.messages)
        done = self._completed(task, seen) if replan else set()
        out = []
        for s in task.steps:
            if s.id in done:
                continue
            name, args = self._draw_call(task, s.id, rng, p)
            out.append({"id": s.id, "tool": name, "args": args, "depends_on": list(s.depends_on)})
        return out

    def _summarize(self, request: ModelRequest) -> str:
        try:
            transcript = json.loads(request.messages[-1].content)
        except (json.JSONDecodeError, IndexError):
            transcript = []
        args: dict[str, dict] = {}
        facts = []
        for m in transcript:
            for c in m.get("tool_calls", []) or []:
                args[c["id"]] = c.get("arguments", {})
            if m.get("role") == "tool" and str(m.get("content", "")).startswith('{"ok":true'):
                cid = m.get("tool_call_id", "")
                facts.append(f"{cid}|{canonical(args.get(cid, {}))}")
        return f"{SUMMARY_PREFIX} facts: " + "; ".join(facts)


def _nonexistent(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    a = dict(args)
    if "customer_id" in a:
        a["customer_id"] = "C999"
    elif "order_id" in a:
        a["order_id"] = "O9999"
    elif "product_id" in a:
        a["product_id"] = "P999"
    elif "to_currency" in a:
        a["to_currency"] = "XXX"
    elif "op" in a:
        a["op"] = "median"
    return a


def _plausible(tool: str, args: dict[str, Any], rng: Any) -> dict[str, Any]:
    a = dict(args)
    shift = 1 + below(rng, 5)
    if tool == "send_report":
        a["value"] = perturb(a["value"])
    elif "customer_id" in a:
        a["customer_id"] = f"C{(int(a['customer_id'][1:]) - 1 + shift) % 200 + 1:03d}"
    elif "order_id" in a:
        a["order_id"] = f"O{(int(a['order_id'][1:]) - 1 + shift) % 2000 + 1:04d}"
    elif "product_id" in a:
        a["product_id"] = f"P{(int(a['product_id'][1:]) - 1 + shift) % 600 + 1:03d}"
    elif "to_currency" in a:
        options = [c for c in ("USD", "EUR", "GBP", "JPY", "CNY") if c != a["to_currency"]]
        a["to_currency"] = options[below(rng, len(options))]
    elif "values" in a:
        vals = list(a["values"])
        if vals:
            vals[below(rng, len(vals))] = round(vals[0] * 1.5 + 1.0, 2)
        a["values"] = vals
    return a
