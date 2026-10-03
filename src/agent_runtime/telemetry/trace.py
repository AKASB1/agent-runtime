"""Trace writer and invariants.

Span ids are ``sp-<n>`` from a counter per trace, in start order; times are integer ms since the
start of the run on the run's clock. ``model`` and ``tool`` spans carry their payloads and the
SHA-256 of each one's canonical JSON. No UUID and no timestamp enters a trace.
"""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.canon import canonical, digest

SPAN_KINDS = (
    "run",
    "step",
    "model",
    "tool",
    "route",
    "plan",
    "validate",
    "context",
    "permission",
    "subagent",
)
PAYLOAD_KEYS = {"model": ("request", "response"), "tool": ("args", "result")}


class TraceInvalid(AssertionError):
    pass


class Span:
    __slots__ = (
        "trace_id",
        "span_id",
        "parent_id",
        "kind",
        "name",
        "start_ms",
        "end_ms",
        "status",
        "attrs",
        "payload",
    )

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        kind: str,
        name: str,
        start_ms: int,
        attrs: dict,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.start_ms = start_ms
        self.end_ms: int | None = None
        self.status = "open"
        self.attrs = attrs
        self.payload: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "v": 1,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "status": self.status,
            "attrs": self.attrs,
        }
        for key in PAYLOAD_KEYS.get(self.kind, ()):
            if key in self.payload:
                d[f"{key}_sha256"] = digest(self.payload[key])
                if self.payload.get("_capture", True):
                    d[key] = self.payload[key]
        return d


class Trace:
    def __init__(self, trace_id: str, clock: Any, *, capture_payloads: bool = True) -> None:
        self.trace_id = trace_id
        self.clock = clock
        self.t0 = clock.now_ms()
        self.capture = capture_payloads
        self.spans: list[Span] = []
        self._n = 0

    def now(self) -> int:
        return self.clock.now_ms() - self.t0

    def start(self, kind: str, name: str, parent: Span | str | None, attrs: dict | None = None) -> Span:
        if kind not in SPAN_KINDS:
            raise ValueError(f"unknown span kind {kind!r}")
        self._n += 1
        pid = parent.span_id if isinstance(parent, Span) else parent
        s = Span(self.trace_id, f"sp-{self._n}", pid, kind, name, self.now(), dict(attrs or {}))
        self.spans.append(s)
        return s

    def end(self, span: Span, status: str = "ok", **attrs: Any) -> None:
        if span.end_ms is not None:
            return
        span.attrs.update(attrs)
        span.end_ms = self.now()
        span.status = status

    def set_payload(self, span: Span, key: str, value: Any) -> None:
        span.payload[key] = value
        span.payload["_capture"] = self.capture

    def close_open(self, status: str, keep: Span | None = None) -> None:
        for s in reversed(self.spans):
            if s.end_ms is None and s is not keep:
                self.end(s, status)

    def dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]

    def jsonl(self) -> str:
        return "".join(canonical(d) + "\n" for d in self.dicts())


def load_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def validate_trace(spans: list[dict[str, Any]]) -> None:
    """Raise :class:`TraceInvalid` naming the first violated invariant."""
    if not spans:
        raise TraceInvalid("empty trace")
    by_id = {s["span_id"]: s for s in spans}
    if len(by_id) != len(spans):
        raise TraceInvalid("duplicate span ids")
    runs = [s for s in spans if s["kind"] == "run"]
    if len(runs) != 1 or spans[0]["kind"] != "run":
        raise TraceInvalid("a trace has exactly one run span, first")
    run = runs[0]
    for s in spans:
        if s["end_ms"] is None or s["status"] == "open":
            raise TraceInvalid(f"{s['span_id']} is not closed")
        if s["end_ms"] < s["start_ms"]:
            raise TraceInvalid(f"{s['span_id']} ends before it starts")
        pid = s["parent_id"]
        if s is run:
            if pid is not None:
                raise TraceInvalid("the run span has a parent")
            continue
        if pid not in by_id:
            raise TraceInvalid(f"{s['span_id']}: parent {pid} does not exist")
        p = by_id[pid]
        if not (p["start_ms"] <= s["start_ms"] and s["end_ms"] <= p["end_ms"]):
            raise TraceInvalid(f"{s['span_id']} is not enclosed by its parent {pid}")
    model_cost = sum(int(s["attrs"].get("agent_runtime.cost_ucu", 0)) for s in spans if s["kind"] == "model")
    if int(run["attrs"].get("agent_runtime.cost_ucu", 0)) != model_cost:
        raise TraceInvalid(
            f"run cost {run['attrs'].get('agent_runtime.cost_ucu')} != sum of model spans {model_cost}"
        )
    # every tool call of a model span is answered once (tool span, or a permission span that denied it)
    answers: dict[tuple[str, str], int] = {}
    for s in spans:
        if s["kind"] == "tool" or (
            s["kind"] == "permission" and s["attrs"].get("agent_runtime.permission.decision") == "deny"
        ):
            key = (s["attrs"].get("agent_runtime.origin_span"), s["attrs"].get("gen_ai.tool.call.id"))
            if s["kind"] == "permission" and key[0] is None:
                continue
            answers[key] = answers.get(key, 0) + 1
    plan_steps: dict[str, set[str]] = {}
    for s in spans:
        if s["kind"] == "plan":
            plan_steps[s["span_id"]] = set(s["attrs"].get("agent_runtime.plan.step_ids", []))
    unfinished_ok = run["status"] in ("cancelled", "budget_exhausted", "failed")

    def in_stopped_child(span: dict[str, Any]) -> bool:
        """Inside a subagent that did not end ok (its last turn's calls may be unanswered)."""
        p = by_id.get(span["parent_id"])
        while p is not None:
            if p["kind"] == "subagent" and p["status"] != "ok":
                return True
            p = by_id.get(p["parent_id"])
        return False

    for s in spans:
        if s["kind"] != "model" or s["status"] != "ok":
            continue
        resp = s.get("response")
        if resp is None:
            continue
        for call in resp.get("tool_calls", []):
            n = answers.get((s["span_id"], call["id"]), 0)
            if n > 1 or (n == 0 and not unfinished_ok and not in_stopped_child(s)):
                raise TraceInvalid(f"tool call {call['id']} of {s['span_id']} is answered {n} times")
    for (origin, call_id), _n in answers.items():
        o = by_id.get(origin)
        if o is None:
            raise TraceInvalid(f"a tool span answers a call of unknown span {origin}")
        if o["kind"] == "plan":
            if call_id not in plan_steps.get(origin, set()):
                raise TraceInvalid(f"tool call {call_id} is not a step of plan {origin}")
        elif o["kind"] == "model":
            resp = o.get("response")
            if resp is not None and call_id not in {c["id"] for c in resp.get("tool_calls", [])}:
                raise TraceInvalid(f"tool call {call_id} does not appear in model span {origin}")
        elif o["kind"] != "step":
            raise TraceInvalid(f"tool call {call_id} answers a {o['kind']} span")
