"""Replay: re-execute a workflow from its trace with a recorded model, recorded tools, and a recorded
approver that answer from the trace, matched by (step path, ordinal within the step) and checked
against the request digests. Permission decisions are decided again on the recorded resolved
arguments and must agree. A replay makes zero provider calls and zero tool calls and must
reproduce the span sequence and the result; otherwise it raises ``ReplayDivergence`` naming the
first differing span."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_runtime.canon import canonical, digest
from agent_runtime.models.types import ModelResponse, ProviderError
from agent_runtime.permissions.policy import decide
from agent_runtime.tools import ToolResult

COMPARED_DIGESTS = ("request_sha256", "response_sha256", "args_sha256", "result_sha256")
#: store-assigned ids differ in a replay (it runs in a scratch session)
NOT_COMPARED = ("agent_runtime.session_id", "agent_runtime.context.compaction_id")


class ReplayDivergence(AssertionError):
    def __init__(self, span_id: str | None, detail: str) -> None:
        super().__init__(f"replay diverged at {span_id}: {detail}")
        self.span_id = span_id
        self.detail = detail

    def report(self) -> dict[str, Any]:
        return {"span_id": self.span_id, "detail": self.detail}


@dataclass
class Replayer:
    spans: list[dict[str, Any]]
    models: dict[tuple, dict] = field(default_factory=dict)
    tools: dict[tuple, dict] = field(default_factory=dict)
    replayed_model_calls: int = 0
    replayed_tool_calls: int = 0

    def __post_init__(self) -> None:
        for s in self.spans:
            a = s["attrs"]
            if s["kind"] == "model":
                if "request" not in s:
                    raise ValueError("payload capture was off: replay is unavailable")
                self.models[(a["agent_runtime.step.path"], "model", a["agent_runtime.ordinal"])] = s
            elif s["kind"] == "tool":
                key = (a["agent_runtime.step.path"], "tool", a["agent_runtime.ordinal"])
                self.tools.setdefault(key, {})["tool"] = s
        # permission spans belong to the tool call with the same path and call id
        by_call: dict[tuple[str, str], tuple] = {}
        for key, rec in self.tools.items():
            t = rec["tool"]
            by_call[(t["attrs"]["agent_runtime.step.path"], t["attrs"]["gen_ai.tool.call.id"])] = key
        self._perm_by_call: dict[tuple[str, str], list[dict]] = {}
        for s in self.spans:
            if s["kind"] == "permission":
                a = s["attrs"]
                self._perm_by_call.setdefault(
                    (a["agent_runtime.step.path"], a["gen_ai.tool.call.id"]), []
                ).append(s)

    async def model(
        self, run: Any, key: tuple, span: Any, payload: dict
    ) -> tuple[ModelResponse | None, ProviderError | None]:
        rec = self.models.get(key)
        if rec is None:
            raise ReplayDivergence(span.span_id, f"no recorded model call at {key}")
        if digest(payload) != rec["request_sha256"]:
            raise ReplayDivergence(rec["span_id"], "the model request differs from the recorded one")
        if rec["end_ms"] > rec["start_ms"]:
            await run.clock.sleep(rec["end_ms"] - rec["start_ms"])
        self.replayed_model_calls += 1
        resp = rec["response"]
        if "error" in resp:
            return None, ProviderError.from_dict(resp["error"])
        return ModelResponse.model_validate(resp), None

    async def tool(self, run: Any, key: tuple, call: Any, tool: Any, base_attrs: dict) -> ToolResult:
        """Answer a tool call from the trace, checking it against the current code where it can:
        the tool lookup and allowlist, argument validation, the capability (a permission span exactly
        when it is not ``none``), the permission decision on the recorded resolved arguments, and the
        argument digests of executed and of denied calls."""
        f = run.frame
        perms = self._perm_by_call.get((f.path, call.id), [])
        perm = perms.pop(0) if perms else None
        rec = self.tools.get(key, {}).get("tool")
        seen = run._seen_calls.setdefault(f.conv, set())
        sig = canonical([call.name, call.arguments])
        if sig in seen:
            run.meter.repeat_reads += 1
        executed = rec is not None and "agent_runtime.tool.side_effect" in rec["attrs"]
        sid = (rec or perm or {}).get("span_id")
        if call.name in run.env.tools:  # the replay environment knows this tool: check against the code
            if tool is None:
                if executed or perm is not None:
                    raise ReplayDivergence(sid, f"{call.name} is not in this agent's allowlist now")
            else:
                _, err = tool.validate_input(call.arguments)
                code = ((rec or {}).get("result") or {}).get("error", {}).get("code")
                recorded_invalid = (
                    rec is not None
                    and not executed
                    and perm is None
                    and code in ("INVALID_ARGUMENTS", "UNKNOWN_TOOL")
                )
                if (err is not None) != recorded_invalid:
                    raise ReplayDivergence(
                        sid, f"argument validation of {call.name} differs from the recorded call"
                    )
                if err is None and (tool.capability != "none") != (perm is not None):
                    raise ReplayDivergence(
                        sid,
                        f"permission decision presence for {call.name} differs (capability {tool.capability})",
                    )
                if (
                    perm is not None
                    and perm["attrs"]["agent_runtime.permission.capability"] != tool.capability
                ):
                    raise ReplayDivergence(
                        perm["span_id"], f"capability of {call.name} differs from the recorded one"
                    )
        elif perm is None and rec is None:
            raise ReplayDivergence(None, f"no recorded tool call at {key} ({call.name})")
        if perm is not None:
            pa = perm["attrs"]
            if pa.get("agent_runtime.permission.args_sha256") != digest(call.arguments):
                raise ReplayDivergence(perm["span_id"], "the arguments of the permission decision differ")
            d = decide(
                run.env.policy,
                pa["agent_runtime.permission.capability"],
                pa["agent_runtime.permission.resolved"],
            )
            recorded = pa["agent_runtime.permission.decision"]
            if d.rule != pa["agent_runtime.permission.rule"] or (
                d.decision != "ask" and d.decision != recorded
            ):
                raise ReplayDivergence(
                    perm["span_id"],
                    f"permission decision {d.decision}/{d.rule} != recorded {recorded}/{pa['agent_runtime.permission.rule']}",
                )
            ps = run.trace.start(
                "permission",
                "permission",
                f.span,
                {
                    "agent_runtime.permission.capability": pa["agent_runtime.permission.capability"],
                    "agent_runtime.permission.resolved": pa["agent_runtime.permission.resolved"],
                    "agent_runtime.permission.args_sha256": pa["agent_runtime.permission.args_sha256"],
                    "agent_runtime.step.path": f.path,
                    "agent_runtime.origin_span": base_attrs["agent_runtime.origin_span"],
                    "gen_ai.tool.call.id": call.id,
                    "gen_ai.tool.name": call.name,
                },
            )
            run.trace.end(
                ps,
                "ok",
                **{
                    "agent_runtime.permission.decision": recorded,
                    "agent_runtime.permission.rule": pa["agent_runtime.permission.rule"],
                    "agent_runtime.permission.approved_by": pa["agent_runtime.permission.approved_by"],
                },
            )
            if recorded != "allow":
                return ToolResult.error("PERMISSION_DENIED", f"denied by rule {d.rule}", rule=d.rule)
        if rec is None:
            # the live call stopped at the budget check after taking a concurrency slot: so must this one
            await run._sem.acquire()
            try:
                run.check_live()
            finally:
                run._sem.release()
            raise ReplayDivergence(None, f"no recorded tool call at {key} ({call.name})")
        if digest(call.arguments) != rec["args_sha256"]:
            raise ReplayDivergence(rec["span_id"], "the tool arguments differ from the recorded ones")
        ra = rec["attrs"]
        attrs = {
            **base_attrs,
            **{k: v for k, v in ra.items() if k not in base_attrs and not k.startswith("wall_")},
        }
        if executed:  # as in the live path: take a slot of the run's concurrency limit, then check the budget
            await run._sem.acquire()
            try:
                run.check_live()
            except BaseException:
                run._sem.release()
                raise
        try:
            span = run.trace.start("tool", call.name, f.span, attrs)
            run.trace.set_payload(span, "args", call.arguments)
            if rec["end_ms"] > rec["start_ms"]:
                await run.clock.sleep(rec["end_ms"] - rec["start_ms"])
            result = rec["result"]
            run.trace.set_payload(span, "result", result)
            run.trace.end(span, rec["status"])
        finally:
            if executed:
                run._sem.release()
        if executed:
            run.meter.tool_calls += 1
            run.meter.tool_retries += int(ra.get("agent_runtime.retry", 0))
            seen.add(sig)
        self.replayed_tool_calls += 1
        return ToolResult.from_dict(result)


def compare_spans(
    recorded: list[dict[str, Any]], replayed: list[dict[str, Any]], *, compare_times: bool
) -> None:
    for a, b in zip(recorded, replayed, strict=False):
        for k in ("kind", "name", "parent_id", "status"):
            if a[k] != b[k]:
                raise ReplayDivergence(a["span_id"], f"{k}: recorded {a[k]!r}, replayed {b[k]!r}")
        aa = {k: v for k, v in a["attrs"].items() if not k.startswith("wall_") and k not in NOT_COMPARED}
        ba = {k: v for k, v in b["attrs"].items() if not k.startswith("wall_") and k not in NOT_COMPARED}
        if aa != ba:
            diff = sorted(k for k in set(aa) | set(ba) if aa.get(k) != ba.get(k))
            raise ReplayDivergence(a["span_id"], f"attributes differ: {diff[:5]}")
        for k in COMPARED_DIGESTS:
            if a.get(k) != b.get(k):
                raise ReplayDivergence(a["span_id"], f"{k} differs")
        if compare_times and (a["start_ms"], a["end_ms"]) != (b["start_ms"], b["end_ms"]):
            raise ReplayDivergence(a["span_id"], "timing differs")
    if len(recorded) != len(replayed):
        longer = recorded if len(recorded) > len(replayed) else replayed
        sid = longer[min(len(recorded), len(replayed))]["span_id"]
        raise ReplayDivergence(sid, f"span count: recorded {len(recorded)}, replayed {len(replayed)}")
