"""``derive_messages(log, policy, budget)``: a pure function of the log, the policy, and the token
budget (no clock, no randomness, no I/O).

The log holds ``message`` events and ``compaction`` events. The derivation applies the logged
compactions in order; if the result exceeds ``compact_at * budget`` under a compacting policy it
also proposes the next compaction, which brings the input to at most ``compact_to * budget``
(hysteresis). The caller logs the proposal (a summary needs one model call first) and derives
again. Rules for every policy: the system prompt and the newest user message are never dropped; an
assistant message with tool calls and the tool messages answering it are kept or compacted
together; the last ``keep_last`` exchanges stay in full as far as the budget allows (given up one
at a time, oldest first); the exchange in progress (the newest exchange after the newest user
message) is never compacted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_runtime.canon import canonical
from agent_runtime.models.accounting import estimate_tokens
from agent_runtime.models.types import Message

POLICIES = ("none", "window", "elide", "summarize")
STUB_PREFIX = "[elided]"
SUMMARY_PREFIX = "[summary]"


class TokenCounter(Protocol):
    base: int
    summary_tokens: int

    def message_tokens(self, m: Message) -> int: ...


class EstimatorCounter:
    """Default for real adapters: UTF-8 bytes of the canonical message / 4, rounded up. Used only to
    decide about compaction; billing uses the provider's usage."""

    base = 0
    summary_tokens = 100

    def message_tokens(self, m: Message) -> int:
        return estimate_tokens(canonical(m.model_dump(exclude_none=True, exclude_defaults=True)))


class SimCounter:
    """The simulator's exact counter: 500 per prompt, plus ``result_tokens`` per tool result in full,
    20 per stub, and 100 per summary."""

    summary_tokens = 100

    def __init__(self, result_tokens: int = 250, base: int = 500) -> None:
        self.result_tokens = result_tokens
        self.base = base

    def message_tokens(self, m: Message) -> int:
        if m.role == "tool":
            return 20 if m.content.startswith(STUB_PREFIX) else self.result_tokens
        if m.role == "assistant" and m.content.startswith(SUMMARY_PREFIX):
            return 100
        return 0


def count(messages: list[Message], counter: TokenCounter) -> int:
    return counter.base + sum(counter.message_tokens(m) for m in messages)


@dataclass
class Compaction:
    policy: str
    covers_from: int
    covers_to: int
    replacement: dict[str, Any]
    tokens_before: int
    tokens_after: int

    def to_event(self) -> dict[str, Any]:
        return {
            "type": "compaction",
            "policy": self.policy,
            "covers_from": self.covers_from,
            "covers_to": self.covers_to,
            "replacement": self.replacement,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
        }


@dataclass
class Derived:
    messages: list[Message]
    tokens: int
    irreducible: int
    proposal: Compaction | None
    last_seq: int
    seqs: list[int] = field(default_factory=list)


def message_from_event(ev: dict[str, Any]) -> Message:
    return Message.model_validate(ev["message"])


def apply_log(events: list[dict[str, Any]]) -> list[tuple[int, Message]]:
    """Messages after applying the logged compactions, as (seq, message)."""
    entries: list[tuple[int, Message]] = []
    for ev in events:
        if ev["type"] == "message":
            entries.append((ev["seq"], message_from_event(ev)))
            continue
        if ev["type"] != "compaction":
            continue
        rep = ev["replacement"]
        if "drop" in rep:
            drop = set(rep["drop"])
            entries = [(s, m) for s, m in entries if s not in drop]
        elif "stubs" in rep:
            stubs = {int(k): v for k, v in rep["stubs"].items()}
            entries = [
                (s, m.model_copy(update={"content": stubs[s]}) if s in stubs else m) for s, m in entries
            ]
        elif "summarize" in rep:
            covered = set(rep["summarize"])
            out: list[tuple[int, Message]] = []
            placed = False
            for s, m in entries:
                if s in covered:
                    if not placed:
                        out.append((ev["covers_from"], Message(role="assistant", content=rep["summary"])))
                        placed = True
                    continue
                out.append((s, m))
            entries = out
    return entries


@dataclass
class _Unit:
    kind: str  # system | user | exchange | answer
    idx: list[int]  # positions in entries


def _units(entries: list[tuple[int, Message]]) -> list[_Unit]:
    units: list[_Unit] = []
    open_calls: set[str] = set()
    for i, (_s, m) in enumerate(entries):
        if m.role == "tool":
            if units and units[-1].kind == "exchange" and (m.tool_call_id in open_calls or not open_calls):
                units[-1].idx.append(i)
                continue
            units.append(_Unit("answer", [i]))  # orphan tool message: never produced by the runtime
            continue
        open_calls = set()
        if m.role == "system":
            units.append(_Unit("system", [i]))
        elif m.role == "user":
            units.append(_Unit("user", [i]))
        elif m.tool_calls:
            open_calls = {c.id for c in m.tool_calls}
            units.append(_Unit("exchange", [i]))
        else:
            units.append(_Unit("answer", [i]))
    return units


def _protected(units: list[_Unit]) -> set[int]:
    prot = {k for k, u in enumerate(units) if u.kind == "system"}
    newest_user = max((k for k, u in enumerate(units) if u.kind == "user"), default=-1)
    if newest_user >= 0:
        prot.add(newest_user)
    if units and units[-1].kind == "exchange" and len(units) - 1 > newest_user:
        prot.add(len(units) - 1)
    return prot


def _stub_text(call: Any) -> str:
    return (
        f"{STUB_PREFIX} result of {call.name}({canonical(call.arguments)}) call {call.id} was removed to save space; "
        "call the tool again if you need it"
    )


def derive_messages(
    events: list[dict[str, Any]],
    policy: str,
    budget_tokens: int,
    counter: TokenCounter,
    *,
    compact_at: float = 0.75,
    compact_to: float = 0.5,
    keep_last: int = 2,
) -> Derived:
    if policy not in POLICIES:
        raise ValueError(f"unknown context policy {policy!r}")
    entries = apply_log(events)
    msgs = [m for _, m in entries]
    toks = [counter.message_tokens(m) for m in msgs]
    total = counter.base + sum(toks)
    units = _units(entries)
    prot = _protected(units)
    last_seq = events[-1]["seq"] if events else 0
    irreducible = _floor(msgs, toks, total, units, prot, policy, counter)
    proposal = None
    if policy != "none" and total > compact_at * budget_tokens:
        proposal = _propose(
            entries, msgs, toks, total, units, prot, policy, budget_tokens, counter, compact_to, keep_last
        )
    return Derived(msgs, total, irreducible, proposal, last_seq, [s for s, _ in entries])


def _floor(msgs, toks, total, units, prot, policy, counter) -> int:  # noqa: ANN001
    """The part of the input that the policy may not drop (what decides whether a target is offered):
    ``none`` everything; ``window`` the protected units; ``summarize`` the protected units plus one
    summary of the rest; ``elide`` everything, with the tool results of every unprotected exchange
    replaced by stubs."""
    if policy == "none":
        return total
    protected = counter.base + sum(toks[i] for k in prot for i in units[k].idx)
    eligible = [k for k, u in enumerate(units) if k not in prot and u.kind != "system"]
    if policy == "window":
        return protected
    if policy == "summarize":
        return protected + (counter.summary_tokens if eligible else 0)
    saving = 0
    for k in eligible:
        u = units[k]
        if u.kind != "exchange":
            continue
        calls = {c.id: c for c in msgs[u.idx[0]].tool_calls}
        for i in u.idx:
            m = msgs[i]
            if m.role == "tool" and not m.content.startswith(STUB_PREFIX):
                stub = m.model_copy(update={"content": _stub_text(calls.get(m.tool_call_id, _Call(m)))})
                saving += toks[i] - counter.message_tokens(stub)
    return total - saving


def _propose(entries, msgs, toks, total, units, prot, policy, budget, counter, compact_to, keep_last):  # noqa: ANN001
    def unit_tokens(u: _Unit) -> int:
        return sum(toks[i] for i in u.idx)

    eligible: list[int] = []
    for k, u in enumerate(units):
        if k in prot or u.kind == "system":
            continue
        if policy == "elide":
            if u.kind != "exchange":
                continue
            if not any(msgs[i].role == "tool" and not msgs[i].content.startswith(STUB_PREFIX) for i in u.idx):
                continue
        eligible.append(k)
    if not eligible:
        return None
    exchanges = [k for k, u in enumerate(units) if u.kind == "exchange"]
    keep = set(exchanges[-keep_last:]) if keep_last > 0 else set()
    target = compact_to * budget

    def saving(k: int) -> int:
        u = units[k]
        if policy == "elide":
            s = 0
            calls = {c.id: c for c in msgs[u.idx[0]].tool_calls}
            for i in u.idx:
                m = msgs[i]
                if m.role == "tool" and not m.content.startswith(STUB_PREFIX):
                    stub = m.model_copy(update={"content": _stub_text(calls.get(m.tool_call_id, _Call(m)))})
                    s += toks[i] - counter.message_tokens(stub)
            return s
        return unit_tokens(u)

    chosen: list[int] = []
    cur = total
    summary_cost = counter.summary_tokens if policy == "summarize" else 0
    for k in [k for k in eligible if k not in keep]:
        if cur <= target:
            break
        if not chosen:
            cur += summary_cost
        chosen.append(k)
        cur -= saving(k)
    for k in [k for k in eligible if k in keep]:
        if cur <= budget:
            break
        if not chosen:
            cur += summary_cost
        chosen.append(k)
        cur -= saving(k)
    if not chosen:
        return None
    chosen.sort()
    seqs = sorted(entries[i][0] for k in chosen for i in units[k].idx)
    if policy == "window":
        rep: dict[str, Any] = {"drop": seqs}
    elif policy == "elide":
        stubs: dict[str, str] = {}
        for k in chosen:
            u = units[k]
            calls = {c.id: c for c in msgs[u.idx[0]].tool_calls}
            for i in u.idx:
                m = msgs[i]
                if m.role == "tool" and not m.content.startswith(STUB_PREFIX):
                    stubs[str(entries[i][0])] = _stub_text(calls.get(m.tool_call_id, _Call(m)))
        rep = {"stubs": stubs}
    else:
        rep = {"summarize": seqs, "summary": None}
    return Compaction(policy, seqs[0], seqs[-1], rep, total, max(cur, 0))


class _Call:
    """Fallback description of a call whose assistant message is not in the unit."""

    def __init__(self, m: Message) -> None:
        self.name = "tool"
        self.arguments: dict[str, Any] = {}
        self.id = m.tool_call_id or "?"


def summary_request_messages(events: list[dict[str, Any]], seqs: list[int]) -> list[Message]:
    """The input of a ``summarize`` call: a pure function of the log and the covered seqs."""
    covered = set(seqs)
    transcript = [
        m.model_dump(exclude_none=True, exclude_defaults=True) for s, m in apply_log(events) if s in covered
    ]
    return [
        Message(
            role="system",
            content=(
                "Summarize the following part of an agent transcript. Keep every fact a later step may need "
                f"(ids, numbers, results of tool calls with their call ids). Start with {SUMMARY_PREFIX}."
            ),
        ),
        Message(role="user", content=canonical(transcript)),
    ]
