"""Check 13: the derivation of the model's input from the log, the context policies, compaction
with hysteresis, fork, and the rebuild of every request from the stored session log."""

import json
import pathlib

import pytest

from agent_runtime.canon import digest
from agent_runtime.context.derive import (
    STUB_PREFIX,
    SimCounter,
    apply_log,
    derive_messages,
    summary_request_messages,
)
from agent_runtime.evals.simrun import Job, run_job
from agent_runtime.models.types import Message, ToolCall
from agent_runtime.runtime.engine import ContextConfig, RunConfig
from agent_runtime.store.memory import MemoryStore
from agent_runtime.store.sqlite import SqliteStore
from agent_runtime.telemetry.trace import load_jsonl
from helpers import ScriptedEnv, call, kinds, run_scripted, u

C = SimCounter(250)


def make_log(n_exchanges: int, *, extra_user: bool = False) -> list[dict]:
    store = MemoryStore()
    sid = store.create_session()

    def add(m: Message):
        store.append_event(
            sid, {"type": "message", "conv": "main", "message": m.model_dump(mode="json", exclude_none=True)}
        )

    add(Message(role="system", content="sys"))
    if extra_user:
        add(Message(role="user", content="old question"))
        add(Message(role="assistant", content="old answer"))
    add(Message(role="user", content="task"))
    for i in range(n_exchanges):
        c = ToolCall(id=f"c{i}", name="get_order", arguments={"order_id": f"O{i:04d}"})
        add(Message(role="assistant", tool_calls=[c]))
        add(Message(role="tool", tool_call_id=c.id, content='{"ok":true,"result":{}}'))
    return store.events(sid)


def check_rules(msgs: list[Message]) -> None:
    assert msgs[0].role == "system"
    assert any(m.role == "user" and m.content == "task" for m in msgs)
    calls = {c.id for m in msgs if m.role == "assistant" for c in m.tool_calls}
    answered = {m.tool_call_id for m in msgs if m.role == "tool"}
    assert calls == answered  # no call without its result and no result without its call


def compact(events, policy, budget, **kw):
    """Derive; if a compaction is proposed, apply it as a logged event and derive again."""
    d = derive_messages(events, policy, budget, C, **kw)
    if d.proposal is None:
        return d, events
    if policy == "summarize":
        d.proposal.replacement["summary"] = "[summary] facts: none"
    ev = {**d.proposal.to_event(), "seq": events[-1]["seq"] + 1, "id": "k-x", "conv": "main"}
    events = events + [ev]
    return derive_messages(events, policy, budget, C, **kw), events


@pytest.mark.parametrize("policy", ["window", "elide", "summarize"])
def test_policies_fit_the_budget_and_keep_the_rules(policy):
    events = make_log(10)  # 500 + 10 * 250 = 3000 tokens
    budget = 3200
    d0 = derive_messages(events, policy, budget, C)
    assert d0.tokens == 3000 and d0.proposal is not None  # above compact_at * budget = 2400
    d, ev2 = compact(events, policy, budget)
    assert d.tokens <= 0.5 * budget  # lands at or below compact_to * budget
    check_rules(d.messages)
    # the last keep_last (2) exchanges are whole
    tail = d.messages[-4:]
    assert [m.role for m in tail] == ["assistant", "tool", "assistant", "tool"]
    assert all(not m.content.startswith(STUB_PREFIX) for m in tail)
    assert derive_messages(ev2, policy, budget, C).proposal is None


def test_none_never_compacts_and_its_irreducible_part_is_everything():
    events = make_log(10)
    d = derive_messages(events, "none", 1000, C)
    assert d.proposal is None and d.irreducible == d.tokens == 3000


def test_hysteresis_compacts_only_above_compact_at():
    events = make_log(7)  # 2250 tokens
    assert derive_messages(events, "window", 3200, C).proposal is None  # 2250 <= 0.75 * 3200 = 2400
    assert derive_messages(make_log(8), "window", 3200, C).proposal is not None  # 2500 > 2400


def test_stubs_name_their_call_and_tell_the_model_to_repeat_it():
    d, _ = compact(make_log(10), "elide", 3200)
    stubs = [m for m in d.messages if m.role == "tool" and m.content.startswith(STUB_PREFIX)]
    assert stubs
    assert (
        "get_order" in stubs[0].content
        and '"order_id":"O0000"' in stubs[0].content
        and "call c0" in stubs[0].content
    )
    assert "call the tool again" in stubs[0].content


def test_keep_last_exchanges_are_given_up_oldest_first_when_the_budget_forces_it():
    events = make_log(4)  # 1500 tokens; keep_last 3 protects 3 exchanges (incl. the one in progress)
    budget = 1100
    d, _ = compact(events, "window", budget, keep_last=3)
    assert d.tokens <= budget
    check_rules(d.messages)
    ids = [c.id for m in d.messages for c in m.tool_calls]
    assert ids[-1] == "c3" and "c0" not in ids


def test_window_drops_older_user_turns_but_never_the_newest_user_message():
    events = make_log(9, extra_user=True)
    d, _ = compact(events, "window", 3000)
    contents = [m.content for m in d.messages]
    assert "task" in contents and "old question" not in contents


def test_irreducible_part_that_fits_no_window_ends_the_step_with_context_length():
    senv = ScriptedEnv({"p": [{"text": "x", "usage": u()}]})
    for policy in ("none", "elide"):
        res = run_scripted(
            senv, "chat", {"message": "y" * 40_000}, config=RunConfig(context=ContextConfig(policy=policy))
        )
        assert (res.status, res.failure_reason, res.meter.provider_calls) == ("failed", "context_length", 0)


class RoleAwareModel:
    """Answers ``summarize`` calls with a summary and act calls with tool calls, then a final text."""

    def __init__(self, n_calls: int = 14) -> None:
        self.n = n_calls
        self.acts = 0

    async def complete(self, request, spec):
        from agent_runtime.providers.scripted import response_from_item

        if request.meta.get("role") == "summarize":
            return response_from_item(
                {"text": "[summary] facts: earlier lookups", "usage": u(80, 20)}, request, spec
            )
        self.acts += 1
        if self.acts > self.n:
            return response_from_item({"text": "done", "usage": u()}, request, spec)
        return response_from_item(
            {"tool_calls": [call(f"c{self.acts}", key="beta")], "usage": u()}, request, spec
        )


def test_summarize_is_a_billed_model_span_and_the_compacted_run_replays():
    from helpers import spec as mkspec

    class Env2(ScriptedEnv):
        def __call__(self, run_spec=None, clock=None):
            env = super().__call__(run_spec, clock)
            env.adapters = {"p": RoleAwareModel()}
            return env

    senv = Env2({"p": []}, specs=[mkspec(provider="p", window=700)])
    res = run_scripted(
        senv, config=RunConfig(context=ContextConfig(policy="summarize"), max_output_tokens=100)
    )
    summ = [m for m in kinds(res, "model") if m["attrs"]["agent_runtime.role"] == "summarize"]
    assert res.status == "succeeded"
    assert summ and all(m["attrs"]["agent_runtime.cost_ucu"] > 0 for m in summ)
    ctx = kinds(res, "context")
    assert ctx and all(
        c["attrs"]["agent_runtime.context.tokens_after"] <= c["attrs"]["agent_runtime.context.budget"]
        for c in ctx
    )
    assert res.meter.cost_ucu == sum(m["attrs"]["agent_runtime.cost_ucu"] for m in kinds(res, "model"))


def test_between_compactions_each_input_extends_the_one_before():
    out = run_job(
        Job(
            "t",
            "x",
            4,
            n_tasks=30,
            result_tokens=1000,
            router={"type": "fixed", "target": "sim-a/medium"},
            run_config={"context": {"policy": "elide"}},
            keep_traces=True,
        )
    )
    checked = 0
    for text in out["traces"].values():
        spans = load_jsonl(text)
        prev = None
        for s in spans:
            if s["kind"] == "context":
                prev = None
            elif (
                s["kind"] == "model"
                and s["status"] == "ok"
                and s["attrs"]["agent_runtime.role"] != "summarize"
            ):
                msgs = s["request"]["messages"]
                if prev is not None and s["attrs"]["agent_runtime.context.conv"] == prev[0]:
                    assert msgs[: len(prev[1])] == prev[1]
                    checked += 1
                prev = (s["attrs"]["agent_runtime.context.conv"], msgs)
    assert checked > 50


def test_every_request_of_a_100_run_batch_is_rebuilt_from_the_stored_log(tmp_path):
    jobs = []
    for i, policy in enumerate(["none", "window", "elide", "summarize"]):
        jobs.append(
            Job(
                "rebuild",
                policy,
                30 + i,
                n_tasks=25,
                result_tokens=1500 if policy != "none" else 250,
                router={"type": "fixed", "target": "sim-a/small", "fallback": ["sim-a/medium"]},
                run_config={"context": {"policy": policy}},
                keep_traces=True,
                store_path=str(tmp_path / f"{policy}.db"),
            )
        )
    rebuilt = 0
    for job in jobs:
        out = run_job(job)
        store = SqliteStore(job.store_path)
        try:
            for text in out["traces"].values():
                spans = load_jsonl(text)
                sid = spans[0]["attrs"]["agent_runtime.session_id"]
                events = store.events(sid)
                for s in spans:
                    if s["kind"] != "model":
                        continue
                    a = s["attrs"]
                    conv_events = [
                        e
                        for e in events
                        if e.get("conv", "main") == a["agent_runtime.context.conv"]
                        and e["seq"] <= a["agent_runtime.context.log_seq"]
                    ]
                    if a["agent_runtime.role"] == "summarize":
                        msgs = summary_request_messages(conv_events, a["agent_runtime.context.summarizes"])
                    else:
                        d = derive_messages(
                            conv_events,
                            a["agent_runtime.context.policy"],
                            a["agent_runtime.context.budget"],
                            SimCounter(job.result_tokens),
                        )
                        assert d.proposal is None
                        msgs = d.messages
                        assert (
                            digest([m.model_dump(mode="json", exclude_none=True) for m in msgs])
                            == a["agent_runtime.context.messages_sha256"]
                        )
                    req = dict(
                        s["request"], messages=[m.model_dump(mode="json", exclude_none=True) for m in msgs]
                    )
                    assert digest(req) == s["request_sha256"]
                    rebuilt += 1
        finally:
            store.close()
    assert len([1 for j in jobs]) == 4 and rebuilt > 300


def test_a_fork_derives_the_same_input_as_the_original_did_at_that_point(tmp_path):
    store = SqliteStore(str(tmp_path / "f.db"))
    try:
        r1 = run_scripted(
            ScriptedEnv({"p": [{"text": "a1", "usage": u()}]}),
            "chat",
            {"message": "q1"},
            store=store,
            run_id="r-1",
        )
        sid = r1.extra["session_id"]
        k = len(store.events(sid))  # system, q1, a1
        orig = ScriptedEnv({"p": [{"text": "a2", "usage": u()}]})
        run_scripted(
            orig, "chat", {"message": "q2"}, store=store, session_id=sid, run_id="r-2", check_replay=False
        )
        fork = store.fork(sid, k)
        assert (
            derive_messages(store.events(fork), "none", 10_000, C).messages
            == derive_messages(store.events(sid)[:k], "none", 10_000, C).messages
        )
        cont = ScriptedEnv({"p": [{"text": "a2", "usage": u()}]})
        run_scripted(
            cont, "chat", {"message": "q2"}, store=store, session_id=fork, run_id="r-3", check_replay=False
        )
        a = orig.models["p"].requests[0].messages
        b = cont.models["p"].requests[0].messages
        assert a == b
        assert [m.content for _, m in apply_log(store.events(fork))] == [a[0].content, "q1", "a1", "q2", "a2"]
    finally:
        store.close()


def test_derivation_is_pure():
    events = make_log(10)
    a = derive_messages(events, "elide", 3200, C)
    b = derive_messages(json.loads(json.dumps(events)), "elide", 3200, C)
    assert a.messages == b.messages and a.proposal == b.proposal
    assert pathlib.Path  # no I/O, no clock: the module imports neither
    import agent_runtime.context.derive as mod

    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert (
        "import time" not in src
        and "import random" not in src
        and "agent_runtime.rand" not in src
        and "open(" not in src
    )
