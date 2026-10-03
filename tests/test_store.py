"""Check 9: sessions and state (append-only log, versioned state, restart persistence, races)."""

import inspect
import os
import subprocess
import sys
import threading

import pytest

from agent_runtime.store import MemoryStore, SqliteStore, Store, VersionConflict
from helpers import ScriptedEnv, run_scripted, u

SCRIPT = r"""
import sys
sys.path.insert(0, sys.argv[3])
from agent_runtime.store import SqliteStore
s = SqliteStore(sys.argv[1])
if sys.argv[2] == "write":
    sid = s.create_session({"who": "first process"})
    s.append_event(sid, {"type": "message", "conv": "main", "message": {"role": "user", "content": "hello"}})
    s.put_state(sid, {"k": 1}, 0)
    rid = s.create_run(sid, {"workflow": "chat", "status": "succeeded"})
    print(sid, rid)
else:
    sid, rid = sys.argv[4], sys.argv[5]
    evs = s.events(sid)
    state, version = s.get_state(sid)
    print(evs[0]["message"]["content"], state["k"], version, s.get_run(rid)["status"])
s.close()
"""


def src_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_messages_state_and_runs_survive_a_process_restart(tmp_path):
    db = str(tmp_path / "store.db")
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")}
    env["PYTHONUTF8"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, db, "write", src_dir()],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=True,
    )
    sid, rid = out.stdout.split()
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, db, "read", src_dir(), sid, rid],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=True,
    )
    assert out.stdout.split() == ["hello", "1", "1", "succeeded"]


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_store_basics_ids_seq_fork_and_version(kind, tmp_path):
    s = MemoryStore() if kind == "memory" else SqliteStore(str(tmp_path / "s.db"))
    try:
        sid = s.create_session()
        assert sid == "s-1"
        e1 = s.append_event(
            sid, {"type": "message", "conv": "main", "message": {"role": "user", "content": "a"}}
        )
        e2 = s.append_event(
            sid,
            {
                "type": "compaction",
                "conv": "main",
                "policy": "window",
                "covers_from": 1,
                "covers_to": 1,
                "replacement": {"drop": [1]},
                "tokens_before": 1,
                "tokens_after": 0,
            },
        )
        e3 = s.append_event(
            sid, {"type": "message", "conv": "main", "message": {"role": "user", "content": "b"}}
        )
        assert [(e["seq"], e["id"]) for e in (e1, e2, e3)] == [(1, "m-1"), (2, "k-1"), (3, "m-2")]
        assert [e["seq"] for e in s.events(sid, upto_seq=2)] == [1, 2]
        assert s.put_state(sid, {"x": 1}, 0) == 1
        with pytest.raises(VersionConflict):
            s.put_state(sid, {"x": 2}, 0)
        f = s.fork(sid, 2)
        assert [e["seq"] for e in s.events(f)] == [1, 2]
        assert s.get_state(f) == ({"x": 1}, 0)
        assert s.events(sid)[-1]["message"]["content"] == "b"
    finally:
        s.close()


def test_the_store_interface_has_no_update_or_delete_for_events():
    for cls in (Store, MemoryStore, SqliteStore):
        names = [n for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)]
        bad = [
            n
            for n in names
            if ("event" in n and any(w in n for w in ("update", "delete", "remove", "edit", "set")))
        ]
        assert bad == [], (cls, bad)


def test_two_concurrent_state_writes_with_the_same_version_exactly_one_wins(tmp_path):
    db = str(tmp_path / "race.db")
    setup = SqliteStore(db)
    sid = setup.create_session()
    setup.close()
    stores = [SqliteStore(db), SqliteStore(db)]  # two connections, as two processes would have
    barrier = threading.Barrier(2)
    outcomes = []

    def writer(i):
        barrier.wait()
        try:
            stores[i].put_state(sid, {"by": i}, 0)
            outcomes.append("ok")
        except VersionConflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for st in stores:
        st.close()
    assert sorted(outcomes) == ["conflict", "ok"]


def test_the_second_run_of_a_session_sees_the_first_runs_messages(tmp_path):
    store = SqliteStore(str(tmp_path / "sess.db"))
    try:
        senv = ScriptedEnv({"p": [{"text": "first answer", "usage": u()}]})
        r1 = run_scripted(senv, "chat", {"message": "first question"}, store=store, run_id="r-1")
        sid = r1.extra["session_id"]
        senv2 = ScriptedEnv({"p": [{"text": "second answer", "usage": u()}]})
        r2 = run_scripted(
            senv2,
            "chat",
            {"message": "second question"},
            store=store,
            session_id=sid,
            run_id="r-2",
            check_replay=False,
        )
        req = senv2.models["p"].requests[0]
        contents = [m.content for m in req.messages]
        assert contents[-4:] == ["first question", "first answer", "second question"][-3:] or contents[
            1:
        ] == ["first question", "first answer", "second question"]
        assert r2.status == "succeeded"
        assert [e["run_id"] for e in store.events(sid)] == ["r-1", "r-1", "r-1", "r-2", "r-2"]
    finally:
        store.close()
