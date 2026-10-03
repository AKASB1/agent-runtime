"""Check 14: workspace tools and the permission layer. Hostile paths and commands are judged by the
policy and the resolver, which run nothing; the commands that do run are harmless (the current
interpreter with -c) in a temporary workspace."""

import asyncio
import hashlib
import os
import sys

import httpx
import pytest

from agent_runtime.clock import SystemClock
from agent_runtime.models import ModelRegistry
from agent_runtime.permissions import DEFAULT_DENY_EXEC, PermissionPolicy, decide, settle
from agent_runtime.procs import pid_alive
from agent_runtime.providers.scripted import ScriptedModel
from agent_runtime.providers.standin import StandIn
from agent_runtime.runtime.engine import Env, RunConfig
from agent_runtime.runtime.runner import RunSpec, execute, replay_spans
from agent_runtime.store.memory import MemoryStore
from agent_runtime.telemetry.trace import validate_trace
from agent_runtime.tools import ToolRegistry
from agent_runtime.workspace import PathRejected, Workspace, resolve
from helpers import call, u
from helpers import spec as mkspec

PY = sys.executable


def tree_hash(root: str) -> str:
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


@pytest.fixture
def sandbox(tmp_path):
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "sub" / "a.txt").write_bytes(b"hello world\n")
    (ws / "twice.txt").write_bytes(b"x x\n")
    out = tmp_path / "outside"
    out.mkdir()
    (out / "secret.txt").write_text("secret", encoding="utf-8")
    return tmp_path, str(ws), str(out)


HOSTILE = [
    "../outside/secret.txt",
    "..\\outside\\secret.txt",
    "sub/../../outside/secret.txt",
    "./../outside",
    "/etc/passwd",
    "\\Windows\\win.ini",
    "C:\\Windows\\win.ini",
    "c:secret.txt",
    "D:/x",
    "\\\\server\\share\\x",
    "//server/share/x",
    "\\\\?\\C:\\Windows",
    "\\\\.\\PhysicalDrive0",
    "//?/C:/x",
    "~/secret",
    "~",
    "CON",
    "con.txt",
    "sub/NUL",
    "aux.log",
    "COM1",
    "lpt9.txt",
    "PRN",
    "a.txt:hidden",
    "sub/a.txt::$DATA",
    "file.",
    "dir /x",
    "sub/name .txt ",
    "",
    "a\x00b",
    "SUB/../../OUTSIDE/SECRET.TXT",
]


@pytest.mark.parametrize("path", HOSTILE)
def test_hostile_paths_are_denied_and_touch_nothing(sandbox, path):
    tmp, ws, out = sandbox
    before = tree_hash(str(tmp))
    with pytest.raises(PathRejected):
        resolve(ws, path)
    assert (
        decide(
            PermissionPolicy(),
            "fs_write",
            Workspace(ws, clock=SystemClock()).resolve_paths({"path": path}, ("path",)),
        ).rule
        == "path_outside_workspace"
    )
    assert tree_hash(str(tmp)) == before


def test_inside_paths_and_case_variants_resolve(sandbox):
    _, ws, _ = sandbox
    assert resolve(ws, "sub/a.txt").endswith(os.path.join("sub", "a.txt"))
    assert resolve(ws, "sub/../sub/./a.txt") == resolve(ws, "sub/a.txt")
    assert resolve(ws, "new/dir/file.txt").startswith(os.path.realpath(ws))
    if sys.platform == "win32":
        assert os.path.normcase(resolve(ws, "SUB/A.TXT")) == os.path.normcase(resolve(ws, "sub/a.txt"))


def test_symbolic_links_and_junctions_that_leave_the_workspace_are_denied(sandbox):
    _, ws, out = sandbox
    made = []
    try:
        os.symlink(out, os.path.join(ws, "link"), target_is_directory=True)
        made.append("link")
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        import _winapi

        try:
            _winapi.CreateJunction(out, os.path.join(ws, "junction"))
            made.append("junction")
        except OSError:
            pass
    if not made:
        pytest.skip("this platform/account can create neither symbolic links nor junctions")
    for name in made:
        with pytest.raises(PathRejected) as ei:
            resolve(ws, f"{name}/secret.txt")
        assert ei.value.reason == "outside_workspace"


# ---- the decision table under each approval mode ----------------------------------------------------

ROWS = [
    # (capability, resolved, read_only expected, workspace_write expected)
    ("fs_write", {"paths_ok": False}, ("deny", "path_outside_workspace"), ("deny", "path_outside_workspace")),
    (
        "exec",
        {"paths_ok": True, "argv": ["C:\\Windows\\System32\\cmd.EXE", "/c", "dir"]},
        ("deny", "deny_exec"),
        ("deny", "deny_exec"),
    ),
    (
        "exec",
        {"paths_ok": True, "argv": ["/bin/rm", "-rf", "x"]},
        ("deny", "deny_exec"),
        ("deny", "deny_exec"),
    ),
    ("none", {}, ("allow", "capability_allowed"), ("allow", "capability_allowed")),
    ("fs_read", {"paths_ok": True}, ("allow", "capability_allowed"), ("allow", "capability_allowed")),
    ("fs_write", {"paths_ok": True}, ("deny", "read_only_sandbox"), ("allow", "workspace_write")),
    (
        "exec",
        {"paths_ok": True, "argv": ["Python.exe", "-m", "pytest", "-q"]},
        ("deny", "read_only_sandbox"),
        ("allow", "allow_exec_prefix"),
    ),
    (
        "exec",
        {"paths_ok": True, "argv": ["python", "-c", "1"]},
        ("deny", "read_only_sandbox"),
        ("ask", "exec_needs_approval"),
    ),
    ("net", {"host": "standin:80"}, ("deny", "read_only_sandbox"), ("allow", "allow_hosts")),
    ("net", {"host": "example.org:443"}, ("deny", "read_only_sandbox"), ("ask", "net_needs_approval")),
]


@pytest.mark.parametrize("approval", ["ask", "auto", "never"])
@pytest.mark.parametrize("answer", [True, False, None])
@pytest.mark.parametrize("row", ROWS)
def test_every_row_of_the_decision_table_under_each_approval_mode(row, approval, answer):
    cap, resolved, ro, ww = row
    asked = []

    def approver(tool, res, rule):
        asked.append(rule)
        return answer

    for sandbox, (exp, rule) in (("read_only", ro), ("workspace_write", ww)):
        pol = PermissionPolicy(
            sandbox=sandbox,
            approval=approval,
            allow_exec=[["python", "-m", "pytest"]],
            allow_hosts=["standin:80"],
        )
        d = decide(pol, cap, resolved)
        assert (d.decision, d.rule) == (exp, rule)
        final, by = asyncio.run(settle(pol, d, approver if answer is not None else None, "t", resolved))
        if exp != "ask":
            assert final == exp and by is None
        elif approval == "auto":
            assert (final, by) == ("allow", "auto")
        elif approval == "never":
            assert (final, by) == ("deny", "never")
        elif answer is None:
            assert (final, by) == ("deny", "none")
        else:
            assert (final, by) == ("allow" if answer else "deny", "approver")
    # a final denial (outside, deny_exec, read_only) never reaches the approver
    if ww[0] != "ask":
        assert asked == []
    assert set(DEFAULT_DENY_EXEC) >= {"cmd", "powershell", "bash", "rm", "curl", "docker", "taskkill"}


# ---- through the engine ---------------------------------------------------------------------------


def agent_run(
    ws_root, script, *, policy=None, approver=None, http_transport=None, timeout_ms=10_000, window=8000
):
    async def go():
        clock = SystemClock()
        ws = Workspace(ws_root, clock=clock, http_transport=http_transport, timeout_ms=timeout_ms)
        model = ScriptedModel(script)
        env = Env(
            registry=ModelRegistry([mkspec(provider="p", window=window)]),
            adapters={"p": model},
            tools=ToolRegistry(ws.tools()),
            policy=policy or PermissionPolicy(),
            approver=approver,
            workspace=ws,
        )
        store = MemoryStore()
        rs = RunSpec(
            run_id="r-ws",
            workflow="agent",
            input={"task": "work"},
            default_target="p/m",
            clock="system",
            config=RunConfig(timeout_ms=timeout_ms),
        )
        try:
            return await execute(rs, env, clock, store, store.create_session()), ws
        finally:
            await ws.close()

    res, ws = asyncio.run(go())
    spans = res.trace.dicts()
    validate_trace(spans)
    return res, spans, ws


def results(spans):
    return {s["attrs"]["gen_ai.tool.call.id"]: s["result"] for s in spans if s["kind"] == "tool"}


def test_denied_calls_have_no_side_effect_and_return_permission_denied(sandbox):
    tmp, ws, out = sandbox
    before = tree_hash(out)
    script = [
        {"tool_calls": [call("c1", "fs_write", path="../outside/pwned.txt", content="x")], "usage": u()},
        {"tool_calls": [call("c2", "run_command", argv=["cmd", "/c", "del", "x"])], "usage": u()},
        {"tool_calls": [call("c3", "fs_read", path="sub/a.txt")], "usage": u()},
        {"text": "done", "usage": u()},
    ]
    asked = []
    res, spans, _ = agent_run(
        ws, script, policy=PermissionPolicy(approval="ask"), approver=lambda *a: asked.append(a) or True
    )
    assert tree_hash(out) == before and not os.path.exists(os.path.join(out, "pwned.txt"))
    perms = [s for s in spans if s["kind"] == "permission"]
    assert [
        (p["attrs"]["agent_runtime.permission.decision"], p["attrs"]["agent_runtime.permission.rule"])
        for p in perms
    ] == [("deny", "path_outside_workspace"), ("deny", "deny_exec"), ("allow", "capability_allowed")]
    assert asked == []  # the approver could not overturn a path escape or a deny_exec match
    tools = [s for s in spans if s["kind"] == "tool"]
    assert [t["attrs"]["gen_ai.tool.call.id"] for t in tools] == ["c3"]  # denied calls have no tool span
    assert results(spans)["c3"]["result"]["content"] == "hello world\n"
    model_spans = [s for s in spans if s["kind"] == "model"]
    denied_msg = model_spans[1]["request"]["messages"][-1]
    assert denied_msg["role"] == "tool" and "PERMISSION_DENIED" in denied_msg["content"]
    assert "path_outside_workspace" in denied_msg["content"]


def test_fs_tools_write_edit_list_and_edit_failures(sandbox):
    _, ws, _ = sandbox
    script = [
        {"tool_calls": [call("w1", "fs_write", path="new/deep/b.txt", content="one two")], "usage": u()},
        {"tool_calls": [call("w2", "fs_write", path="new/deep/b.txt", content="again")], "usage": u()},
        {"tool_calls": [call("e1", "fs_edit", path="new/deep/b.txt", old="two", new="2")], "usage": u()},
        {"tool_calls": [call("e2", "fs_edit", path="twice.txt", old="x", new="y")], "usage": u()},
        {"tool_calls": [call("e3", "fs_edit", path="sub/a.txt", old="absent", new="y")], "usage": u()},
        {"tool_calls": [call("l1", "fs_list", path="new/deep")], "usage": u()},
        {
            "tool_calls": [call("a1", "fs_write", path="new/deep/b.txt", content="!", mode="append")],
            "usage": u(),
        },
        {"text": "done", "usage": u()},
    ]
    res, spans, _ = agent_run(ws, script)
    r = results(spans)
    assert r["w1"]["ok"] and not r["w2"]["ok"]  # create refuses to overwrite
    assert (
        r["e1"]["ok"]
        and "occurs 2 times" in r["e2"]["error"]["message"]
        and "occurs 0 times" in r["e3"]["error"]["message"]
    )
    assert r["l1"]["result"]["entries"] == [{"name": "b.txt", "type": "file", "size": 5}]
    assert open(os.path.join(ws, "new", "deep", "b.txt"), encoding="utf-8").read() == "one 2!"
    with open(os.path.join(ws, "bin.dat"), "wb") as f:
        f.write(b"\x00\x01\x02")
    res, spans, _ = agent_run(
        ws,
        [{"tool_calls": [call("b", "fs_read", path="bin.dat")], "usage": u()}, {"text": "x", "usage": u()}],
    )
    assert "binary" in results(spans)["b"]["error"]["message"]


def test_run_command_env_allowlist_output_cap_and_approval(sandbox):
    _, ws, _ = sandbox
    os.environ["AR_TEST_CANARY_VARIABLE"] = "must-not-leak"
    try:
        script = [
            {
                "tool_calls": [
                    call("env", "run_command", argv=[PY, "-c", "import os; print(sorted(os.environ))"])
                ],
                "usage": u(),
            },
            {
                "tool_calls": [call("big", "run_command", argv=[PY, "-c", "print('x' * 100000)"])],
                "usage": u(),
            },
            {"text": "done", "usage": u()},
        ]
        pol = PermissionPolicy(allow_exec=[[os.path.basename(PY), "-c"]])
        res, spans, _ = agent_run(ws, script, policy=pol)
    finally:
        del os.environ["AR_TEST_CANARY_VARIABLE"]
    r = results(spans)
    names = r["env"]["result"]["stdout"]
    assert "AR_TEST_CANARY_VARIABLE" not in names and "must-not-leak" not in names
    # LC_CTYPE: a child Python started under the C/POSIX locale sets it itself (PEP 538), it is not passed on
    assert set(eval(names)) <= set(
        __import__("agent_runtime.workspace", fromlist=["x"]).DEFAULT_ENV_ALLOW
    ) | {"SYSTEMROOT", "PYTHONUTF8", "LC_CTYPE"}
    out = r["big"]["result"]["stdout"]
    assert (
        out.startswith("x" * 1000)
        and out.endswith("[... output truncated at 65536 bytes]")
        and len(out) < 65536 + 100
    )
    # without an allow_exec prefix the call needs approval; approval "never" denies it
    res, spans, _ = agent_run(ws, script[:1] + [{"text": "x", "usage": u()}])
    p = [s for s in spans if s["kind"] == "permission"][0]["attrs"]
    assert (
        p["agent_runtime.permission.decision"],
        p["agent_runtime.permission.rule"],
        p["agent_runtime.permission.approved_by"],
    ) == ("deny", "exec_needs_approval", "never")


def test_run_command_timeout_kills_the_process_tree_by_pid(sandbox, tmp_path):
    _, ws, _ = sandbox
    child_code = "import time; time.sleep(60)"
    parent = (
        "import subprocess, sys, time; "
        f"p = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "open('child.pid', 'w').write(str(p.pid)); time.sleep(60)"
    )
    script = [
        {"tool_calls": [call("t", "run_command", argv=[PY, "-c", parent], timeout_ms=1500)], "usage": u()},
        {"text": "x", "usage": u()},
    ]
    res, spans, wsobj = agent_run(ws, script, policy=PermissionPolicy(approval="auto"))
    t = [s for s in spans if s["kind"] == "tool"][0]
    assert t["attrs"]["error.type"] == "tool_timeout" and t["attrs"]["agent_runtime.retry"] == 0
    parent_pid = wsobj.pids[0]
    for _ in range(100):
        if os.path.exists(os.path.join(ws, "child.pid")):
            break
        asyncio.run(asyncio.sleep(0.05))
    child_pid = int(open(os.path.join(ws, "child.pid")).read())
    for _ in range(100):
        if not pid_alive(parent_pid) and not pid_alive(child_pid):
            break
        asyncio.run(asyncio.sleep(0.05))
    assert not pid_alive(parent_pid) and not pid_alive(child_pid)


def test_http_get_reaches_only_allowlisted_hosts_and_does_not_follow_redirects(sandbox):
    _, ws, _ = sandbox
    transport = httpx.ASGITransport(app=StandIn().app)
    script = [
        {"tool_calls": [call("ok", "http_get", url="http://standin/text/hello")], "usage": u()},
        {"tool_calls": [call("redir", "http_get", url="http://standin/redirect")], "usage": u()},
        {"tool_calls": [call("big", "http_get", url="http://standin/big")], "usage": u()},
        {"tool_calls": [call("other", "http_get", url="http://elsewhere/text/x")], "usage": u()},
        {"text": "done", "usage": u()},
    ]
    res, spans, _ = agent_run(
        ws,
        script,
        policy=PermissionPolicy(allow_hosts=["standin:80"]),
        http_transport=transport,
        window=200_000,
    )
    r = results(spans)
    assert r["ok"]["result"]["content"] == "page hello\n"
    assert r["redir"]["result"]["status"] == 302 and r["redir"]["result"]["location"] == "/text/target"
    assert r["big"]["result"]["content"].endswith("[... output truncated at 262144 bytes]")
    assert "other" not in r
    perms = [s["attrs"] for s in spans if s["kind"] == "permission"]
    assert len(perms) == 4  # every decision for a capability other than none is a span
    assert (
        perms[-1]["agent_runtime.permission.rule"] == "net_needs_approval"
        and perms[-1]["agent_runtime.permission.decision"] == "deny"
    )


def test_replay_reproduces_approver_decisions_without_calling_the_approver(sandbox):
    _, ws, _ = sandbox
    answers = iter([True, False])
    calls_made = []

    def approver(tool, resolved, rule):
        calls_made.append(rule)
        return next(answers)

    script = [
        {"tool_calls": [call("a", "run_command", argv=[PY, "-c", "print(1)"])], "usage": u()},
        {"tool_calls": [call("b", "run_command", argv=[PY, "-c", "print(2)"])], "usage": u()},
        {"text": "done", "usage": u()},
    ]
    res, spans, _ = agent_run(ws, script, policy=PermissionPolicy(approval="ask"), approver=approver)
    assert calls_made == ["exec_needs_approval"] * 2
    perms = [s["attrs"] for s in spans if s["kind"] == "permission"]
    assert [
        (p["agent_runtime.permission.decision"], p["agent_runtime.permission.approved_by"]) for p in perms
    ] == [("allow", "approver"), ("deny", "approver")]

    def factory(spec):
        w = Workspace(ws, clock=SystemClock())
        return Env(
            registry=ModelRegistry([mkspec(provider="p")]),
            adapters={"p": None},
            tools=ToolRegistry(w.tools()),
            policy=PermissionPolicy(approval="ask"),
            approver=lambda *a: calls_made.append("REPLAY") or True,
            workspace=w,
        )

    rep = replay_spans(spans, factory)
    assert rep["divergence"] is None and rep["tool_calls"] == 0 and rep["provider_calls"] == 0
    assert "REPLAY" not in calls_made
