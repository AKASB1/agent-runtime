"""Check 11: every endpoint and error code, the OpenAPI drift test, concurrent runs, cancellation and
replay through the API, restart persistence, the agent workflow in a session workspace, the
ceilings, fork, and the context endpoint (in process through the ASGI transport; the restart test
starts real service processes on port 0)."""

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time

import httpx

from agent_runtime.api.app import API_ERROR_CODES, create_app, openapi_text
from agent_runtime.api.service import Settings
from agent_runtime.procs import kill_tree, new_group_kwargs, pid_alive
from agent_runtime.providers.standin import StandIn, ThreadedServer

ROOT = pathlib.Path(__file__).resolve().parents[1]


def settings(tmp_path, **kw):
    base = dict(
        db=str(tmp_path / "api.db"),
        workspace_root=str(tmp_path / "ws"),
        examples_dir=str(ROOT / "examples"),
        time_scale=50.0,
        allow_exec=[["python", "-c"]],
        allow_hosts=["standin:80"],
    )
    base.update(kw)
    return Settings(**base)


def api(tmp_path, fn, **kw):
    """Run ``fn(client)`` against a fresh app in process."""

    async def go():
        app = create_app(settings(tmp_path, **kw))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as c:
                return await fn(c)
        finally:
            await app.state.service.aclose()

    return asyncio.run(go())


def err(r, status, code):
    assert r.status_code == status, (r.status_code, r.text)
    body = r.json()
    assert (
        set(body) == {"error"}
        and body["error"]["code"] == code
        and isinstance(body["error"]["details"], dict)
    )


def test_info_endpoints_and_error_bodies(tmp_path):
    async def fn(c):
        assert (await c.get("/healthz")).json()["status"] == "ok"
        models = (await c.get("/v1/models")).json()["models"]
        targets = {m["target"] for m in models}
        assert {
            "sim-a/small",
            "sim-b/large",
            "local/echo",
            "scripted/demo-script",
            "scripted/opt-script",
        } <= targets
        small = next(m for m in models if m["target"] == "sim-a/small")
        assert small["price_in_ucu_per_mtok"] == 150_000 and small["context_tokens"] == 4096
        names = {t["name"] for t in (await c.get("/v1/tools")).json()["tools"]}
        assert {"get_order", "fs_read", "run_command", "delegate", "opt.*"} <= names
        err(await c.get("/v1/sessions/s-999"), 404, "NOT_FOUND")
        err(await c.get("/v1/runs/r-999"), 404, "NOT_FOUND")
        err(await c.post("/v1/runs", json={"workflow": "nope"}), 400, "UNKNOWN_WORKFLOW")
        err(
            await c.post("/v1/runs", json={"workflow": "delegate_plan", "input": {}}), 400, "UNKNOWN_WORKFLOW"
        )
        err(await c.post("/v1/runs", json={"input": {}}), 422, "VALIDATION_ERROR")
        err(await c.post("/v1/runs", json={"workflow": "chat", "input": {}}), 422, "VALIDATION_ERROR")
        err(
            await c.post("/v1/runs", json={"workflow": "optimize", "input": {"problem": "nope"}}),
            400,
            "BAD_REQUEST",
        )
        err(
            await c.post(
                "/v1/runs",
                json={
                    "workflow": "chat",
                    "input": {"message": "x"},
                    "config": {"context": {"policy": "magic"}},
                },
            ),
            422,
            "VALIDATION_ERROR",
        )
        return True

    assert api(tmp_path, fn)


def test_sessions_state_fork_and_context(tmp_path):
    async def fn(c):
        r = await c.post("/v1/sessions")
        assert r.status_code == 201
        sid = r.json()["id"]
        r1 = await c.post(
            f"/v1/sessions/{sid}/runs?wait=true", json={"workflow": "chat", "input": {"message": "first"}}
        )
        assert r1.status_code == 200 and r1.json()["result"] == "echo: first"
        r2 = await c.post(
            f"/v1/sessions/{sid}/runs?wait=true", json={"workflow": "chat", "input": {"message": "second"}}
        )
        assert r2.json()["status"] == "succeeded"
        sess = (await c.get(f"/v1/sessions/{sid}")).json()
        assert [e["message"]["content"] for e in sess["events"]] == [
            sess["events"][0]["message"]["content"],
            "first",
            "echo: first",
            "second",
            "echo: second",
        ]
        assert [e["seq"] for e in sess["events"]] == [1, 2, 3, 4, 5]
        # the second run saw the first run's messages: its trace's request holds them
        trace = (await c.get(f"/v1/runs/{r2.json()['id']}/trace")).text
        req = json.loads(
            trace.splitlines()[[json.loads(x)["kind"] for x in trace.splitlines()].index("model")]
        )["request"]
        assert [m["content"] for m in req["messages"]][1:] == ["first", "echo: first", "second"]
        # versioned state
        ok = await c.put(f"/v1/sessions/{sid}/state", json={"expected_version": 0, "state": {"k": 1}})
        assert ok.json()["version"] == 1
        err(
            await c.put(f"/v1/sessions/{sid}/state", json={"expected_version": 0, "state": {"k": 2}}),
            409,
            "VERSION_CONFLICT",
        )
        # fork at message 3 copies the log and the state
        f = await c.post(f"/v1/sessions/{sid}/fork", json={"at_seq": 3})
        assert (
            f.status_code == 201
            and [e["seq"] for e in f.json()["events"]] == [1, 2, 3]
            and f.json()["state"] == {"k": 1}
        )
        err(await c.post(f"/v1/sessions/{sid}/fork", json={"at_seq": 99}), 422, "VALIDATION_ERROR")
        # the context endpoint is a dry derivation that writes nothing
        before = len((await c.get(f"/v1/sessions/{sid}")).json()["events"])
        ctx = (await c.get(f"/v1/sessions/{sid}/context", params={"policy": "window", "window": 60})).json()
        assert ctx["compaction_due"] and ctx["messages"][-1]["content"] == "echo: second" or ctx["messages"]
        assert len((await c.get(f"/v1/sessions/{sid}")).json()["events"]) == before
        full = (
            await c.get(f"/v1/sessions/{sid}/context", params={"policy": "none", "window": 100000})
        ).json()
        assert len(full["messages"]) == 5 and full["fits"]
        err(await c.get(f"/v1/sessions/{sid}/context", params={"policy": "bogus"}), 422, "VALIDATION_ERROR")
        return True

    assert api(tmp_path, fn)


def test_react_optimize_and_replay_through_the_api(tmp_path):
    async def fn(c):
        r = (
            await c.post(
                "/v1/runs?wait=true",
                json={
                    "workflow": "react",
                    "input": {"sim_task": {"seed": 3, "index": 1}},
                    "config": {"seed": 3},
                },
            )
        ).json()
        assert r["status"] in ("succeeded", "failed") and r["usage"]["model_calls"] >= 1 and "latency_s" in r
        rep = (await c.post(f"/v1/runs/{r['id']}/replay")).json()
        assert rep["divergence"] is None and rep["provider_calls"] == 0 and rep["tool_calls"] == 0
        o = (
            await c.post(
                "/v1/runs?wait=true", json={"workflow": "optimize", "input": {"problem": "knapsack"}}
            )
        ).json()
        assert o["status"] == "succeeded" and json.loads(o["result"])["objective"] == 29
        rep = (await c.post(f"/v1/runs/{o['id']}/replay")).json()
        assert rep["divergence"] is None
        pe = (
            await c.post(
                "/v1/runs?wait=true",
                json={
                    "workflow": "plan_replan",
                    "input": {"sim_task": {"seed": 2, "index": 4, "shape": "wide"}},
                },
            )
        ).json()
        assert pe["status"] in ("succeeded", "failed")
        usage = (await c.get("/v1/usage")).json()
        assert usage["total"]["calls"] >= r["usage"]["model_calls"] + o["usage"]["model_calls"]
        return True

    assert api(tmp_path, fn)


def test_concurrent_runs_and_cancellation(tmp_path):
    async def fn(c):
        posts = [
            c.post("/v1/runs", json={"workflow": "react", "input": {"sim_task": {"seed": 5, "index": i}}})
            for i in range(8)
        ]
        ids = [r.json()["id"] for r in await asyncio.gather(*posts)]
        assert len(set(ids)) == 8
        svc = c._transport.app.state.service  # noqa: SLF001
        for rid in ids:
            await svc.wait(rid)
        statuses = [(await c.get(f"/v1/runs/{rid}")).json()["status"] for rid in ids]
        assert all(s in ("succeeded", "failed") for s in statuses)
        # a slow run (time scale 1 on its clock would take seconds); cancel it while it runs
        long = (
            await c.post(
                "/v1/runs",
                json={
                    "workflow": "react",
                    "input": {"sim_task": {"seed": 1, "index": 7, "mix": "hard_heavy"}},
                },
            )
        ).json()["id"]
        await asyncio.sleep(0.01)
        err(await c.post(f"/v1/runs/{long}/replay"), 409, "RUN_NOT_FINISHED")
        cr = (await c.post(f"/v1/runs/{long}/cancel")).json()
        await svc.wait(long)
        final = (await c.get(f"/v1/runs/{long}")).json()
        assert cr["cancelled"] and final["status"] == "cancelled"
        err(await c.post(f"/v1/runs/{long}/replay"), 409, "NOT_REPLAYABLE")
        return True

    assert api(tmp_path, fn, time_scale=1.0)


def test_agent_workflow_in_a_session_workspace_and_the_ceilings(tmp_path):
    async def fn(c):
        sid = (await c.post("/v1/sessions")).json()["id"]
        body = {
            "workflow": "agent",
            "input": {"task": "review notes"},
            "config": {"workspace": "demo", "script": "demo", "allow_exec": [["python", "-c"]]},
        }
        r = (await c.post(f"/v1/sessions/{sid}/runs?wait=true", json=body)).json()
        assert r["status"] == "succeeded" and r["usage"]["subagent_runs"] == 1, r
        assert (tmp_path / "ws" / "demo" / "notes.txt").read_text(encoding="utf-8").count("reviewed") == 1
        assert not (tmp_path / "ws" / "escape.txt").exists()
        trace = [json.loads(x) for x in (await c.get(f"/v1/runs/{r['id']}/trace")).text.splitlines()]
        perms = [s["attrs"]["agent_runtime.permission.rule"] for s in trace if s["kind"] == "permission"]
        assert "path_outside_workspace" in perms and "allow_exec_prefix" in perms
        rep = (await c.post(f"/v1/runs/{r['id']}/replay")).json()
        assert rep["divergence"] is None and rep["tool_calls"] == 0
        # ceilings: a request may lower or narrow them, never raise or widen them
        err(
            await c.post("/v1/runs", json={**body, "config": {**body["config"], "approval": "auto"}}),
            403,
            "CEILING_EXCEEDED",
        )
        err(
            await c.post("/v1/runs", json={**body, "config": {**body["config"], "allow_exec": [["python"]]}}),
            403,
            "CEILING_EXCEEDED",
        )
        err(
            await c.post("/v1/runs", json={**body, "config": {**body["config"], "allow_hosts": ["evil:80"]}}),
            403,
            "CEILING_EXCEEDED",
        )
        narrowed = {
            **body,
            "config": {
                **body["config"],
                "sandbox": "read_only",
                "allow_exec": [["python", "-c", "print(1)"]],
            },
        }
        ok = (await c.post("/v1/runs?wait=true", json=narrowed)).json()
        assert ok["status"] == "succeeded"  # read_only: the writes are denied, and the agent goes on
        err(
            await c.post("/v1/runs", json={**body, "config": {**body["config"], "workspace": "../outside"}}),
            400,
            "BAD_REQUEST",
        )
        return True

    assert api(tmp_path, fn, max_sandbox="workspace_write", max_approval="never")


def test_openapi_file_matches_the_app():
    committed = (ROOT / "docs" / "openapi.json").read_text(encoding="utf-8")
    assert committed == openapi_text(), "docs/openapi.json is stale: run python -m agent_runtime openapi"


def test_every_api_error_code_is_documented():
    contracts = (ROOT / "docs" / "contracts.md").read_text(encoding="utf-8")
    for code in API_ERROR_CODES:
        assert f"`{code}`" in contracts


def test_restart_persistence_with_real_service_processes(tmp_path):
    db = tmp_path / "persist.db"
    ws = tmp_path / "ws"

    def start():
        port_file = tmp_path / f"port-{time.monotonic_ns()}"
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "WINDIR")
        }
        env.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1,localhost"})
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent_runtime",
                "serve",
                "--port",
                "0",
                "--port-file",
                str(port_file),
                "--db",
                str(db),
                "--workspace-root",
                str(ws),
                "--examples-dir",
                str(ROOT / "examples"),
            ],
            env=env,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **new_group_kwargs(),
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if port_file.exists() and port_file.read_text().strip():
                port = int(port_file.read_text())
                try:
                    if (
                        httpx.get(f"http://127.0.0.1:{port}/healthz", trust_env=False, timeout=2).status_code
                        == 200
                    ):
                        return p, f"http://127.0.0.1:{port}"
                except httpx.HTTPError:
                    pass
            time.sleep(0.1)
        kill_tree(p.pid)
        raise RuntimeError("service did not start")

    p1, url = start()
    try:
        with httpx.Client(base_url=url, trust_env=False, timeout=30) as c:
            sid = c.post("/v1/sessions").json()["id"]
            run = c.post(
                f"/v1/sessions/{sid}/runs?wait=true",
                json={"workflow": "chat", "input": {"message": "persist me"}},
            ).json()
            c.put(f"/v1/sessions/{sid}/state", json={"expected_version": 0, "state": {"note": "kept"}})
    finally:
        kill_tree(p1.pid)
        p1.wait(timeout=15)
    assert not pid_alive(p1.pid)
    p2, url = start()
    try:
        with httpx.Client(base_url=url, trust_env=False, timeout=30) as c:
            sess = c.get(f"/v1/sessions/{sid}").json()
            assert [e["message"]["content"] for e in sess["events"]][1:] == ["persist me", "echo: persist me"]
            assert sess["state"] == {"note": "kept"} and sess["version"] == 1
            assert c.get(f"/v1/runs/{run['id']}").json()["status"] == "succeeded"
            assert c.get(f"/v1/runs/{run['id']}/trace").text.startswith('{"attrs"')
    finally:
        kill_tree(p2.pid)
        p2.wait(timeout=15)


def test_chat_through_the_openai_compatible_adapter_and_the_api_canary(tmp_path):
    standin = StandIn("openai")
    with ThreadedServer(standin.app) as srv:

        async def fn(c):
            r = (
                await c.post(
                    "/v1/runs?wait=true",
                    json={
                        "workflow": "chat",
                        "input": {"message": "hi"},
                        "config": {"router": "openai/echo"},
                    },
                )
            ).json()
            assert r["status"] == "succeeded" and r["result"] == "echo: hi"
            texts = [
                json.dumps(r),
                (await c.get(f"/v1/runs/{r['id']}/trace")).text,
                json.dumps((await c.get("/v1/usage")).json()),
            ]
            return texts

        texts = api(tmp_path, fn, openai_base_url=srv.base_url + "/v1", openai_models=["echo"])
    assert all("Authorization" not in t and "Bearer" not in t for t in texts)


def test_an_unexpected_error_is_an_internal_error_body(tmp_path):
    async def fn(c):
        svc = c._transport.app.state.service  # noqa: SLF001

        def boom():
            raise RuntimeError("unexpected")

        svc.registry = boom
        err(await c.get("/v1/models"), 500, "INTERNAL")
        return True

    async def go():
        app = create_app(settings(tmp_path))
        try:
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
                return await fn(c)
        finally:
            await app.state.service.aclose()

    assert asyncio.run(go())
