"""End-to-end demo against real local processes.

Starts the stand-in provider (a test double of an OpenAI-compatible endpoint) and the service, both
on 127.0.0.1 and recorded by PID; runs a chat session across two runs (through the OpenAI-compatible
adapter), a react run on the simulated models, an optimize run (the worked example through its MCP
server), and an agent run in a session workspace with the scripted target (it writes, reads, and
edits a file, runs a harmless command, is denied a write outside the workspace, and delegates a
lookup to a child agent); forks the session, reads a trace, replays a run, prints /v1/usage, and
stops both processes by PID.

    python scripts/demo.py [--port 18500] [--standin-port 18510]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from agent_runtime.procs import kill_tree, new_group_kwargs, pid_alive  # noqa: E402


def free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_http(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, trust_env=False, timeout=2).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{url} did not answer within {timeout_s} s")


def child_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "LANG")
    }
    env.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1,localhost"})
    return env


def show(title: str, obj: object) -> None:
    print(f"\n== {title}")
    print(json.dumps(obj, indent=2)[:1500])


def main() -> int:
    p = argparse.ArgumentParser(description="agent-runtime demo")
    p.add_argument("--port", type=int, default=18500)
    p.add_argument("--standin-port", type=int, default=18510)
    args = p.parse_args()
    for port in (args.port, args.standin_port):
        if not free(port):
            print(f"port {port} is in use; pass --port/--standin-port", file=sys.stderr)
            return 2
    tmp = Path(tempfile.mkdtemp(prefix="agent-runtime-demo-"))
    logs = tmp / "logs"
    logs.mkdir()
    procs: list[subprocess.Popen] = []
    try:
        standin = subprocess.Popen(
            [sys.executable, "-m", "agent_runtime.providers.standin", "--port", str(args.standin_port)],
            cwd=ROOT,
            env=child_env(),
            stdout=open(logs / "standin.log", "w"),
            stderr=subprocess.STDOUT,
            **new_group_kwargs(),
        )
        procs.append(standin)
        service = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent_runtime",
                "serve",
                "--port",
                str(args.port),
                "--time-scale",
                "50",
                "--db",
                str(tmp / "demo.db"),
                "--workspace-root",
                str(tmp / "workspaces"),
                "--examples-dir",
                str(ROOT / "examples"),
                "--openai-base-url",
                f"http://127.0.0.1:{args.standin_port}/v1",
                "--openai-models",
                "echo",
                "--allow-exec",
                '["python", "-c"]',
            ],
            cwd=ROOT,
            env=child_env(),
            stdout=open(logs / "service.log", "w"),
            stderr=subprocess.STDOUT,
            **new_group_kwargs(),
        )
        procs.append(service)
        print(
            f"stand-in pid {standin.pid} on 127.0.0.1:{args.standin_port}; service pid {service.pid} on 127.0.0.1:{args.port}"
        )
        base = f"http://127.0.0.1:{args.port}"
        wait_http(f"http://127.0.0.1:{args.standin_port}/v1/models")
        wait_http(f"{base}/healthz")
        with httpx.Client(base_url=base, trust_env=False, timeout=120) as c:
            sid = c.post("/v1/sessions").json()["id"]
            r1 = c.post(
                f"/v1/sessions/{sid}/runs?wait=true",
                json={
                    "workflow": "chat",
                    "input": {"message": "What is a harness?"},
                    "config": {"router": "openai/echo"},
                },
            ).json()
            r2 = c.post(
                f"/v1/sessions/{sid}/runs?wait=true",
                json={
                    "workflow": "chat",
                    "input": {"message": "And a session log?"},
                    "config": {"router": "openai/echo"},
                },
            ).json()
            show(
                "chat session, two runs (OpenAI-compatible adapter -> stand-in)",
                {"session": sid, "run1": r1["result"], "run2": r2["result"]},
            )
            react = c.post(
                "/v1/runs?wait=true",
                json={
                    "workflow": "react",
                    "input": {"sim_task": {"seed": 1, "index": 3}},
                    "config": {"router": "sim-a/medium", "seed": 1},
                },
            ).json()
            show(
                "react run on the simulated models",
                {k: react[k] for k in ("id", "status", "result", "cost_ucu", "latency_s")},
            )
            opt = c.post(
                "/v1/runs?wait=true", json={"workflow": "optimize", "input": {"problem": "product_mix"}}
            ).json()
            show(
                "optimize run (scripted formulations, solver and verifier over MCP)",
                {k: opt[k] for k in ("id", "status", "result", "cost_ucu")},
            )
            agent = c.post(
                f"/v1/sessions/{sid}/runs?wait=true",
                json={
                    "workflow": "agent",
                    "input": {"task": "Review notes.txt"},
                    "config": {"workspace": "demo", "script": "demo", "allow_exec": [["python", "-c"]]},
                },
            ).json()
            trace = [json.loads(x) for x in c.get(f"/v1/runs/{agent['id']}/trace").text.splitlines()]
            perms = [
                (
                    s["attrs"]["gen_ai.tool.name"],
                    s["attrs"]["agent_runtime.permission.decision"],
                    s["attrs"]["agent_runtime.permission.rule"],
                )
                for s in trace
                if s["kind"] == "permission"
            ]
            show(
                "agent run in the session workspace 'demo'",
                {
                    "status": agent["status"],
                    "result": agent["result"],
                    "subagent_runs": agent["usage"]["subagent_runs"],
                    "permission decisions": perms,
                },
            )
            fork = c.post(f"/v1/sessions/{sid}/fork", json={"at_seq": 3}).json()
            show(
                "fork of the session at seq 3",
                {"new_session": fork["id"], "events": [e["message"]["content"][:40] for e in fork["events"]]},
            )
            kinds = {}
            for s in trace:
                kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
            show("trace of the agent run (span kinds)", kinds)
            rep = c.post(f"/v1/runs/{react['id']}/replay").json()
            show("replay of the react run", rep)
            show("/v1/usage", c.get("/v1/usage").json())
            ok = all(r["status"] == "succeeded" for r in (r1, r2, opt, agent)) and rep["divergence"] is None
    finally:
        for proc in reversed(procs):
            kill_tree(proc.pid)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        alive = [proc.pid for proc in procs if pid_alive(proc.pid)]
        print(f"\nstopped by PID: {[proc.pid for proc in procs]}; still alive: {alive}")
        shutil.rmtree(tmp, ignore_errors=True)
    print("demo ok" if ok else "demo FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
