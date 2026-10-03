"""Run the scripted demo agent in a temporary workspace (no service needed).

    python examples/agent/run_agent.py

The script in examples/agent/scripts/demo.json is data that stands in for a model: the example
shows the harness (workspace tools, the permission layer, a subagent, the trace, and replay) and
says nothing about any model. The permission layer is policy inside this process, not isolation
by the operating system.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent_runtime.agents import AgentSpec, delegate_tool  # noqa: E402
from agent_runtime.clock import SystemClock  # noqa: E402
from agent_runtime.models import ModelRegistry, ModelSpec  # noqa: E402
from agent_runtime.permissions import PermissionPolicy  # noqa: E402
from agent_runtime.providers.scripted import ScriptedModel  # noqa: E402
from agent_runtime.runtime.engine import Env, RunConfig  # noqa: E402
from agent_runtime.runtime.runner import RunSpec, execute, replay_spans  # noqa: E402
from agent_runtime.store.memory import MemoryStore  # noqa: E402
from agent_runtime.telemetry.trace import validate_trace  # noqa: E402
from agent_runtime.tools import ToolRegistry  # noqa: E402
from agent_runtime.workspace import Workspace  # noqa: E402

SCRIPT = ROOT / "examples" / "agent" / "scripts" / "demo.json"
CATALOG = {
    "lookup": AgentSpec(
        name="lookup", description="reads files and answers", tools=["fs_read", "fs_list"], max_steps=4
    )
}
POLICY = PermissionPolicy(sandbox="workspace_write", approval="never", allow_exec=[["python", "-c"]])
SPEC = ModelSpec(
    id="demo-script",
    provider="scripted",
    wire="scripted",
    price_in_ucu=1_000_000,
    price_out_ucu=4_000_000,
    context_tokens=32_000,
)


def make_env(ws_root: str, clock) -> Env:
    ws = Workspace(ws_root, clock=clock)
    tools = ToolRegistry(ws.tools() + [delegate_tool(CATALOG)])
    return Env(
        registry=ModelRegistry([SPEC]),
        adapters={"scripted": ScriptedModel.from_file(SCRIPT, clock=clock)},
        tools=tools,
        policy=POLICY,
        workspace=ws,
        agents=CATALOG,
    )


async def run_demo(tmp: str):
    ws_root = os.path.join(tmp, "workspace")
    os.makedirs(ws_root)
    clock = SystemClock()
    env = make_env(ws_root, clock)
    store = MemoryStore()
    spec = RunSpec(
        run_id="r-1",
        workflow="agent",
        input={"task": "Review notes.txt"},
        default_target="scripted/demo-script",
        config=RunConfig(),
        clock="system",
    )
    res = await execute(spec, env, clock, store, store.create_session())
    await env.workspace.close()
    spans = res.trace.dicts()
    validate_trace(spans)
    print(
        f"status={res.status} model_calls={res.meter.model_calls} tool_calls={res.meter.tool_calls} subagents={res.meter.subagent_runs}"
    )
    for s in spans:
        a = s["attrs"]
        if s["kind"] == "permission":
            print(
                f"  permission {a['gen_ai.tool.name']:12s} {a['agent_runtime.permission.decision']:5s} rule={a['agent_runtime.permission.rule']}"
            )
        elif s["kind"] == "tool":
            r = s.get("result", {})
            print(
                f"  tool       {s['name']:12s} {'ok' if r.get('ok') else 'error ' + str(r.get('error', {}).get('code'))}"
            )
        elif s["kind"] == "subagent":
            print(
                f"  subagent   {a['agent_runtime.agent.name']} depth={a['agent_runtime.agent.depth']} steps={a.get('agent_runtime.agent.steps')}"
            )
    print("answer:", res.text)
    print("escape.txt exists outside the workspace:", os.path.exists(os.path.join(tmp, "escape.txt")))
    return res, spans, ws_root


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent-demo-") as tmp:
        res, spans, ws_root = asyncio.run(run_demo(tmp))
        rep = replay_spans(spans, lambda s: make_env(ws_root, SystemClock()))
        print(
            f"replay: divergence={rep['divergence']} provider_calls={rep['provider_calls']} tool_calls={rep['tool_calls']}"
        )
        return 0 if res.status == "succeeded" and rep["divergence"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
