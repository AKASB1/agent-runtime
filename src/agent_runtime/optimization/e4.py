"""E4a: the six problems end to end through the runtime and the MCP server with the scripted model,
on the system clock. One row per problem. The reference formulations are read here (and in tests),
never by the workflow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import agent_runtime.optimization.workflow  # noqa: F401  (registers the workflow)
from agent_runtime.clock import SystemClock
from agent_runtime.mcp.client import McpServer, python_server
from agent_runtime.models import ModelRegistry, ModelSpec
from agent_runtime.providers.scripted import RoleScriptedModel
from agent_runtime.runtime.engine import Env, RunConfig, RunResult
from agent_runtime.runtime.runner import RunSpec, execute
from agent_runtime.store.memory import MemoryStore
from agent_runtime.tools import ToolRegistry

PROBLEMS = ("product_mix", "diet", "transportation", "knapsack", "facility", "assignment")
#: the scripted target is priced like the simulated `medium` model; its token counts are estimates
OPT_SPEC = ModelSpec(
    id="opt-script",
    provider="scripted",
    wire="scripted",
    price_in_ucu=1_000_000,
    price_out_ucu=4_000_000,
    context_tokens=32_000,
)


def example_dir(root: Path) -> Path:
    return Path(root) / "examples" / "optimization"


def load_card(root: Path, pid: str) -> dict[str, Any]:
    return json.loads((example_dir(root) / "problems" / f"{pid}.json").read_text(encoding="utf-8"))


def load_script(root: Path, pid: str) -> dict[str, Any]:
    return json.loads((example_dir(root) / "scripts" / f"{pid}.json").read_text(encoding="utf-8"))


def scripted_model(script: dict[str, Any], clock: Any = None) -> RoleScriptedModel:
    return RoleScriptedModel(
        {
            "formulate": [{"text": c["text"]} for c in script["candidates"]],
            "report": [{"text": r["text"]} for r in script["reports"]],
        },
        clock=clock,
    )


def opt_server() -> McpServer:
    return McpServer(
        python_server(
            "opt",
            "agent_runtime.optimization.server",
            capability="none",
            side_effect="none",
            timeout_ms=30_000,
        )
    )


def opt_env(script: dict[str, Any], tools: list, clock: Any = None) -> Env:
    return Env(
        registry=ModelRegistry([OPT_SPEC]),
        adapters={"scripted": scripted_model(script, clock)},
        tools=ToolRegistry(list(tools)),
    )


async def run_problem(
    root: Path, pid: str, server: McpServer, store: Any = None, run_id: str | None = None
) -> RunResult:
    clock = SystemClock()
    store = store or MemoryStore()
    card, script = load_card(root, pid), load_script(root, pid)
    spec = RunSpec(
        run_id=run_id or f"r-e4.{pid}",
        workflow="optimize",
        input={"card": card},
        default_target=OPT_SPEC.target,
        config=RunConfig(),
        clock="system",
        env={"name": "optimize", "problem": pid},
    )
    return await execute(
        spec, opt_env(script, server.tools(), clock), clock, store, store.create_session({"problem": pid})
    )


def candidate_codes(res: RunResult) -> tuple[list[list[str]], list[list[str]]]:
    """Per formulation candidate and per report, the diagnostic codes in the order they arose."""
    cands: list[list[str]] = []
    reports: list[list[str]] = []
    for s in res.trace.dicts():
        a = s["attrs"]
        if s["kind"] == "model" and s["status"] == "ok" and a.get("agent_runtime.role") == "formulate":
            cands.append([])
        elif s["kind"] == "validate" and s["status"] == "error" and s["name"] == "formulate":
            cands[-1].append(a.get("agent_runtime.validation.error", "").split(":")[0])
        elif s["kind"] == "tool" and s["name"] == "opt.verify" and s.get("result", {}).get("ok"):
            cands[-1].extend(dict.fromkeys(d["code"] for d in s["result"]["result"]["diagnostics"]))
        elif s["kind"] == "tool" and s["name"] == "opt.check_report" and s.get("result", {}).get("ok"):
            reports.append(list(dict.fromkeys(d["code"] for d in s["result"]["result"]["diagnostics"])))
    return cands, reports


def row_for(root: Path, pid: str, res: RunResult) -> dict[str, Any]:
    ref = json.loads((example_dir(root) / "reference" / f"{pid}.json").read_text(encoding="utf-8"))
    script = load_script(root, pid)
    cands, reports = candidate_codes(res)
    obj = res.extra.get("objective")
    accepted = len(cands) if res.status == "succeeded" else 0
    mistake = script["candidates"][accepted - 1]["mistake"] if accepted else ""

    def fmt(groups: list[list[str]]) -> str:
        return " | ".join(",".join(g) if g else "accepted" for g in groups)

    return {
        "problem": pid,
        "kind": load_card(root, pid)["kind"],
        "status": res.status,
        "candidates_tried": len(cands),
        "candidate_codes": fmt(cands),
        "report_codes": fmt(reports),
        "accepted_candidate_mistake": mistake or "",
        "accepted_objective": round(obj, 6) if obj is not None else "",
        "reference_objective": round(ref["objective"], 6),
        "objective_matches": int(
            obj is not None and abs(obj - ref["objective"]) <= 1e-6 * max(1.0, abs(ref["objective"]))
        ),
        "model_calls": res.meter.model_calls,
        "tool_calls": res.meter.tool_calls,
        "cost_mcu_estimated": round(res.meter.cost_ucu / 1000.0, 3),
    }


def run_e4a(root: Path) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        server = opt_server()
        await server.start()
        try:
            rows = []
            for pid in PROBLEMS:
                res = await run_problem(root, pid, server)
                rows.append(row_for(root, pid, res))
            return rows
        finally:
            await server.close()

    return asyncio.run(go())
