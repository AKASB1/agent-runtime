"""MCP server with the solver and the verifier of the worked example (stdio).

Tools: ``solve(formulation)``, ``verify(card, formulation, solution)``, ``check_report(card, report,
solution)``. Run: ``python -m agent_runtime.optimization.server``.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from agent_runtime.optimization import core

server = MCPServer("opt", log_level="CRITICAL")


@server.tool(description="Solve a formulation with SciPy milp (HiGHS). Returns status, objective, values.")
def solve(formulation: dict[str, Any]) -> dict[str, Any]:
    f, diags = core.check_structure(formulation, None)
    if f is None or diags:
        return {"status": "INVALID", "objective": None, "values": {}, "diagnostics": diags}
    return core.solve(f)


@server.tool(description="Verify a formulation and its solution against a problem card (checks V1-V4).")
def verify(card: dict[str, Any], formulation: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    sol = solution if solution.get("status") not in (None, "INVALID") else None
    out = core.verify(card, formulation, sol)
    return {"accepted": out["accepted"], "diagnostics": out["diagnostics"]}


@server.tool(description="Check that a report repeats the solver's objective and variable values (check V5).")
def check_report(card: dict[str, Any], report: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    return core.check_report(card, report, solution)


if __name__ == "__main__":
    server.run("stdio")
