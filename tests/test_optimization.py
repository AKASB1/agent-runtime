"""Check 12: the worked example end to end through MCP, the verifier's checks, and replay."""

import asyncio
import json
import pathlib

import pytest

from agent_runtime.optimization import core
from agent_runtime.optimization.e4 import (
    PROBLEMS,
    candidate_codes,
    load_card,
    load_script,
    opt_env,
    opt_server,
    row_for,
    run_problem,
)
from agent_runtime.procs import pid_alive
from agent_runtime.runtime.runner import replay_spans
from agent_runtime.telemetry.trace import validate_trace
from agent_runtime.tools import ToolRegistry

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_all_six_problems_end_to_end_through_mcp_with_the_scripted_codes_and_replay():
    async def go():
        server = opt_server()
        await server.start()
        try:
            return {pid: await run_problem(ROOT, pid, server) for pid in PROBLEMS}, server.pid
        finally:
            await server.close()

    results, pid = asyncio.run(go())
    assert not pid_alive(pid)
    blind_spots = 0
    for problem, res in results.items():
        spans = res.trace.dicts()
        validate_trace(spans)
        row = row_for(ROOT, problem, res)
        assert res.status == "succeeded" and row["objective_matches"] == 1, (problem, row)
        script = load_script(ROOT, problem)
        cands, reports = candidate_codes(res)
        for got, cand in zip(cands, script["candidates"], strict=False):
            if cand["expect"] is None:
                assert got == [], (problem, cand["mistake"], got)
                blind_spots += cand["mistake"] is not None
            else:
                assert got and got[0] == cand["expect"], (problem, cand["mistake"], got)
        for got, rep in zip(reports, script["reports"], strict=False):
            assert (got[0] if got else None) == rep["expect"], (problem, got)
    assert blind_spots >= 1  # at least one wrong candidate is accepted by the verifier
    # one run replays from its trace without the MCP server and without the model
    spans = results["product_mix"].trace.dicts()
    rep = replay_spans(spans, lambda s: opt_env(load_script(ROOT, "product_mix"), []))
    assert rep["divergence"] is None and rep["provider_calls"] == 0 and rep["tool_calls"] == 0


def test_the_workflow_never_reads_the_reference_formulations():
    for name in ("workflow.py", "server.py", "core.py"):
        src = (ROOT / "src" / "agent_runtime" / "optimization" / name).read_text(encoding="utf-8")
        assert "reference" not in src.lower().replace("references", ""), name


CARD = load_card(ROOT, "product_mix")
REF = json.loads(
    (ROOT / "examples" / "optimization" / "reference" / "product_mix.json").read_text(encoding="utf-8")
)["formulation"]


def mutate(fn):
    f = json.loads(json.dumps(REF))
    fn(f)
    return f


@pytest.mark.parametrize(
    "form,code",
    [
        (mutate(lambda f: f["variables"].append(dict(f["variables"][0]))), "V1_DUPLICATE_NAME"),
        (mutate(lambda f: f["objective"]["terms"].update({"x_c": 1})), "V1_UNDECLARED_VARIABLE"),
        (mutate(lambda f: f["constraints"][0]["terms"].update({"x_a": float("inf")})), "V1_NONFINITE"),
        (mutate(lambda f: f["variables"][0].update({"lb": 50, "ub": 40})), "V1_BOUNDS"),
        (mutate(lambda f: f.pop("sense")), "V1_PARSE"),
        (
            mutate(
                lambda f: f["constraints"].append(
                    {"name": "impossible", "terms": {"x_a": 1}, "sense": ">=", "rhs": 1000}
                )
            ),
            "V2_INFEASIBLE",
        ),
        (mutate(lambda f: f["constraints"][0].update({"rhs": 120})), "V4_INFEASIBLE_WITNESS_ACCEPTED"),
        (
            mutate(
                lambda f: (
                    f["variables"][0].update({"name": "units_a"})
                    or f["objective"]["terms"].update({"units_a": f["objective"]["terms"].pop("x_a")})
                    or [c["terms"].update({"units_a": c["terms"].pop("x_a")}) for c in f["constraints"]]
                )
            ),
            "V1_MISSING_CARD_VARIABLE",
        ),
        (
            mutate(lambda f: f.update({"constraints": []}) or f["variables"][0].update({"ub": None})),
            "V2_UNBOUNDED",
        ),
        (
            mutate(
                lambda f: f["constraints"].append(
                    {"name": "cap_b", "terms": {"x_b": 1}, "sense": "<=", "rhs": 30}
                )
            ),
            "V4_FEASIBLE_WITNESS_REJECTED",
        ),
    ],
)
def test_verifier_checks_raise_their_codes(form, code):
    out = core.verify(CARD, form)
    assert not out["accepted"] and out["diagnostics"][0]["code"] == code


def test_v3_rechecks_the_solution_without_the_solver_and_v4_catches_a_witness_that_beats_the_optimum():
    sol = core.solve(core.Formulation.model_validate(REF))
    bad = dict(sol, values={"x_a": 20, "x_b": 70})
    codes = [d["code"] for d in core.verify(CARD, REF, bad)["diagnostics"]]
    assert "V3_CONSTRAINT" in codes and "V3_OBJECTIVE" in codes
    frac = core.verify(
        load_card(ROOT, "knapsack"),
        json.loads((ROOT / "examples/optimization/reference/knapsack.json").read_text())["formulation"],
        {
            "status": "OPTIMAL",
            "objective": 29.0,
            "values": {"x_1": 0.5, "x_2": 1, "x_3": 1, "x_4": 0, "x_5": 0, "x_6": 0},
        },
    )
    assert frac["diagnostics"][0]["code"] in ("V3_INTEGRALITY", "V3_OBJECTIVE")
    over = mutate(
        lambda f: f["constraints"].append(
            {"name": "tight", "terms": {"x_a": 1, "x_b": 1}, "sense": "<=", "rhs": 15}
        )
    )
    card = dict(CARD, witnesses=[{"point": {"x_a": 10, "x_b": 5}, "feasible": True, "note": "fifteen units"}])
    # within the over-constrained formulation the optimum is 450; a feasible witness reaches 400, fine
    assert core.verify(card, over)["accepted"]
    card2 = dict(CARD, witnesses=[{"point": {"x_a": 15, "x_b": 0}, "feasible": True, "note": "450"}])
    over2 = mutate(lambda f: f["objective"]["terms"].update({"x_b": 40}))
    out = core.verify(card2, over2)
    assert out["accepted"]  # an objective coefficient no witness reveals: a documented blind spot
    assert core.check_report(CARD, {"objective": 1800, "values": {"x_a": 20, "x_b": 60}}, sol)["accepted"]
    bad_report = core.check_report(CARD, {"objective": 1800, "values": {"x_a": 20}}, sol)
    assert not bad_report["accepted"] and bad_report["diagnostics"][0]["code"] == "V5_REPORT_MISMATCH"


def test_a_failing_solver_tool_fails_the_run_with_step_failed_and_three_rejections_with_verifier_rejected():
    from pydantic import BaseModel

    from agent_runtime.clock import run_virtual
    from agent_runtime.runtime.runner import RunSpec, execute
    from agent_runtime.store.memory import MemoryStore
    from agent_runtime.tools import Tool, ToolError

    class In(BaseModel):
        formulation: dict

    class VIn(BaseModel):
        card: dict
        formulation: dict
        solution: dict

    async def broken(v):
        raise ToolError("FAILED", "solver crashed")

    async def solve(v):
        return core.solve(core.Formulation.model_validate(v.formulation))

    async def reject(v):
        return {
            "accepted": False,
            "diagnostics": [{"code": "V4_INFEASIBLE_WITNESS_ACCEPTED", "message": "scripted rejection"}],
        }

    def go(tools):
        script = load_script(ROOT, "product_mix")
        env = opt_env({"candidates": script["candidates"] * 2, "reports": script["reports"]}, tools)
        spec = RunSpec(
            run_id="r-x", workflow="optimize", input={"card": CARD}, default_target="scripted/opt-script"
        )
        store = MemoryStore()
        return run_virtual(lambda clock: execute(spec, env, clock, store, store.create_session()))

    res = go([Tool(name="opt.solve", handler=broken, input_model=In)])
    assert (res.status, res.failure_reason) == ("failed", "step_failed")
    res = go(
        [
            Tool(name="opt.solve", handler=solve, input_model=In),
            Tool(name="opt.verify", handler=reject, input_model=VIn),
        ]
    )
    assert (res.status, res.failure_reason) == ("failed", "verifier_rejected")


def test_a_tool_registry_from_the_live_server_validates_arguments_against_its_schema():
    async def go():
        server = opt_server()
        await server.start()
        try:
            reg = ToolRegistry(server.tools())
            return reg.get("opt.solve").validate_input({"formulation": "not an object"})[1], sorted(
                reg.names()
            )
        finally:
            await server.close()

    err, names = asyncio.run(go())
    assert err is not None and names == ["opt.check_report", "opt.solve", "opt.verify"]


def test_e4b_mutants_are_seeded_and_ground_truth_uses_the_reference_outside_the_workflow():
    from agent_runtime.optimization.mutants import OPERATORS, ground_truth, mutate, reference, run_e4b
    from agent_runtime.rand import stream

    ref = reference(ROOT, "product_mix")["formulation"]
    a = mutate(ref, "drop_constraint", stream(1, "mutate/product_mix/drop_constraint/0"))
    b = mutate(ref, "drop_constraint", stream(1, "mutate/product_mix/drop_constraint/0"))
    assert a == b and a != ref
    assert ground_truth(CARD, ref, ref) == (False, "equivalent at the optimum")
    assert ground_truth(CARD, ref, mutate(ref, "rename_variable", stream(1, "x")))[0]
    swapped = mutate(ref, "swap_sense", stream(1, "y"))
    assert (
        ground_truth(CARD, ref, swapped)[0] and core.verify(CARD, swapped)["accepted"]
    )  # a swapped sense the witnesses miss
    rows, summary = run_e4b(ROOT)
    assert {r["operator"] for r in rows} == set(OPERATORS)
    rename = [s for s in summary if s["operator"] == "rename_variable"]
    assert all(s["detection_rate"] == 1.0 for s in rename)
    none_cfg = [s for s in summary if s["config"] == "no_witnesses" and s["operator"] == "drop_constraint"][0]
    full_cfg = [s for s in summary if s["config"] == "all_witnesses" and s["operator"] == "drop_constraint"][
        0
    ]
    assert full_cfg["detection_rate"] >= none_cfg["detection_rate"]
