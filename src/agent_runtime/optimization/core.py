"""Formulations, the solver (SciPy ``milp``, HiGHS), and the verifier checks V1-V5.

Verifier checks (each failure has a diagnostic code that goes back to the model):

- V1 the formulation parses, names are unique, every term names a declared variable, no coefficient
  is NaN or infinite, bounds are consistent, and the card's variables are declared
  (``V1_PARSE``, ``V1_DUPLICATE_NAME``, ``V1_UNDECLARED_VARIABLE``, ``V1_NONFINITE``, ``V1_BOUNDS``,
  ``V1_MISSING_CARD_VARIABLE``);
- V2 the solver status is optimal (``V2_INFEASIBLE``, ``V2_UNBOUNDED``, ``V2_SOLVER_ERROR``);
- V3 the returned solution satisfies every bound, integrality condition, and constraint when
  recomputed without the solver, and its objective equals the reported one (``V3_BOUND``,
  ``V3_INTEGRALITY``, ``V3_CONSTRAINT``, ``V3_OBJECTIVE``);
- V4 witnesses: a feasible witness must be feasible for the formulation (card variables fixed,
  auxiliary variables solved for) and must not beat the optimum; an infeasible witness must be
  infeasible (``V4_FEASIBLE_WITNESS_REJECTED``, ``V4_WITNESS_BEATS_OPTIMUM``,
  ``V4_INFEASIBLE_WITNESS_ACCEPTED``);
- V5 the report repeats the solver's numbers (``V5_REPORT_MISMATCH``).

What the verifier does not check: that the formulation means what the statement says beyond what
the witnesses test; a missing constraint that is not binding at the optimum and that no infeasible
witness violates; an objective coefficient that no witness reveals; the correctness of the data on
the card; the uniqueness of the optimum; anything about the solver beyond its tolerances.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError

DEFAULT_TOLERANCES = {
    "feasibility": 1e-6,
    "integrality": 1e-6,
    "objective_abs": 1e-6,
    "objective_rel": 1e-6,
    "report_abs": 1e-4,
}

FORMULATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sense": {"enum": ["min", "max"]},
        "variables": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "lb": {"type": ["number", "null"]},
                    "ub": {"type": ["number", "null"]},
                    "type": {"enum": ["continuous", "integer", "binary"]},
                },
                "required": ["name", "lb", "ub", "type"],
            },
        },
        "objective": {
            "type": "object",
            "properties": {
                "terms": {"type": "object", "additionalProperties": {"type": "number"}},
                "constant": {"type": "number"},
            },
            "required": ["terms"],
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "terms": {"type": "object", "additionalProperties": {"type": "number"}},
                    "sense": {"enum": ["<=", ">=", "=="]},
                    "rhs": {"type": "number"},
                },
                "required": ["name", "terms", "sense", "rhs"],
            },
        },
    },
    "required": ["sense", "variables", "objective", "constraints"],
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {"type": "number"},
        "values": {"type": "object", "additionalProperties": {"type": "number"}},
    },
    "required": ["objective", "values"],
}


class Variable(BaseModel):
    name: str
    lb: float | None = 0.0
    ub: float | None = None
    type: str = "continuous"


class Objective(BaseModel):
    terms: dict[str, float] = Field(default_factory=dict)
    constant: float = 0.0


class Constraint(BaseModel):
    name: str
    terms: dict[str, float]
    sense: str
    rhs: float


class Formulation(BaseModel):
    sense: str
    variables: list[Variable]
    objective: Objective
    constraints: list[Constraint] = Field(default_factory=list)


def _tol(card: dict[str, Any] | None) -> dict[str, float]:
    return {**DEFAULT_TOLERANCES, **((card or {}).get("tolerances") or {})}


def bounds_of(v: Variable) -> tuple[float, float]:
    lb = -math.inf if v.lb is None else v.lb
    ub = math.inf if v.ub is None else v.ub
    if v.type == "binary":
        lb, ub = max(lb, 0.0), min(ub, 1.0)
    return lb, ub


# -- V1 -------------------------------------------------------------------------------------------


def check_structure(raw: Any, card: dict[str, Any] | None) -> tuple[Formulation | None, list[dict[str, str]]]:
    diags: list[dict[str, str]] = []
    try:
        f = Formulation.model_validate(raw)
    except ValidationError as e:
        return None, [{"code": "V1_PARSE", "message": str(e.errors()[0].get("msg"))}]
    if f.sense not in ("min", "max"):
        diags.append({"code": "V1_PARSE", "message": f"sense must be min or max, not {f.sense!r}"})
    names = [v.name for v in f.variables]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        diags.append({"code": "V1_DUPLICATE_NAME", "message": f"duplicate variable names: {dup}"})
    cnames = [c.name for c in f.constraints]
    cdup = sorted({n for n in cnames if cnames.count(n) > 1})
    if cdup:
        diags.append({"code": "V1_DUPLICATE_NAME", "message": f"duplicate constraint names: {cdup}"})
    declared = set(names)
    for where, terms in [("objective", f.objective.terms)] + [
        (f"constraint {c.name}", c.terms) for c in f.constraints
    ]:
        for var, coef in terms.items():
            if var not in declared:
                diags.append(
                    {"code": "V1_UNDECLARED_VARIABLE", "message": f"{where} uses undeclared variable {var!r}"}
                )
            if not math.isfinite(coef):
                diags.append(
                    {"code": "V1_NONFINITE", "message": f"{where}: coefficient of {var!r} is not finite"}
                )
    for c in f.constraints:
        if c.sense not in ("<=", ">=", "=="):
            diags.append({"code": "V1_PARSE", "message": f"constraint {c.name}: sense {c.sense!r}"})
        if not math.isfinite(c.rhs):
            diags.append(
                {"code": "V1_NONFINITE", "message": f"constraint {c.name}: right-hand side is not finite"}
            )
    for v in f.variables:
        lb, ub = bounds_of(v)
        if v.type not in ("continuous", "integer", "binary"):
            diags.append({"code": "V1_PARSE", "message": f"variable {v.name}: type {v.type!r}"})
        if lb > ub or (v.lb is not None and math.isnan(v.lb)) or (v.ub is not None and math.isnan(v.ub)):
            diags.append({"code": "V1_BOUNDS", "message": f"variable {v.name}: lb {v.lb} > ub {v.ub}"})
    if card is not None:
        missing = [n for n in card["variables"] if n not in declared]
        if missing:
            diags.append(
                {
                    "code": "V1_MISSING_CARD_VARIABLE",
                    "message": f"the card's variables {missing} are not declared (use the card's names)",
                }
            )
    return f, diags


# -- solver ---------------------------------------------------------------------------------------


def solve(f: Formulation, fixed: dict[str, float] | None = None, objective: bool = True) -> dict[str, Any]:
    """Solve with SciPy ``milp`` (HiGHS). ``fixed`` pins variables (witness checks)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    names = [v.name for v in f.variables]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    c = np.zeros(n)
    if objective:
        for var, coef in f.objective.terms.items():
            c[idx[var]] = coef
        if f.sense == "max":
            c = -c
    lb = np.array([bounds_of(v)[0] for v in f.variables], dtype=float)
    ub = np.array([bounds_of(v)[1] for v in f.variables], dtype=float)
    for var, val in (fixed or {}).items():
        if var in idx:  # a fixed value must also respect the variable's own bounds
            i = idx[var]
            if val < lb[i] - 1e-9 or val > ub[i] + 1e-9:
                return {"status": "INFEASIBLE", "objective": None, "values": {}}
            lb[i] = ub[i] = val
    integrality = np.array([0 if v.type == "continuous" else 1 for v in f.variables])
    cons = []
    if f.constraints:
        a = np.zeros((len(f.constraints), n))
        lo = np.full(len(f.constraints), -np.inf)
        hi = np.full(len(f.constraints), np.inf)
        for r, con in enumerate(f.constraints):
            for var, coef in con.terms.items():
                a[r, idx[var]] = coef
            if con.sense in ("<=", "=="):
                hi[r] = con.rhs
            if con.sense in (">=", "=="):
                lo[r] = con.rhs
        cons.append(LinearConstraint(a, lo, hi))
    if np.any(lb > ub):
        return {"status": "INFEASIBLE", "objective": None, "values": {}}
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
    status = {0: "OPTIMAL", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(res.status, "ERROR")
    if status == "ERROR" and "unbounded or infeasible" in str(res.message) and objective:
        # HiGHS may not tell the two apart for a MIP: a feasible model is then unbounded
        status = (
            "UNBOUNDED" if solve(f, fixed=fixed, objective=False)["status"] == "OPTIMAL" else "INFEASIBLE"
        )
    if status != "OPTIMAL":
        return {"status": status, "objective": None, "values": {}, "message": str(res.message)}
    x = res.x
    obj = float(
        sum(f.objective.terms.get(nm, 0.0) * x[i] for i, nm in enumerate(names)) + f.objective.constant
    )
    return {"status": "OPTIMAL", "objective": obj, "values": {nm: float(x[i]) for i, nm in enumerate(names)}}


def objective_value(f: Formulation, values: dict[str, float]) -> float:
    return float(
        sum(coef * values.get(v, 0.0) for v, coef in f.objective.terms.items()) + f.objective.constant
    )


# -- V3 -------------------------------------------------------------------------------------------


def recheck(f: Formulation, sol: dict[str, Any], card: dict[str, Any] | None) -> list[dict[str, str]]:
    t = _tol(card)
    diags = []
    vals = sol.get("values") or {}
    for v in f.variables:
        x = vals.get(v.name)
        if x is None:
            diags.append({"code": "V3_BOUND", "message": f"no value for {v.name}"})
            continue
        lb, ub = bounds_of(v)
        if x < lb - t["feasibility"] or x > ub + t["feasibility"]:
            diags.append({"code": "V3_BOUND", "message": f"{v.name}={x} outside [{lb}, {ub}]"})
        if v.type != "continuous" and abs(x - round(x)) > t["integrality"]:
            diags.append({"code": "V3_INTEGRALITY", "message": f"{v.name}={x} is not integral"})
    for c in f.constraints:
        lhs = sum(coef * vals.get(var, 0.0) for var, coef in c.terms.items())
        tol = t["feasibility"] * max(1.0, abs(c.rhs))
        bad = (
            (c.sense == "<=" and lhs > c.rhs + tol)
            or (c.sense == ">=" and lhs < c.rhs - tol)
            or (c.sense == "==" and abs(lhs - c.rhs) > tol)
        )
        if bad:
            diags.append(
                {
                    "code": "V3_CONSTRAINT",
                    "message": f"constraint {c.name}: {lhs} {c.sense} {c.rhs} does not hold",
                }
            )
    obj = objective_value(f, vals)
    rep = sol.get("objective")
    if rep is None or abs(obj - rep) > t["objective_abs"] + t["objective_rel"] * abs(obj):
        diags.append({"code": "V3_OBJECTIVE", "message": f"recomputed objective {obj} != reported {rep}"})
    return diags


# -- V4 -------------------------------------------------------------------------------------------


def witness_checks(
    f: Formulation, optimum: float, card: dict[str, Any], limit: int | None = None
) -> list[dict[str, str]]:
    t = _tol(card)
    diags = []
    witnesses = card.get("witnesses", [])
    if limit is not None:
        witnesses = witnesses[:limit]
    for w in witnesses:
        point = {k: float(v) for k, v in w["point"].items()}
        res = solve(f, fixed=point)  # auxiliary variables are solved for (best objective)
        if w["feasible"]:
            if res["status"] != "OPTIMAL":
                diags.append(
                    {
                        "code": "V4_FEASIBLE_WITNESS_REJECTED",
                        "message": f"feasible witness {w['note']!r} is infeasible for the formulation",
                    }
                )
                continue
            better = res["objective"] > optimum if f.sense == "max" else res["objective"] < optimum
            if better and abs(res["objective"] - optimum) > t["objective_abs"] + t["objective_rel"] * abs(
                optimum
            ):
                diags.append(
                    {
                        "code": "V4_WITNESS_BEATS_OPTIMUM",
                        "message": f"feasible witness {w['note']!r} has objective {res['objective']}, better than the optimum {optimum}",
                    }
                )
        elif res["status"] == "OPTIMAL":
            diags.append(
                {
                    "code": "V4_INFEASIBLE_WITNESS_ACCEPTED",
                    "message": f"infeasible witness {w['note']!r} is feasible for the formulation",
                }
            )
    return diags


def verify(
    card: dict[str, Any], raw: Any, solution: dict[str, Any] | None = None, *, witnesses: int | None = None
) -> dict[str, Any]:
    """V1-V4. Returns ``{"accepted": bool, "diagnostics": [...], "solution": {...}}``."""
    f, diags = check_structure(raw, card)
    if f is None or diags:
        return {"accepted": False, "diagnostics": diags, "solution": None}
    sol = solution if solution is not None else solve(f)
    st = sol.get("status")
    if st != "OPTIMAL":
        code = {"INFEASIBLE": "V2_INFEASIBLE", "UNBOUNDED": "V2_UNBOUNDED"}.get(st, "V2_SOLVER_ERROR")
        return {
            "accepted": False,
            "diagnostics": [{"code": code, "message": f"solver status {st}"}],
            "solution": sol,
        }
    diags = recheck(f, sol, card)
    if not diags:
        diags = witness_checks(f, sol["objective"], card, witnesses)
    return {"accepted": not diags, "diagnostics": diags, "solution": sol}


def check_report(card: dict[str, Any], report: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    """V5: the report repeats the solver's objective and the values of the card's variables."""
    t = _tol(card)
    diags = []
    if abs(float(report.get("objective", math.nan)) - solution["objective"]) > t["report_abs"] * max(
        1.0, abs(solution["objective"])
    ):
        diags.append(
            {
                "code": "V5_REPORT_MISMATCH",
                "message": f"reported objective {report.get('objective')} != solver {solution['objective']}",
            }
        )
    for var in card["variables"]:
        got = (report.get("values") or {}).get(var)
        want = solution["values"].get(var)
        if got is None or want is None or abs(float(got) - want) > t["report_abs"] * max(1.0, abs(want)):
            diags.append({"code": "V5_REPORT_MISMATCH", "message": f"reported {var}={got} != solver {want}"})
    return {"accepted": not diags, "diagnostics": diags}
