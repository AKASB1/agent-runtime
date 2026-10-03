"""E4b, the verifier study: seeded mutations of the reference formulations, ground truth computed
with the reference outside the workflow, and the verifier's detection under three witness
configurations (none, the first 2, all). The claim is limited to these injected mutations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent_runtime.optimization import core
from agent_runtime.optimization.e4 import PROBLEMS, load_card
from agent_runtime.rand import below, stream

OPERATORS = (
    "drop_constraint", "reverse_inequality", "scale_coef_10", "scale_coef_0.1", "change_rhs",
    "change_bound", "drop_integrality", "swap_sense", "rename_variable", "add_constraint",
)  # fmt: skip
WITNESS_CONFIGS = {"no_witnesses": 0, "two_witnesses": 2, "all_witnesses": None}
MUTANTS_PER_OPERATOR = 5


def reference(root: Path, pid: str) -> dict[str, Any]:
    return json.loads(
        (Path(root) / "examples" / "optimization" / "reference" / f"{pid}.json").read_text(encoding="utf-8")
    )


def mutate(ref: dict[str, Any], op: str, rng: Any) -> dict[str, Any] | None:
    """One mutant of ``ref`` by operator ``op`` (None if the operator does not apply)."""
    f = copy.deepcopy(ref)
    cons = f["constraints"]
    if op == "drop_constraint":
        if not cons:
            return None
        cons.pop(below(rng, len(cons)))
    elif op == "reverse_inequality":
        ineq = [c for c in cons if c["sense"] in ("<=", ">=")] or cons
        if not ineq:
            return None
        c = ineq[below(rng, len(ineq))]
        c["sense"] = {"<=": ">=", ">=": "<=", "==": "<="}[c["sense"]]
    elif op in ("scale_coef_10", "scale_coef_0.1"):
        factor = 10.0 if op == "scale_coef_10" else 0.1
        places = [("objective", None, v) for v in f["objective"]["terms"]] + [
            ("constraint", i, v) for i, c in enumerate(cons) for v in c["terms"]
        ]
        kind, i, v = places[below(rng, len(places))]
        terms = f["objective"]["terms"] if kind == "objective" else cons[i]["terms"]
        terms[v] = terms[v] * factor
    elif op == "change_rhs":
        if not cons:
            return None
        c = cons[below(rng, len(cons))]
        delta = 0.5 + rng.random()  # +-50% .. +-150% of a unit-scaled change
        c["rhs"] = (
            round(c["rhs"] * (1 + (delta if rng.random() < 0.5 else -delta / 2)), 6) if c["rhs"] else 1.0
        )
    elif op == "change_bound":
        v = f["variables"][below(rng, len(f["variables"]))]
        if v["type"] == "binary":
            v["type"] = "integer"
            v["ub"] = 2
        elif v["ub"] is None:
            v["ub"] = round(5 + rng.random() * 50, 3)
        else:
            v["ub"] = round(v["ub"] * (0.5 if rng.random() < 0.5 else 1.5), 6)
    elif op == "drop_integrality":
        ints = [v for v in f["variables"] if v["type"] != "continuous"]
        if not ints:
            return None
        for v in ints:
            if v["type"] == "binary":
                v["lb"], v["ub"] = 0, 1
            v["type"] = "continuous"
    elif op == "swap_sense":
        f["sense"] = "min" if f["sense"] == "max" else "max"
    elif op == "rename_variable":
        v = f["variables"][below(rng, len(f["variables"]))]
        old, new = v["name"], v["name"] + "_x"
        v["name"] = new
        for terms in [f["objective"]["terms"]] + [c["terms"] for c in cons]:
            if old in terms:
                terms[new] = terms.pop(old)
    elif op == "add_constraint":
        names = [v["name"] for v in f["variables"]]
        picks = sorted({names[below(rng, len(names))] for _ in range(2)})
        sol = core.solve(core.Formulation.model_validate(ref))
        lhs = sum(sol["values"].get(n, 0.0) for n in picks)
        # a random cap on the sum of one or two variables, around the reference optimum's value
        cons.append(
            {
                "name": "extra",
                "terms": {n: 1.0 for n in picks},
                "sense": "<=",
                "rhs": round(lhs * (0.5 + rng.random()), 6),
            }
        )
    else:
        raise ValueError(op)
    return f


def ground_truth(card: dict[str, Any], ref: dict[str, Any], mutant: dict[str, Any]) -> tuple[bool, str]:
    """(wrong, reason): wrong if the mutant's optimum is infeasible for the reference or its objective
    differs from the reference optimum beyond the tolerance."""
    t = {**core.DEFAULT_TOLERANCES, **card.get("tolerances", {})}
    names = {v["name"] for v in mutant["variables"]}
    if any(v not in names for v in card["variables"]):
        return True, "a card variable is renamed"
    rf = core.Formulation.model_validate(ref)
    ropt = core.solve(rf)
    try:
        mf = core.Formulation.model_validate(mutant)
    except Exception:  # noqa: BLE001
        return True, "does not parse"
    msol = core.solve(mf)
    if msol["status"] != "OPTIMAL":
        return True, f"mutant {msol['status'].lower()}"
    point = {v: msol["values"][v] for v in card["variables"]}
    fixed = core.solve(rf, fixed=point, objective=False)
    if fixed["status"] != "OPTIMAL":
        return True, "the mutant's optimum is infeasible for the reference"
    if abs(msol["objective"] - ropt["objective"]) > t["objective_abs"] + t["objective_rel"] * abs(
        ropt["objective"]
    ):
        return True, f"objective {msol['objective']:.6g} != reference {ropt['objective']:.6g}"
    return False, "equivalent at the optimum"


def run_e4b(root: Path, seed: int = 1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (per-mutant rows, per (operator, configuration) summary rows)."""
    rows = []
    for pid in PROBLEMS:
        card = load_card(root, pid)
        ref = reference(root, pid)["formulation"]
        for op in OPERATORS:
            for k in range(MUTANTS_PER_OPERATOR):
                rng = stream(seed, f"mutate/{pid}/{op}/{k}")
                m = mutate(ref, op, rng)
                if m is None:
                    continue
                wrong, why = ground_truth(card, ref, m)
                row = {"problem": pid, "operator": op, "k": k, "wrong": int(wrong), "truth": why}
                for cfg, n in WITNESS_CONFIGS.items():
                    out = core.verify(card, m, witnesses=n)
                    row[f"{cfg}_accepted"] = int(out["accepted"])
                    row[f"{cfg}_code"] = out["diagnostics"][0]["code"] if out["diagnostics"] else ""
                rows.append(row)
    summary = []
    for op in OPERATORS:
        sub = [r for r in rows if r["operator"] == op]
        wrong = [r for r in sub if r["wrong"]]
        equiv = [r for r in sub if not r["wrong"]]
        for cfg in WITNESS_CONFIGS:
            summary.append(
                {
                    "operator": op,
                    "config": cfg,
                    "mutants": len(sub),
                    "wrong": len(wrong),
                    "equivalent": len(equiv),
                    "detection_rate": round(sum(1 - r[f"{cfg}_accepted"] for r in wrong) / len(wrong), 4)
                    if wrong
                    else "",
                    "false_reject_rate": round(sum(1 - r[f"{cfg}_accepted"] for r in equiv) / len(equiv), 4)
                    if equiv
                    else "",
                }
            )
    return rows, summary
