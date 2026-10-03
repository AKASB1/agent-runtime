"""``python -m agent_runtime report``: figures (scripts/plot_results.py) and markdown tables
(results/tables.md) from the committed results only. Every number is simulated."""

from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str, scale: float = 1.0, nd: int = 3) -> str:
    if x in ("", None):
        return ""
    return f"{float(x) / scale:.{nd}f}"


def _ci(r: dict, m: str, scale: float = 1.0, nd: int = 3) -> str:
    if r.get(m, "") == "":
        return ""
    ci = r.get(m + "_ci", "")
    return _f(r[m], scale, nd) + (f" ± {_f(ci, scale, nd)}" if ci not in ("", None) else "")


def e1_table(res: Path) -> str:
    rows = _rows(res / "e1" / "table.csv")
    out = [
        "| Router | Budget B (mcu) | Configuration | Success | Cost (mcu) | P95 latency (s) | Escalations | Over 1.05 B | Δ success vs random_mix | W/T/L vs random_mix |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = f"{r['budget']} = {_f(r['B_ucu'], 1000, 2)}" if r.get("B_ucu") else "-"
        if r["config"] == "(none within B)":
            out.append(f"| {r['router']} | {b} | none within B on the tuning seeds | | | | | | | |")
            continue
        d = _ci(r, "d_success_vs_mix") if r.get("d_success_vs_mix") else ""
        wtl = f"{r['win']}/{r['tie']}/{r['loss']}" if r.get("win") not in (None, "") else ""
        name = r["router"] + (" (upper bound, not deployable)" if r["router"] == "oracle" else "")
        out.append(
            f"| {name} | {b} | `{r['config']}` | {_ci(r, 'success')} | {_ci(r, 'cost_ucu', 1000, 2)} | {_ci(r, 'latency_p95_ms', 1000, 2)} | "
            f"{_ci(r, 'escalations', 1, 3)} | {r.get('over_budget', '')} | {d} | {wtl} |"
        )
    return "\n".join(out)


def e2_table(res: Path) -> str:
    agg = {r["config"]: r for r in _rows(res / "e2" / "aggregate.csv")}
    pair = {r["config"]: r for r in _rows(res / "e2" / "paired.csv")}
    out = [
        "| Profile | Configuration | Success | P95 latency (s) | Cost (mcu) | Provider calls / task | Wasted calls / task | Δ success vs retry | W/T/L vs retry |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for prof in ("calm", "flaky", "brownout"):
        for cfg in ("no_retry", "retry", "retry_fallback"):
            r = agg.get(f"{prof}/{cfg}")
            if not r:
                continue
            p = pair.get(f"{prof}/{cfg}", {})
            d = _ci(p, "d_success") if p else "(reference)"
            wtl = f"{p['win']}/{p['tie']}/{p['loss']}" if p else ""
            out.append(
                f"| {prof} | {cfg} | {_ci(r, 'success')} | {_ci(r, 'latency_p95_ms', 1000, 2)} | {_ci(r, 'cost_ucu', 1000, 2)} | "
                f"{_ci(r, 'provider_calls', 1, 2)} | {_ci(r, 'wasted_calls', 1, 2)} | {d} | {wtl} |"
            )
    return "\n".join(out)


def e3_table(res: Path) -> str:
    agg = {r["config"]: r for r in _rows(res / "e3" / "aggregate.csv")}
    pair = {r["config"]: r for r in _rows(res / "e3" / "paired.csv")}
    out = [
        "| Tool failures | Mode | k | Success | Makespan P50 (s) | Makespan P95 (s) | Model calls | Cost (mcu) | bound_gap | Duplicate writes | Replans | Δ P50 vs react (s) | W/T/L vs react |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(
        agg,
        key=lambda n: (
            n.split("/")[0],
            ["react", "react_parallel", "plan_execute", "plan_replan"].index(n.split("/")[1]),
            n,
        ),
    ):
        r = agg[name]
        parts = name.split("/")
        k = parts[2][1:] if len(parts) > 2 else "-"
        p = pair.get(name, {})
        d = _ci(p, "d_latency_p50_ms", 1000, 2) if p else "(reference)"
        wtl = f"{p['win']}/{p['tie']}/{p['loss']}" if p else ""
        out.append(
            f"| {int(parts[0][1:])}% | {parts[1]} | {k} | {_ci(r, 'success')} | {_ci(r, 'latency_p50_ms', 1000, 2)} | {_ci(r, 'latency_p95_ms', 1000, 2)} | "
            f"{_ci(r, 'model_calls', 1, 2)} | {_ci(r, 'cost_ucu', 1000, 2)} | {_ci(r, 'bound_gap', 1, 3)} | {_f(r['duplicate_writes'], 1, 0)} | {_ci(r, 'replans', 1, 3)} | {d} | {wtl} |"
        )
    return "\n".join(out)


def e4_table(res: Path) -> str:
    rows = _rows(res / "e4" / "table.csv")
    out = [
        "| Problem | Kind | Candidates tried | Codes per candidate | Report codes | Accepted objective | Reference objective | Match | Model calls | Cost (mcu, estimated) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cands = r["candidate_codes"].replace(" | ", "; ")
        reps = r["report_codes"].replace(" | ", "; ")
        out.append(
            f"| {r['problem']} | {r['kind']} | {r['candidates_tried']} | {cands} | {reps} | {r['accepted_objective']} | "
            f"{r['reference_objective']} | {'yes' if r['objective_matches'] == '1' else 'no'} | {r['model_calls']} | {r['cost_mcu_estimated']} |"
        )
    return "\n".join(out)


def e4b_table(res: Path) -> str:
    rows = _rows(res / "e4b" / "summary.csv")
    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        by.setdefault(r["operator"], {})[r["config"]] = r
    out = [
        "| Operator | Mutants (wrong / equivalent) | Detection: no witnesses | 2 witnesses | all witnesses | False rejects: no witnesses | 2 witnesses | all witnesses |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for op, cfgs in by.items():
        a = cfgs["no_witnesses"]
        det = [cfgs[c]["detection_rate"] for c in ("no_witnesses", "two_witnesses", "all_witnesses")]
        fr = [cfgs[c]["false_reject_rate"] for c in ("no_witnesses", "two_witnesses", "all_witnesses")]
        out.append(
            f"| {op} | {a['mutants']} ({a['wrong']} / {a['equivalent']}) | "
            + " | ".join(x or "-" for x in det)
            + " | "
            + " | ".join(x or "-" for x in fr)
            + " |"
        )
    missed = _rows(res / "e4b" / "missed_by_all.csv")
    reasons: dict[str, int] = {}
    for m in missed:
        reasons[m["operator"]] = reasons.get(m["operator"], 0) + 1
    out.append("")
    out.append(
        f"Wrong mutants accepted by every configuration: {len(missed)} ("
        + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
        + "); the list is in `results/e4b/missed_by_all.csv`."
    )
    return "\n".join(out)


def e5_table(res: Path) -> str:
    agg = {r["config"]: r for r in _rows(res / "e5" / "aggregate.csv")}
    pair = {r["config"]: r for r in _rows(res / "e5" / "paired.csv")}
    out = [
        "| Setup | Result tokens | Cache | Configuration | Success | Cost (mcu) | P95 latency (s) | Model calls | Calls on medium | context_length failures | Peak context | Compactions | Repeat reads | Cache share | Δ success vs none | W/T/L vs none |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(
        agg,
        key=lambda n: (
            n.split("/")[0],
            int(n.split("/")[1][1:]),
            n.split("/")[2],
            n.split("/")[3] != "none",
            n,
        ),
    ):
        r = agg[name]
        setup, size, cache, cfg = name.split("/", 3)
        p = pair.get(name, {})
        d = _ci(p, "d_success") if p else "(reference)"
        wtl = f"{p['win']}/{p['tie']}/{p['loss']}" if p else ""
        out.append(
            f"| {setup} | {size[1:]} | {cache} | `{cfg}` | {_ci(r, 'success')} | {_ci(r, 'cost_ucu', 1000, 2)} | {_ci(r, 'latency_p95_ms', 1000, 2)} | {_ci(r, 'model_calls', 1, 2)} | "
            f"{_ci(r, 'medium_calls', 1, 2)} | {_ci(r, 'context_length_failures', 1, 3)} | {_f(r['peak_context_tokens'], 1, 0)} | {_ci(r, 'compactions', 1, 2)} | {_ci(r, 'repeat_reads', 1, 2)} | {_ci(r, 'cache_share', 1, 3)} | {d} | {wtl} |"
        )
    return "\n".join(out)


def e6_table(res: Path) -> str:
    agg = {r["config"]: r for r in _rows(res / "e6" / "aggregate.csv")}
    pair = {r["config"]: r for r in _rows(res / "e6" / "paired.csv")}
    out = [
        "| Configuration | Success | Cost (mcu) | Makespan P50 (s) | Makespan P95 (s) | Model calls | Parent peak context | Subagent runs | Δ success vs react_medium | W/T/L vs react_medium |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(agg, key=lambda n: ("/" in n, n)):
        r = agg[name]
        p = pair.get(name, {})
        d = _ci(p, "d_success") if p else "(reference)"
        wtl = f"{p['win']}/{p['tie']}/{p['loss']}" if p else ""
        out.append(
            f"| `{name}` | {_ci(r, 'success')} | {_ci(r, 'cost_ucu', 1000, 2)} | {_ci(r, 'latency_p50_ms', 1000, 2)} | {_ci(r, 'latency_p95_ms', 1000, 2)} | "
            f"{_ci(r, 'model_calls', 1, 2)} | {_f(r['peak_parent_context_tokens'], 1, 0)} | {_ci(r, 'subagent_runs', 1, 2)} | {d} | {wtl} |"
        )
    return "\n".join(out)


def sensitivity_table(res: Path) -> str:
    base = res / "e1_sensitivity"
    if not base.exists():
        return ""
    out = [
        "| Variant | Router | Configuration (re-tuned at B3) | Success | Cost (mcu) | Δ success vs random_mix | W/T/L |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in sorted(p for p in base.iterdir() if (p / "table.csv").exists()):
        for r in _rows(d / "table.csv"):
            if r.get("budget") not in ("B3", "") or r["router"] == "oracle":
                continue
            if r["config"] == "(none within B)":
                out.append(f"| {d.name} | {r['router']} | none within B | | | | |")
                continue
            dd = _ci(r, "d_success_vs_mix") if r.get("d_success_vs_mix") else ""
            wtl = f"{r['win']}/{r['tie']}/{r['loss']}" if r.get("win") not in (None, "") else ""
            out.append(
                f"| {d.name} | {r['router']} | `{r['config']}` | {_ci(r, 'success')} | {_ci(r, 'cost_ucu', 1000, 2)} | {dd} | {wtl} |"
            )
    return "\n".join(out)


def main(root: Path) -> int:
    res = root / "results"
    sys.argv = ["plot_results.py", "--results", str(res), "--out", str(root / "docs" / "figures")]
    try:  # the script ends with SystemExit(main()), which must not end the report before the tables
        runpy.run_path(str(root / "scripts" / "plot_results.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    parts = [
        "# Result tables (simulated)",
        "",
        "Generated by `python -m agent_runtime report` from the CSV files in this folder. Means over seeds with 95% Student-t intervals. "
        "Every number is simulated under the assumptions of docs/contracts.md section 10 and says nothing about hosted models.",
        "",
    ]
    for title, fn in (
        ("E1 routing under a cost budget", e1_table),
        ("E1 sensitivity (seeds 101-110, B3, re-tuned)", sensitivity_table),
        ("E2 reliability under provider faults", e2_table),
        ("E3 planning and scheduling", e3_table),
        ("E4a optimization agent (scripted model, system clock)", e4_table),
        ("E4b verifier study (Tier 2)", e4b_table),
        ("E5 context policies (Tier 2)", e5_table),
        ("E6 delegation (Tier 2)", e6_table),
    ):
        try:
            body = fn(res)
        except FileNotFoundError as e:
            body = f"(missing: {Path(e.filename).name})"
        if body:
            parts += [f"## {title}", "", body, ""]
    manifests = {
        p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in res.glob("*/manifest.json")
    }
    if manifests:
        parts += [
            "## Manifests",
            "",
            "| Experiment | Commit | Config SHA-256 | Seeds | Workers | Wall (s) | Load note |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, m in sorted(manifests.items()):
            seeds = m.get("seeds") or []
            seed_txt = f"{seeds[0]}-{seeds[-1]}" if seeds else "-"
            parts.append(
                f"| {name} | {m['commit'][:12]} | {m['config_sha256'][:12]} | {seed_txt} | {m['workers']} | {m['wall_s']} | {m['load_note']} |"
            )
        parts.append("")
    (res / "tables.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"wrote {res / 'tables.md'}")
    return 0
