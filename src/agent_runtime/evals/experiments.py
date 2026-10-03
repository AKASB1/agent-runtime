"""Experiments E1-E3 (simulated) and their protocol: tuning on seeds 1-10, frozen selections,
evaluation on seeds 101-130, informativeness check, paired differences, win/tie/loss.

Everything here is a simulation under the assumptions of ``docs/contracts.md``; results say how
the runtime and its decision rules behave under those assumptions and nothing about hosted models.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_runtime.evals.simrun import TASK_COLUMNS, Job, run_job
from agent_runtime.evals.stats import (
    COST,
    LAT_P95,
    MAKESPAN_P50,
    SUCCESS,
    mean_ci,
    paired,
    per_seed,
    std_error,
    wtl,
)

TUNING_SEEDS = list(range(1, 11))
EVAL_SEEDS = list(range(101, 131))
SENS_SEEDS = list(range(101, 111))
QUICK_SEEDS = [101, 102, 103]
M = {"small": "sim-a/small", "medium": "sim-a/medium", "large": "sim-a/large"}
E1_DEADLINE_MS = 30_000
THRESHOLD_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CASCADE_CHAINS = [("small", "medium", "large"), ("small", "large"), ("medium", "large")]


# -- running jobs ---------------------------------------------------------------------------------


def run_jobs(jobs: list[Job], workers: int) -> list[dict[str, Any]]:
    """Run jobs in a spawn process pool (or inline for one worker); results sorted by
    (experiment, config, seed) so the output does not depend on the worker count."""
    if workers <= 1:
        results = [run_job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            results = list(ex.map(run_job, jobs, chunksize=1))
    results.sort(key=lambda r: (r["job"].experiment, r["job"].config, r["job"].seed))
    return results


def summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        j = r["job"]
        out.append(
            {
                "experiment": j.experiment,
                "config": j.config,
                "seed": j.seed,
                **per_seed(r["rows"]),
                "wall_ms": r["wall_ms"],
            }
        )
    return out


def by_config(summary: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float]]]:
    d: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for row in summary:
        d[row["config"]][row["seed"]] = row
    return d


# -- E1 -------------------------------------------------------------------------------------------


def e1_grid() -> list[tuple[str, str, dict[str, Any]]]:
    """(router family, configuration name, router config); at most 64 per family."""
    grid: list[tuple[str, str, dict[str, Any]]] = []
    for name, t in M.items():
        grid.append(("fixed", f"fixed_{name}", {"type": "fixed", "target": t}))
    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ps, pm in itertools.product(steps, steps):
        pl = round(1.0 - ps - pm, 2)
        if pl < -1e-9:
            continue
        probs = {M["small"]: ps, M["medium"]: pm, M["large"]: pl}
        grid.append(("random_mix", f"mix_{ps:.2f}_{pm:.2f}_{pl:.2f}", {"type": "random_mix", "probs": probs}))
    for chain in CASCADE_CHAINS:
        grid.append(
            (
                "cascade",
                "cascade_" + "-".join(c[0] for c in chain),
                {"type": "cascade", "chain": [M[c] for c in chain]},
            )
        )
    for t1, t2 in itertools.product(THRESHOLD_GRID, THRESHOLD_GRID):
        if t1 <= t2:
            grid.append(
                (
                    "threshold",
                    f"thr_{t1:.1f}_{t2:.1f}",
                    {
                        "type": "threshold",
                        "t1": t1,
                        "t2": t2,
                        "models": [M["small"], M["medium"], M["large"]],
                    },
                )
            )
    return grid


def e1_job(
    name: str, router: dict, seed: int, n: int, variant: dict[str, Any], experiment: str = "e1"
) -> Job:
    return Job(
        experiment=experiment,
        config=name,
        seed=seed,
        workflow="react",
        router=router,
        run_config={"budget": {"deadline_ms": E1_DEADLINE_MS, "max_model_calls": 40}},
        n_tasks=n,
        mix=variant.get("mix", "mixed"),
        hint_sigma=variant.get("hint_sigma", 0.6),
        recall=variant.get("recall", 0.7),
    )


def budget_levels(fixed_summary: dict[str, dict[int, dict[str, float]]]) -> list[float]:
    lo = mean_ci(r["cost_ucu"] for r in fixed_summary["fixed_small"].values())[0]
    hi = mean_ci(r["cost_ucu"] for r in fixed_summary["fixed_large"].values())[0]
    # four levels spaced geometrically strictly inside [cost(small), cost(large)]
    return [round(lo * (hi / lo) ** (i / 5)) for i in range(1, 5)]


def select(
    tuning: dict[str, dict[int, dict[str, float]]], grid: list[tuple[str, str, dict]], levels: list[float]
) -> dict[str, dict[str, str]]:
    """Per router family and budget: highest mean tuning success with mean cost <= B
    (ties: lower cost, then the configuration's index in the grid)."""
    out: dict[str, dict[str, str]] = {}
    index = {name: i for i, (_, name, _) in enumerate(grid)}
    for family in ("random_mix", "cascade", "threshold"):
        out[family] = {}
        for bi, b in enumerate(levels):
            best = None
            for fam, name, _ in grid:
                if fam != family:
                    continue
                s = mean_ci(r["success"] for r in tuning[name].values())[0]
                c = mean_ci(r["cost_ucu"] for r in tuning[name].values())[0]
                if c > b:
                    continue
                key = (-s, c, index[name])
                if best is None or key < best[0]:
                    best = (key, name)
            out[family][f"B{bi + 1}"] = best[1] if best else ""
    return out


def e1_tune(
    workers: int, variant: dict[str, Any] | None = None, levels: list[float] | None = None, n: int = 300
) -> dict[str, Any]:
    variant = variant or {}
    grid = e1_grid()
    exp = "e1tune" + variant.get("tag", "")
    fixed = [g for g in grid if g[0] == "fixed"]
    res_fixed = run_jobs(
        [e1_job(name, r, s, n, variant, exp) for _, name, r in fixed for s in TUNING_SEEDS], workers
    )
    summ = summary_rows(res_fixed)
    if levels is None:
        levels = budget_levels(by_config(summ))  # chosen before any other router has run
    rest = [g for g in grid if g[0] != "fixed"]
    res_rest = run_jobs(
        [e1_job(name, r, s, n, variant, exp) for _, name, r in rest for s in TUNING_SEEDS], workers
    )
    summ += summary_rows(res_rest)
    tuning = by_config(summ)
    selected = select(tuning, grid, levels)
    return {
        "variant": variant,
        "budget_levels": levels,
        "selected": selected,
        "summary": summ,
        "routers": {name: r for _, name, r in grid},
    }


def oracle_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per seed: per task the cheapest of the three fixed runs that succeeded, else the cheapest.
    An upper bound that cannot be implemented (it knows outcomes)."""
    by_seed: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        j = r["job"]
        if j.config.startswith("fixed_"):
            for row in r["rows"]:
                by_seed[j.seed][row["task_id"]].append(row)
    out = []
    for seed in sorted(by_seed):
        rows = []
        for tid in sorted(by_seed[seed]):
            cands = by_seed[seed][tid]
            ok = [c for c in cands if c["success"]]
            pick = min(ok or cands, key=lambda c: (c["cost_ucu"], c["config"]))
            rows.append({**pick, "config": "oracle"})
        out.append({"experiment": "e1", "config": "oracle", "seed": seed, **per_seed(rows), "wall_ms": 0.0})
    return out


def e1_eval(
    frozen: dict[str, Any],
    workers: int,
    seeds: list[int],
    n: int,
    levels: list[str] | None = None,
    experiment: str = "e1",
) -> list[dict[str, Any]]:
    variant = frozen.get("variant", {})
    routers = frozen["routers"]
    names = ["fixed_small", "fixed_medium", "fixed_large"]
    for _family, per_b in frozen["selected"].items():
        for b, name in per_b.items():
            if name and (levels is None or b in levels) and name not in names:
                names.append(name)
    jobs = [e1_job(name, routers[name], s, n, variant, experiment) for name in names for s in seeds]
    results = run_jobs(jobs, workers)
    summ = summary_rows(results) + [dict(r, experiment=experiment) for r in oracle_rows(results)]
    return summ, results


def e1_table(frozen: dict[str, Any], summ: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = by_config(summ)
    rows = []
    levels = frozen["budget_levels"]

    def agg(name: str) -> dict[str, Any]:
        d = cfg[name]
        out = {}
        for m in ("success", "cost_ucu", "latency_p95_ms", "escalations"):
            mean, half, n = mean_ci(r[m] for r in d.values())
            out[m], out[m + "_ci"] = mean, half
        out["n_seeds"] = len(d)
        return out

    for name in ("fixed_small", "fixed_medium", "fixed_large", "oracle"):
        rows.append(
            {"router": name, "budget": "", "B_ucu": "", "config": name, **agg(name), "over_budget": ""}
        )
    for bi, b in enumerate(levels):
        bk = f"B{bi + 1}"
        mix = frozen["selected"]["random_mix"].get(bk)
        for family in ("random_mix", "cascade", "threshold"):
            name = frozen["selected"][family].get(bk)
            if not name:
                rows.append({"router": family, "budget": bk, "B_ucu": b, "config": "(none within B)"})
                continue
            if name not in cfg:  # this budget level was not evaluated in this run (e.g. the quick benchmark)
                continue
            a = agg(name)
            row = {
                "router": family,
                "budget": bk,
                "B_ucu": b,
                "config": name,
                **a,
                "over_budget": int(a["cost_ucu"] > 1.05 * b),
            }
            if family != "random_mix" and mix and mix in cfg:
                d, h, _ = paired(
                    {s: r["success"] for s, r in cfg[mix].items()},
                    {s: r["success"] for s, r in cfg[name].items()},
                )
                dc, hc, _ = paired(
                    {s: r["cost_ucu"] for s, r in cfg[mix].items()},
                    {s: r["cost_ucu"] for s, r in cfg[name].items()},
                )
                w = wtl(cfg[mix], cfg[name], SUCCESS, [COST])
                row.update(
                    {
                        "d_success_vs_mix": d,
                        "d_success_vs_mix_ci": h,
                        "d_cost_vs_mix": dc,
                        "d_cost_vs_mix_ci": hc,
                        "win": w["win"],
                        "tie": w["tie"],
                        "loss": w["loss"],
                    }
                )
            rows.append(row)
    return rows


# -- E2 -------------------------------------------------------------------------------------------

E2_CONFIGS = {
    "no_retry": {"retry": {"max_attempts": 1}},
    "retry": {},
    "retry_fallback": {"fallback": ["sim-b/medium"]},
}
E2_PROFILES = ("calm", "flaky", "brownout")


def e2_jobs(seeds: list[int], duration_s: int = 600) -> list[Job]:
    jobs = []
    for prof in E2_PROFILES:
        for cname, rc in E2_CONFIGS.items():
            for s in seeds:
                jobs.append(
                    Job(
                        experiment="e2",
                        config=f"{prof}/{cname}",
                        seed=s,
                        workflow="react",
                        router={"type": "fixed", "target": M["medium"]},
                        run_config={
                            "timeout_ms": 10_000,
                            "budget": {"deadline_ms": None, "max_model_calls": 40},
                            **rc,
                        },
                        providers=("sim-a", "sim-b"),
                        faults={"sim-a": prof, "sim-b": "calm"},
                        arrivals={"rate_per_s": 0.5, "duration_s": duration_s, "max_in_flight": 16},
                    )
                )
    return jobs


def e2_windows(results: list[dict[str, Any]], window_ms: int = 30_000) -> list[dict[str, Any]]:
    out = []
    per: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        j = r["job"]
        if not j.config.startswith("brownout/"):
            continue
        for row in r["rows"]:
            per[(j.config, row["arrival_ms"] // window_ms)][j.seed].append(row["success"])
    for (cfgname, w), seeds in sorted(per.items()):
        mean, half, n = mean_ci(sum(v) / len(v) for v in seeds.values())
        out.append(
            {
                "config": cfgname,
                "window_start_s": w * window_ms // 1000,
                "success": mean,
                "success_ci": half,
                "n_seeds": n,
            }
        )
    return out


# -- E3 -------------------------------------------------------------------------------------------

E3_KS = (1, 2, 4, 8)
E3_FAIL = (0.0, 0.1)


def e3_jobs(seeds: list[int], n: int) -> list[Job]:
    jobs = []
    for fail in E3_FAIL:
        configs = [("react", 4)] + [
            (mode, k) for mode in ("react_parallel", "plan_execute", "plan_replan") for k in E3_KS
        ]
        for mode, k in configs:
            name = f"f{int(fail * 100):02d}/{mode}" + ("" if mode == "react" else f"/k{k}")
            for s in seeds:
                jobs.append(
                    Job(
                        experiment="e3",
                        config=name,
                        seed=s,
                        workflow=mode,
                        router={"type": "fixed", "target": M["medium"]},
                        run_config={"concurrency": k, "budget": {"deadline_ms": None, "max_model_calls": 40}},
                        n_tasks=n,
                        shape="wide",
                        tool_fail_rate=fail,
                    )
                )
    return jobs


# -- tables -----------------------------------------------------------------------------------------


def aggregate(summ: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for name, d in sorted(by_config(summ).items()):
        row: dict[str, Any] = {"config": name, "n_seeds": len(d)}
        for m in metrics:
            mean, half, _ = mean_ci(r[m] for r in d.values())
            row[m], row[m + "_ci"] = mean, half
        out.append(row)
    return out


def paired_table(
    summ: list[dict[str, Any]], ref_of: Any, metrics: tuple[str, ...], primary: Any, guards: list
) -> list[dict[str, Any]]:
    cfg = by_config(summ)
    out = []
    for name in sorted(cfg):
        ref = ref_of(name)
        if ref is None or ref == name or ref not in cfg:
            continue
        row: dict[str, Any] = {"config": name, "reference": ref}
        for m in metrics:
            d, h, n = paired({s: r[m] for s, r in cfg[ref].items()}, {s: r[m] for s, r in cfg[name].items()})
            row[f"d_{m}"], row[f"d_{m}_ci"] = d, h
        row.update(wtl(cfg[ref], cfg[name], primary, guards))
        out.append(row)
    return out


def informativeness(
    summ: list[dict[str, Any]], metric: str, groups: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Per group of configurations: spread of the mean metric across configurations vs the
    largest standard error across seeds."""
    cfg = by_config(summ)
    out = []
    for g, names in groups.items():
        names = [n for n in names if n in cfg]
        means = [mean_ci(r[metric] for r in cfg[n].values())[0] for n in names]
        ses = [std_error(r[metric] for r in cfg[n].values()) for n in names]
        spread = max(means) - min(means) if means else float("nan")
        se = max(ses) if ses else float("nan")
        out.append(
            {
                "group": g,
                "metric": metric,
                "configs": len(names),
                "spread": spread,
                "max_se": se,
                "separates": int(spread > se),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or sorted(
        {k for r in rows for k in r},
        key=lambda k: (k not in ("experiment", "config", "seed", "router", "budget"), k),
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6g}"
    return v


def write_task_rows(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [row for r in results for row in r["rows"]]
    write_csv(path, rows, TASK_COLUMNS)


def default_workers() -> int:
    return int(os.environ.get("AGENT_RUNTIME_WORKERS", "4"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["E2_CONFIGS", "LAT_P95", "MAKESPAN_P50"]
