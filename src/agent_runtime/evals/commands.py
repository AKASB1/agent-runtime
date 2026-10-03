"""``python -m agent_runtime eval ...``: tuning, the informativeness check, E1-E3 (simulated), E4a,
the quick and the full benchmark. Writes small CSV files and manifests under ``results/`` and
per-task rows under the ignored ``eval-results/``."""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_runtime.canon import digest
from agent_runtime.clock import wall_ms
from agent_runtime.evals import experiments as X
from agent_runtime.evals.stats import COST, LAT_P95, MAKESPAN_P50, SUCCESS

E2_METRICS = (
    "success",
    "latency_p95_ms",
    "cost_ucu",
    "provider_calls",
    "wasted_calls",
    "retries",
    "fallbacks",
    "failed_runs",
)
E3_METRICS = (
    "success",
    "latency_p50_ms",
    "latency_p95_ms",
    "model_calls",
    "cost_ucu",
    "bound_gap",
    "duplicate_writes",
    "replans",
)
E1_METRICS = ("success", "cost_ucu", "latency_p95_ms", "escalations")
SENSITIVITY = {
    "hint_sigma_0.3": {"hint_sigma": 0.3},
    "hint_sigma_1.2": {"hint_sigma": 1.2},
    "recall_0.5": {"recall": 0.5},
    "recall_0.9": {"recall": 0.9},
    "hard_heavy": {"mix": "hard_heavy"},
}


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# the commit at the start of a run (Ctx): a run rewrites tracked result files before it writes a
# manifest, so a check at that point would call the run's own output a dirty tree
_START_COMMIT: dict[Path, str] = {}


def manifest(
    root: Path,
    name: str,
    *,
    seeds: list[int],
    config: Any,
    workers: int,
    wall: float,
    load_note: str,
    tasks: int,
) -> dict:
    libs = {}
    for lib in ("pydantic", "httpx", "numpy", "scipy", "matplotlib", "mcp", "fastapi"):
        try:
            libs[lib] = md.version(lib)
        except md.PackageNotFoundError:
            libs[lib] = None
    return {
        "experiment": name,
        "commit": _START_COMMIT.get(root) or git_commit(root),
        "config_sha256": digest(config),
        "seeds": seeds,
        "python": platform.python_version(),
        "libraries": libs,
        "platform": platform.platform(),
        "workers": workers,
        "load_note": load_note,
        "wall_s": round(wall / 1000.0, 1),
        "task_runs": tasks,
        "tasks_per_s_wall": round(tasks / (wall / 1000.0), 1) if wall > 0 else None,
        "label": "simulated",
    }


def _cfg_jobs(jobs: list[X.Job]) -> list[dict]:
    return [dict(vars(j)) for j in jobs]


class Ctx:
    def __init__(self, root: Path, workers: int, load_note: str, out: str = "results") -> None:
        self.root = root
        self.workers = workers
        self.load_note = load_note
        self.res = root / out
        self.raw = root / "eval-results"
        _START_COMMIT[root] = git_commit(root)

    def log(self, msg: str) -> None:
        print(msg, flush=True)


# -- E1 -------------------------------------------------------------------------------------------


def e1_tune(c: Ctx, variant_name: str | None = None, levels: list[float] | None = None) -> dict:
    variant = dict(SENSITIVITY[variant_name], tag=f"_{variant_name}") if variant_name else {}
    t0 = wall_ms()
    frozen = X.e1_tune(c.workers, variant, levels)
    wall = wall_ms() - t0
    sub = f"e1_sensitivity/{variant_name}" if variant_name else "e1"
    X.write_csv(c.res / sub / "tuning_summary.csv", frozen["summary"])
    out = {k: v for k, v in frozen.items() if k != "summary"}
    cfg_dir = c.root / "configs"
    X.dump_json(cfg_dir / (f"e1_frozen_{variant_name}.json" if variant_name else "e1_frozen.json"), out)
    n = len(frozen["summary"]) * 300
    X.dump_json(
        c.res / sub / "manifest_tuning.json",
        manifest(
            c.root,
            f"{sub}-tuning",
            seeds=X.TUNING_SEEDS,
            config=out,
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=n,
        ),
    )
    c.log(
        f"[{sub}] tuned: B={out['budget_levels']} selected={json.dumps(out['selected'])} ({wall / 1000:.1f}s)"
    )
    return out


def e1_eval(
    c: Ctx,
    *,
    seeds: list[int],
    n: int,
    levels: list[str] | None,
    sub: str = "e1",
    frozen_file: str = "e1_frozen.json",
) -> None:
    frozen = json.loads((c.root / "configs" / frozen_file).read_text(encoding="utf-8"))
    t0 = wall_ms()
    summ, results = X.e1_eval(frozen, c.workers, seeds, n, levels)
    wall = wall_ms() - t0
    X.write_csv(c.res / sub / "summary.csv", summ)
    X.write_csv(c.res / sub / "table.csv", X.e1_table(frozen, summ))
    X.write_task_rows(c.raw / sub / "tasks.csv", results)
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root,
            sub,
            seeds=seeds,
            config={"frozen": frozen, "n": n, "levels": levels},
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=sum(len(r["rows"]) for r in results),
        ),
    )
    c.log(f"[{sub}] {len(results)} jobs, {wall / 1000:.1f}s")


def e1_sensitivity(c: Ctx, level: str = "B3") -> None:
    main = json.loads((c.root / "configs" / "e1_frozen.json").read_text(encoding="utf-8"))
    bi = int(level[1:]) - 1
    for name in SENSITIVITY:
        frozen = e1_tune(c, name, levels=main["budget_levels"])
        frozen["selected"] = {fam: {level: per_b.get(level, "")} for fam, per_b in frozen["selected"].items()}
        X.dump_json(c.root / "configs" / f"e1_frozen_{name}.json", frozen)
        e1_eval(
            c,
            seeds=X.SENS_SEEDS,
            n=300,
            levels=[level],
            sub=f"e1_sensitivity/{name}",
            frozen_file=f"e1_frozen_{name}.json",
        )
    _ = bi


# -- E2 / E3 --------------------------------------------------------------------------------------


def e2(c: Ctx, *, seeds: list[int], duration_s: int, sub: str = "e2") -> None:
    jobs = X.e2_jobs(seeds, duration_s)
    t0 = wall_ms()
    results = X.run_jobs(jobs, c.workers)
    wall = wall_ms() - t0
    summ = X.summary_rows(results)
    X.write_csv(c.res / sub / "summary.csv", summ)
    X.write_csv(c.res / sub / "aggregate.csv", X.aggregate(summ, E2_METRICS))
    X.write_csv(
        c.res / sub / "paired.csv",
        X.paired_table(summ, lambda n: n.split("/")[0] + "/retry", E2_METRICS, SUCCESS, [LAT_P95, COST]),
    )
    X.write_csv(c.res / sub / "brownout_windows.csv", X.e2_windows(results))
    X.write_task_rows(c.raw / sub / "tasks.csv", results)
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root,
            sub,
            seeds=seeds,
            config=_cfg_jobs(jobs[:9]),
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=sum(len(r["rows"]) for r in results),
        ),
    )
    c.log(f"[{sub}] {len(results)} jobs, {wall / 1000:.1f}s")


def e3(c: Ctx, *, seeds: list[int], n: int, sub: str = "e3") -> None:
    jobs = X.e3_jobs(seeds, n)
    t0 = wall_ms()
    results = X.run_jobs(jobs, c.workers)
    wall = wall_ms() - t0
    summ = X.summary_rows(results)
    X.write_csv(c.res / sub / "summary.csv", summ)
    X.write_csv(c.res / sub / "aggregate.csv", X.aggregate(summ, E3_METRICS))
    X.write_csv(
        c.res / sub / "paired.csv",
        X.paired_table(
            summ, lambda nm: nm.split("/")[0] + "/react", E3_METRICS, MAKESPAN_P50, [SUCCESS, COST]
        ),
    )
    X.write_task_rows(c.raw / sub / "tasks.csv", results)
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root,
            sub,
            seeds=seeds,
            config=_cfg_jobs(jobs[: len(jobs) // len(seeds)]),
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=sum(len(r["rows"]) for r in results),
        ),
    )
    c.log(f"[{sub}] {len(results)} jobs, {wall / 1000:.1f}s")


def informativeness(c: Ctx) -> list[dict]:
    """On the tuning seeds: does each scenario separate its configurations beyond the seed noise?"""
    out = []
    tun = c.res / "e1" / "tuning_summary.csv"
    if tun.exists():
        import csv

        rows = [
            dict(r, seed=int(r["seed"]), success=float(r["success"]), cost_ucu=float(r["cost_ucu"]))
            for r in csv.DictReader(open(tun, encoding="utf-8"))
        ]
        frozen = json.loads((c.root / "configs" / "e1_frozen.json").read_text(encoding="utf-8"))
        for bk in ("B1", "B2", "B3", "B4"):
            names = [frozen["selected"][f].get(bk) for f in ("random_mix", "cascade", "threshold")]
            out += [
                dict(r, experiment="e1", group=f"{bk} selected routers")
                for r in X.informativeness(rows, "success", {bk: [n for n in names if n]})
            ]
        out += [
            dict(r, experiment="e1", group="fixed models")
            for r in X.informativeness(
                rows, "success", {"fixed": ["fixed_small", "fixed_medium", "fixed_large"]}
            )
        ]
    res2 = X.run_jobs(X.e2_jobs(X.TUNING_SEEDS, 600), c.workers)
    s2 = X.summary_rows(res2)
    for prof in X.E2_PROFILES:
        out += [
            dict(r, experiment="e2")
            for r in X.informativeness(s2, "success", {prof: [f"{prof}/{k}" for k in X.E2_CONFIGS]})
        ]
    res3 = X.run_jobs(X.e3_jobs(X.TUNING_SEEDS, 200), c.workers)
    s3 = X.summary_rows(res3)
    for f in ("f00", "f10"):
        names = sorted({r["config"] for r in s3 if r["config"].startswith(f)})
        out += [dict(r, experiment="e3") for r in X.informativeness(s3, "latency_p50_ms", {f: names})]
        out += [
            dict(r, experiment="e3", group=f + " (success)")
            for r in X.informativeness(s3, "success", {f: names})
        ]
    X.write_csv(
        c.res / "informativeness.csv",
        out,
        ["experiment", "group", "metric", "configs", "spread", "max_se", "separates"],
    )
    for r in out:
        c.log(
            f"[informativeness] {r['experiment']} {r['group']}: {r['metric']} spread={r['spread']:.4g} max_se={r['max_se']:.4g} separates={r['separates']}"
        )
    return out


# -- E4a ----------------------------------------------------------------------------------------------


def e4a(c: Ctx, sub: str = "e4") -> None:
    try:
        from agent_runtime.optimization.e4 import run_e4a
    except ImportError:
        c.log("[e4] not available")
        return
    t0 = wall_ms()
    rows = run_e4a(c.root)
    wall = wall_ms() - t0
    X.write_csv(c.res / sub / "table.csv", rows)
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root, sub, seeds=[], config=rows, workers=1, wall=wall, load_note=c.load_note, tasks=len(rows)
        ),
    )
    c.log(f"[{sub}] {len(rows)} problems, {wall / 1000:.1f}s")


def e4b(c: Ctx, sub: str = "e4b") -> None:
    from agent_runtime.optimization.mutants import run_e4b

    t0 = wall_ms()
    rows, summary = run_e4b(c.root)
    wall = wall_ms() - t0
    X.write_csv(c.res / sub / "mutants.csv", rows)
    X.write_csv(
        c.res / sub / "summary.csv",
        summary,
        ["operator", "config", "mutants", "wrong", "equivalent", "detection_rate", "false_reject_rate"],
    )
    missed = [r for r in rows if r["wrong"] and r["all_witnesses_accepted"]]
    X.write_csv(c.res / sub / "missed_by_all.csv", missed, ["problem", "operator", "k", "truth"])
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root,
            sub,
            seeds=[1],
            config={"operators": sorted({r["operator"] for r in rows})},
            workers=1,
            wall=wall,
            load_note=c.load_note,
            tasks=len(rows),
        ),
    )
    c.log(
        f"[{sub}] {len(rows)} mutants, {len(missed)} wrong mutants accepted by every configuration, {wall / 1000:.1f}s"
    )


def e5_tune(c: Ctx) -> dict:
    from agent_runtime.evals import context_study as E5

    t0 = wall_ms()
    frozen = E5.tune(c.workers)
    wall = wall_ms() - t0
    X.write_csv(c.res / "e5" / "tuning_summary.csv", frozen["summary"])
    X.write_csv(
        c.res / "e5" / "informativeness.csv",
        frozen["informativeness"],
        ["group", "metric", "configs", "spread", "max_se", "separates"],
    )
    out = {"selected": frozen["selected"], "grid": frozen["grid"]}
    X.dump_json(c.root / "configs" / "e5_frozen.json", out)
    X.dump_json(
        c.res / "e5" / "manifest_tuning.json",
        manifest(
            c.root,
            "e5-tuning",
            seeds=X.TUNING_SEEDS,
            config=out,
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=len(frozen["summary"]) * 200,
        ),
    )
    c.log(f"[e5] tuned in {wall / 1000:.1f}s: {json.dumps(out['selected'])}")
    return out


def e5(c: Ctx, *, seeds: list[int], n: int, sub: str = "e5") -> None:
    from agent_runtime.evals import context_study as E5

    frozen = json.loads((c.root / "configs" / "e5_frozen.json").read_text(encoding="utf-8"))
    jobs = E5.eval_jobs(frozen, seeds, n)
    t0 = wall_ms()
    results = X.run_jobs(jobs, c.workers)
    wall = wall_ms() - t0
    summ = X.summary_rows(results)
    agg, paired = E5.tables(summ)
    X.write_csv(c.res / sub / "summary.csv", summ)
    X.write_csv(c.res / sub / "aggregate.csv", agg)
    X.write_csv(c.res / sub / "paired.csv", paired)
    X.write_task_rows(c.raw / sub / "tasks.csv", results)
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root,
            sub,
            seeds=seeds,
            config=frozen,
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=sum(len(r["rows"]) for r in results),
        ),
    )
    c.log(f"[{sub}] {len(results)} jobs, {wall / 1000:.1f}s")


def e6(
    c: Ctx, *, seeds: list[int], n: int, sensitivity_seeds: list[int] | None = None, sub: str = "e6"
) -> None:
    from agent_runtime.evals import delegation_study as E6

    jobs = E6.eval_jobs(c.root, seeds, n)
    if sensitivity_seeds:
        jobs += E6.sensitivity_jobs(c.root, sensitivity_seeds, n)
    t0 = wall_ms()
    results = X.run_jobs(jobs, c.workers)
    wall = wall_ms() - t0
    summ = X.summary_rows(results)
    agg, paired = E6.tables(summ)
    X.write_csv(c.res / sub / "summary.csv", summ)
    X.write_csv(c.res / sub / "aggregate.csv", agg)
    X.write_csv(c.res / sub / "paired.csv", paired)
    X.write_task_rows(c.raw / sub / "tasks.csv", results)
    cfg = {"configs": E6.configs(c.root), "n": n}
    X.dump_json(
        c.res / sub / "manifest.json",
        manifest(
            c.root,
            sub,
            seeds=seeds,
            config=cfg,
            workers=c.workers,
            wall=wall,
            load_note=c.load_note,
            tasks=sum(len(r["rows"]) for r in results),
        ),
    )
    c.log(f"[{sub}] {len(results)} jobs, {wall / 1000:.1f}s")


# -- entry points -------------------------------------------------------------------------------------


def quick(c: Ctx) -> None:
    c.res = c.root / "results" / "quick"
    c.raw = c.root / "eval-results" / "quick"
    t0 = wall_ms()
    e1_eval(c, seeds=X.QUICK_SEEDS, n=100, levels=["B3"])
    e2(c, seeds=X.QUICK_SEEDS, duration_s=200)
    e3(c, seeds=X.QUICK_SEEDS, n=100)
    e4a(c)
    c.log(f"[quick] done in {(wall_ms() - t0) / 1000:.1f}s (simulated results; see results/quick/)")


def full(c: Ctx, *, sensitivity: bool = True) -> None:
    t0 = wall_ms()
    e1_eval(c, seeds=X.EVAL_SEEDS, n=300, levels=None)
    e2(c, seeds=X.EVAL_SEEDS, duration_s=600)
    e3(c, seeds=X.EVAL_SEEDS, n=200)
    e4a(c)
    e4b(c)
    if (c.root / "configs" / "e5_frozen.json").exists():
        e5(c, seeds=X.EVAL_SEEDS, n=200)
    e6(c, seeds=X.EVAL_SEEDS, n=200, sensitivity_seeds=X.SENS_SEEDS)
    if sensitivity:
        e1_sensitivity(c)
    c.log(f"[full] done in {(wall_ms() - t0) / 1000:.1f}s")


def main(args: Any) -> int:
    root = Path(args.root).resolve()
    c = Ctx(root, args.workers, args.load_note)
    if args.quick:
        quick(c)
    elif args.full:
        full(c, sensitivity=not args.no_sensitivity)
    elif args.experiment == "e1-tune":
        e1_tune(c)
    elif args.experiment == "informativeness":
        informativeness(c)
    elif args.experiment == "e1":
        e1_eval(c, seeds=X.EVAL_SEEDS, n=300, levels=None)
    elif args.experiment == "e1-sensitivity":
        e1_sensitivity(c)
    elif args.experiment == "e2":
        e2(c, seeds=X.EVAL_SEEDS, duration_s=600)
    elif args.experiment == "e3":
        e3(c, seeds=X.EVAL_SEEDS, n=200)
    elif args.experiment == "e4":
        e4a(c)
    elif args.experiment == "e4b":
        e4b(c)
    elif args.experiment == "e5-tune":
        e5_tune(c)
    elif args.experiment == "e5":
        e5(c, seeds=X.EVAL_SEEDS, n=200)
    elif args.experiment == "e6":
        e6(c, seeds=X.EVAL_SEEDS, n=200, sensitivity_seeds=X.SENS_SEEDS)
    elif args.live:
        print(
            "not run: no endpoint configured"
            if not _live_configured()
            else "use scripts/live_smoke.py for the live smoke test"
        )
    else:
        print("nothing to do: use --quick, --full, --experiment NAME, or --live", file=sys.stderr)
        return 2
    return 0


def _live_configured() -> bool:
    import os

    return all(
        os.environ.get(v)
        for v in ("AGENT_RUNTIME_LIVE_BASE_URL", "AGENT_RUNTIME_LIVE_API_KEY", "AGENT_RUNTIME_LIVE_MODELS")
    )
