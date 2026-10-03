"""E5, the context study (Tier 2 item 1).

Task set ``mixed``, shape ``chain``, planner ``react``, ``sim-a`` without faults, 200 tasks per seed.
Setups: ``small`` = fixed(sim-a/small) with the fallback sim-a/medium (a step moves to medium when
small is not offered because its window cannot hold the part of the input the policy may not drop)
and ``medium`` = fixed(sim-a/medium); summaries by small in both. Result sizes 250, 1500 (primary),
3000; the prefix cache off and on. Configurations: ``none``; ``window``, ``elide``, ``summarize`` each
with keep_last {1,2,3} x compact_to {0.4,0.5,0.6} (compact_at 0.75), tuned on the tuning seeds per
(setup, size) with the cache off (highest mean success; ties: lower cost, then index), frozen, then
evaluated; plus the ablation ``elide`` with compact_to = compact_at (compaction at every step).
Reference for the paired differences: ``none``. Everything is simulated.
"""

from __future__ import annotations

import itertools
from typing import Any

from agent_runtime.evals import experiments as X
from agent_runtime.evals.simrun import Job
from agent_runtime.evals.stats import COST, LAT_P95, SUCCESS, mean_ci

SETUPS: dict[str, dict[str, Any]] = {
    "small": {"router": {"type": "fixed", "target": "sim-a/small", "fallback": ["sim-a/medium"]}},
    "medium": {"router": {"type": "fixed", "target": "sim-a/medium"}},
}
SIZES = (250, 1500, 3000)
POLICIES = ("window", "elide", "summarize")
KEEP_LAST = (1, 2, 3)
COMPACT_TO = (0.4, 0.5, 0.6)
METRICS = (
    "success",
    "cost_ucu",
    "latency_p95_ms",
    "model_calls",
    "medium_calls",
    "context_length_failures",
    "peak_context_tokens",
    "compactions",
    "repeat_reads",
    "cache_share",
)


def grid() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = [("none", {"policy": "none"})]
    for pol, k, t in itertools.product(POLICIES, KEEP_LAST, COMPACT_TO):
        out.append((f"{pol}_k{k}_t{t}", {"policy": pol, "keep_last": k, "compact_to": t, "compact_at": 0.75}))
    return out


def job(
    setup: str,
    size: int,
    cache: bool,
    name: str,
    ctx: dict[str, Any],
    seed: int,
    n: int,
    experiment: str = "e5",
) -> Job:
    return Job(
        experiment=experiment,
        config=f"{setup}/r{size}/{'cache' if cache else 'nocache'}/{name}",
        seed=seed,
        workflow="react",
        router=SETUPS[setup]["router"],
        run_config={"context": ctx, "budget": {"deadline_ms": None, "max_model_calls": 40}},
        n_tasks=n,
        result_tokens=size,
        cache=cache,
        summarizer="sim-a/small",
    )


def tune(workers: int, n: int = 200) -> dict[str, Any]:
    g = grid()
    jobs = [
        job(s, r, False, name, ctx, seed, n, "e5tune")
        for s in SETUPS
        for r in SIZES
        for name, ctx in g
        for seed in X.TUNING_SEEDS
    ]
    summ = X.summary_rows(X.run_jobs(jobs, workers))
    cfg = X.by_config(summ)
    index = {name: i for i, (name, _) in enumerate(g)}
    selected: dict[str, dict[str, str]] = {}
    informative = []
    for s in SETUPS:
        for r in SIZES:
            key = f"{s}/r{r}"
            selected[key] = {}
            for pol in POLICIES:
                best = None
                for name, _ in g:
                    if not name.startswith(pol + "_"):
                        continue
                    d = cfg[f"{key}/nocache/{name}"]
                    sc = mean_ci(x["success"] for x in d.values())[0]
                    co = mean_ci(x["cost_ucu"] for x in d.values())[0]
                    k = (-sc, co, index[name])
                    if best is None or k < best[0]:
                        best = (k, name)
                selected[key][pol] = best[1]
            names = [f"{key}/nocache/none"] + [f"{key}/nocache/{selected[key][p]}" for p in POLICIES]
            informative += X.informativeness(summ, "success", {key: names})
    return {"selected": selected, "grid": dict(g), "summary": summ, "informativeness": informative}


def eval_jobs(frozen: dict[str, Any], seeds: list[int], n: int, experiment: str = "e5") -> list[Job]:
    g = frozen["grid"]
    jobs = []
    for s in SETUPS:
        for r in SIZES:
            sel = frozen["selected"][f"{s}/r{r}"]
            elide = g[sel["elide"]]
            configs = [("none", g["none"])] + [(sel[p], g[sel[p]]) for p in POLICIES]
            configs.append(
                (
                    f"ablation_elide_k{elide['keep_last']}_every_step",
                    {**elide, "compact_to": 0.75, "compact_at": 0.75},
                )
            )
            for cache in (False, True):
                for name, ctx in configs:
                    for seed in seeds:
                        jobs.append(job(s, r, cache, name, ctx, seed, n, experiment))
    return jobs


def ref_of(name: str) -> str:
    return "/".join(name.split("/")[:3]) + "/none"


def tables(summ: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return X.aggregate(summ, METRICS), X.paired_table(
        summ, ref_of, ("success", "cost_ucu", "latency_p95_ms"), SUCCESS, [COST, LAT_P95]
    )
