"""E6, the delegation study (Tier 2 item 5).

Task set ``mixed``, shape ``wide``, result_tokens 1500, cache off, 200 tasks per seed. Configurations:
``react_small_elide`` (fixed small with the fallback medium and the elide parameters selected in E5
for (small, 1500)), ``react_medium``, ``react_parallel_medium_k4``, and ``delegate`` (a medium planning
call; the plan's branches run as child ``react`` agents on small with the fallback medium; the parent
runs the plan's tail and writes the answer from the children's texts; a child's text carries its
branch's results with probability ``keep_delegate`` = 0.95). Reference: ``react_medium``. Sensitivity:
``keep_delegate`` 0.8 and 1.0 on seeds 101-110. Everything is simulated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_runtime.evals import experiments as X
from agent_runtime.evals.simrun import Job
from agent_runtime.evals.stats import COST, MAKESPAN_P50, SUCCESS

METRICS = (
    "success",
    "cost_ucu",
    "latency_p50_ms",
    "latency_p95_ms",
    "model_calls",
    "peak_parent_context_tokens",
    "subagent_runs",
    "medium_calls",
)
SMALL = {"type": "fixed", "target": "sim-a/small", "fallback": ["sim-a/medium"]}
MEDIUM = {"type": "fixed", "target": "sim-a/medium"}


def elide_params(root: Path) -> dict[str, Any]:
    path = Path(root) / "configs" / "e5_frozen.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        return dict(frozen["grid"][frozen["selected"]["small/r1500"]["elide"]])
    return {"policy": "elide", "keep_last": 2, "compact_to": 0.5, "compact_at": 0.75}


def configs(root: Path) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "react_small_elide",
            {"workflow": "react", "router": SMALL, "run_config": {"context": elide_params(root)}},
        ),
        ("react_medium", {"workflow": "react", "router": MEDIUM, "run_config": {}}),
        (
            "react_parallel_medium_k4",
            {"workflow": "react_parallel", "router": MEDIUM, "run_config": {"concurrency": 4}},
        ),
        ("delegate", {"workflow": "delegate_plan", "router": MEDIUM, "run_config": {}}),
    ]


def job(
    name: str, cfg: dict[str, Any], seed: int, n: int, keep_delegate: float = 0.95, experiment: str = "e6"
) -> Job:
    rc = {"budget": {"deadline_ms": None, "max_model_calls": 40}, **cfg["run_config"]}
    return Job(
        experiment=experiment,
        config=name,
        seed=seed,
        workflow=cfg["workflow"],
        router=cfg["router"],
        run_config=rc,
        n_tasks=n,
        shape="wide",
        result_tokens=1500,
        cache=False,
        summarizer="sim-a/small",
        keep_delegate=keep_delegate,
    )


def eval_jobs(root: Path, seeds: list[int], n: int) -> list[Job]:
    return [job(name, cfg, s, n) for name, cfg in configs(root) for s in seeds]


def sensitivity_jobs(root: Path, seeds: list[int], n: int) -> list[Job]:
    cfg = dict(configs(root))
    out = []
    for keep in (0.8, 1.0):
        for name in ("react_medium", "delegate"):
            for s in seeds:
                out.append(
                    job(
                        f"keep{keep}/{name}", cfg[name], s, n, keep_delegate=keep, experiment="e6_sensitivity"
                    )
                )
    return out


def tables(summ: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def ref_of(name: str) -> str:
        return name.rsplit("/", 1)[0] + "/react_medium" if "/" in name else "react_medium"

    return X.aggregate(summ, METRICS), X.paired_table(
        summ, ref_of, ("success", "cost_ucu", "latency_p50_ms"), SUCCESS, [COST, MAKESPAN_P50]
    )
