"""Aggregation (TASK section 3, Metrics): per seed the mean over tasks and P50/P95 of the latency;
across seeds the mean and the 95% Student-t interval; paired differences against a reference on
common seeds; win/tie/loss per seed with fixed tie bands and the dominance rule of PLAN.md."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import stats as sps

MEAN_METRICS = (
    "success", "accepted", "cost_ucu", "latency_ms", "model_calls", "tool_calls", "provider_calls", "retries",
    "fallbacks", "repairs", "replans", "escalations", "peak_context_tokens", "compactions", "repeat_reads",
    "cache_read_tokens", "input_tokens", "subagent_runs", "duplicate_writes", "wrong_writes", "bound_gap",
    "medium_calls", "context_length_failures", "peak_parent_context_tokens",
)  # fmt: skip


def per_seed(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summary of one (experiment, configuration, seed)."""
    out: dict[str, float] = {"n_tasks": len(rows)}
    for m in MEAN_METRICS:
        vals = [
            float(r[m])
            for r in rows
            if r[m] is not None and not (isinstance(r[m], float) and math.isnan(r[m]))
        ]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    lat = np.array([r["latency_ms"] for r in rows], dtype=float)
    out["latency_p50_ms"] = float(np.percentile(lat, 50)) if len(lat) else float("nan")
    out["latency_p95_ms"] = float(np.percentile(lat, 95)) if len(lat) else float("nan")
    out["wasted_calls"] = (
        float(np.mean([r["provider_calls"] - r["model_calls"] for r in rows])) if rows else float("nan")
    )
    out["failed_runs"] = float(np.mean([r["status"] != "succeeded" for r in rows])) if rows else float("nan")
    total_in = sum(float(r.get("input_tokens", 0) or 0) for r in rows)
    out["cache_share"] = (
        sum(float(r.get("cache_read_tokens", 0) or 0) for r in rows) / total_in if total_in else 0.0
    )
    return out


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    """(mean, half-width of the 95% Student-t interval, n)."""
    v = np.array([x for x in values if not math.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    half = float(sps.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))
    return float(v.mean()), half, n


def std_error(values: Iterable[float]) -> float:
    v = np.array([x for x in values if not math.isnan(x)], dtype=float)
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def paired(ref: dict[int, float], other: dict[int, float]) -> tuple[float, float, int]:
    seeds = sorted(set(ref) & set(other))
    return mean_ci(other[s] - ref[s] for s in seeds)


def in_band(diff: float, ref: float, band: float, relative: bool) -> bool:
    limit = band * abs(ref) if relative else band
    return abs(diff) <= limit


def wtl(
    ref: dict[int, dict[str, float]],
    other: dict[int, dict[str, float]],
    primary: tuple[str, float, bool, bool],
    guards: list[tuple[str, float, bool, bool]],
) -> dict[str, int]:
    """Win/tie/loss per seed. A metric spec is (name, band, relative, higher_is_better).

    Win: better on the primary metric beyond its band and not worse beyond its band on any guard.
    Loss: worse on the primary beyond its band, or within its band on the primary and worse beyond
    the band on a guard (dominated). Otherwise tie (including a trade-off: better on the primary,
    worse on a guard).
    """
    out = {"win": 0, "tie": 0, "loss": 0}
    for s in sorted(set(ref) & set(other)):

        def cmp(spec: tuple[str, float, bool, bool], s: int = s) -> int:
            name, band, rel, hib = spec
            d = other[s][name] - ref[s][name]
            if in_band(d, ref[s][name], band, rel):
                return 0
            better = d > 0 if hib else d < 0
            return 1 if better else -1

        p = cmp(primary)
        worse_guard = any(cmp(g) < 0 for g in guards)
        if p > 0 and not worse_guard:
            out["win"] += 1
        elif p < 0 or (p == 0 and worse_guard):
            out["loss"] += 1
        else:
            out["tie"] += 1
    return out


SUCCESS = ("success", 0.01, False, True)
COST = ("cost_ucu", 0.02, True, False)
LAT_P95 = ("latency_p95_ms", 0.02, True, False)
MAKESPAN_P50 = ("latency_p50_ms", 0.02, True, False)
