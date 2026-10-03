"""Figures from committed files only (matplotlib, PNG under docs/figures/, each below 300 KB).

    python scripts/plot_results.py [--results results] [--out docs/figures]

- framework.png: the harness layers and their status (built, Tier 2, absent)
- timeline_plan_execute.png: one plan_execute run from examples/traces/plan_execute_wide.jsonl
- e1_success_vs_cost.png, e2_brownout_windows.png, e3_makespan_vs_k.png: simulated results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# the reference categorical palette, in its fixed order; text stays in ink colours
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b3"
MARKERS = ["o", "s", "^", "D"]
SIMULATED = "simulated: accuracies, prices, and latencies are assumptions"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    size = (out / name).stat().st_size
    assert size < 300_000, f"{name} is {size} bytes"
    print(f"wrote {out / name} ({size // 1024} KB)")


# -- framework ----------------------------------------------------------------------------------

LAYERS = [
    ("Agent API", "FastAPI service, CLI", "built"),
    ("Agent loop", "step-graph executor, four planner modes, agent workflow", "built"),
    ("Model adapters", "scripted, simulated, OpenAI-compatible (openai, deepseek)", "built"),
    ("Context manager", "session log, derived input, none/window/elide/summarize, fork", "built"),
    ("Tool runtime", "typed native tools, MCP over stdio, workspace tools", "built"),
    ("Permissions", "sandbox + approval, path/command/host policy (in process)", "built"),
    ("Subagents", "delegate, catalog, allowlist, budget share, nested spans", "built"),
    ("Session and state", "SQLite store, append-only log, versioned state", "built"),
    ("Retry and recovery", "retry, fallback, repair, budgets, cancellation", "built"),
    ("Observability", "trace with replay, GenAI attribute names", "built"),
    ("Skills", "catalog in the prompt, load_skill (Tier 2 item, built)", "built"),
    ("Breaker, hedging, resume", "", "tier2"),
    ("Anthropic adapter", "", "tier2"),
    ("OTel export, PostgreSQL", "", "tier2"),
    ("Retrieval, memory", "scaffold protocol only", "tier2"),
    ("Plugins, hooks, web UI", "out of scope on purpose", "absent"),
]
STATUS = {
    "built": (BLUE, "built (Tier 1)"),
    "tier2": (YELLOW, "Tier 2: not built"),
    "absent": (NEUTRAL, "absent on purpose"),
}


def framework(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.set_axis_off()
    n = len(LAYERS)
    for i, (name, detail, st) in enumerate(LAYERS):
        y = n - 1 - i
        color, _ = STATUS[st]
        ax.add_patch(
            FancyBboxPatch(
                (0.0, y + 0.08), 0.06, 0.84, boxstyle="round,pad=0,rounding_size=0.03", fc=color, ec="none"
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.08, y + 0.08),
                0.92,
                0.84,
                boxstyle="round,pad=0,rounding_size=0.03",
                fc="#f3f2ef",
                ec="none",
            )
        )
        ax.text(0.10, y + 0.5, name, va="center", ha="left", fontsize=9, fontweight="bold", color=INK)
        ax.text(0.40, y + 0.5, detail, va="center", ha="left", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, n + 0.9)
    for k, (color, label) in enumerate(STATUS.values()):
        ax.add_patch(
            FancyBboxPatch(
                (0.02 + k * 0.3, n + 0.25),
                0.03,
                0.45,
                boxstyle="round,pad=0,rounding_size=0.02",
                fc=color,
                ec="none",
            )
        )
        ax.text(0.06 + k * 0.3, n + 0.47, label, va="center", fontsize=8.5, color=INK)
    ax.set_title("Harness layers of agent-runtime and their status", loc="left", pad=4)
    save(fig, out, "framework.png")


# -- timeline -----------------------------------------------------------------------------------


def timeline(out: Path) -> None:
    path = ROOT / "examples" / "traces" / "plan_execute_wide.jsonl"
    spans = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [s for s in spans if s["kind"] in ("model", "tool")]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.3))
    labels = []
    for i, s in enumerate(rows):
        y = len(rows) - 1 - i
        if s["kind"] == "model":
            color, label = BLUE, f"model: {s['attrs'].get('agent_runtime.role')}"
        else:
            color, label = ORANGE, f"tool: {s['name']} ({s['attrs']['gen_ai.tool.call.id'].split('#')[-1]})"
        width = max((s["end_ms"] - s["start_ms"]) / 1000, 0.03)  # keep very short calls visible
        ax.barh(y, width, left=s["start_ms"] / 1000, height=0.62, color=color, linewidth=0)
        labels.append((y, label))
    ax.set_yticks([y for y, _ in labels], [lab for _, lab in labels])
    ax.set_xlabel("virtual time since run start (s)")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "One plan_execute run (k = 4): independent tool steps run in parallel  [" + SIMULATED + "]",
        loc="left",
        fontsize=8.5,
    )
    save(fig, out, "timeline_plan_execute.png")


# -- E1 -----------------------------------------------------------------------------------------------


def e1(results: Path, out: Path) -> None:
    table = read_csv(results / "e1" / "table.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fams = [("random_mix", BLUE, "o"), ("cascade", ORANGE, "s"), ("threshold", AQUA, "^")]
    for fam, color, marker in fams:
        pts = [r for r in table if r["router"] == fam and r.get("cost_ucu")]
        if pts:
            ax.errorbar(
                [num(r["cost_ucu"]) / 1000 for r in pts],
                [num(r["success"]) for r in pts],
                xerr=[num(r["cost_ucu_ci"]) / 1000 for r in pts],
                yerr=[num(r["success_ci"]) for r in pts],
                fmt=marker,
                color=color,
                ms=8,
                mec=SURFACE,
                mew=2,
                capsize=3,
                lw=1.2,
                label=f"{fam} (tuned per budget)",
            )
    for r in table:
        if r["router"].startswith("fixed_"):
            x, y = num(r["cost_ucu"]) / 1000, num(r["success"])
            ax.errorbar(
                [x],
                [y],
                yerr=[num(r["success_ci"])],
                fmt="D",
                color=INK2,
                ms=7,
                mec=SURFACE,
                mew=2,
                capsize=3,
                lw=1.2,
            )
            ax.annotate(
                r["router"].replace("fixed_", "fixed(") + ")",
                (x, y),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=8,
                color=INK2,
            )
        if r["router"] == "oracle":
            x, y = num(r["cost_ucu"]) / 1000, num(r["success"])
            ax.plot(
                [x],
                [y],
                marker="*",
                ms=12,
                mfc="none",
                mec=INK,
                mew=1.4,
                ls="none",
                label="oracle (upper bound, not deployable)",
            )
    levels = sorted({num(r["B_ucu"]) for r in table if r.get("B_ucu")})
    for i, b in enumerate(levels):
        ax.axvline(b / 1000, color=NEUTRAL, lw=1, ls="--", zorder=0)
        ax.text(
            b / 1000,
            0.0,
            f" B{i + 1}",
            color=INK2,
            fontsize=8,
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xscale("log")
    ax.set_xlabel("realized mean cost per task (mcu, log scale)")
    ax.set_ylabel("success rate")
    n = max(int(num(r["n_seeds"])) for r in table if r.get("n_seeds"))
    ax.set_title(
        f"E1: success against cost, {n} seeds, 95% intervals  [" + SIMULATED + "]", loc="left", fontsize=8.5
    )
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out, "e1_success_vs_cost.png")


# -- E2 -----------------------------------------------------------------------------------------------


def e2(results: Path, out: Path) -> None:
    rows = read_csv(results / "e2" / "brownout_windows.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.axvspan(150, 270, color="#f0efec", zorder=0)
    ax.text(
        210,
        0.97,
        "brownout: sim-a returns 5xx",
        ha="center",
        va="top",
        fontsize=8,
        color=INK2,
        transform=ax.get_xaxis_transform(),
    )
    for cfg, color, marker, ls in [
        ("brownout/no_retry", ORANGE, "s", "--"),
        ("brownout/retry", BLUE, "o", "-"),
        ("brownout/retry_fallback", AQUA, "^", "-"),
    ]:
        pts = sorted(
            (num(r["window_start_s"]), num(r["success"]), num(r["success_ci"]))
            for r in rows
            if r["config"] == cfg
        )
        if not pts:
            continue
        xs = [p[0] + 15 for p in pts]
        ax.plot(
            xs,
            [p[1] for p in pts],
            color=color,
            marker=marker,
            ms=6,
            mec=SURFACE,
            mew=1.5,
            ls=ls,
            label=cfg.split("/")[1],
        )
        ax.fill_between(
            xs, [p[1] - p[2] for p in pts], [p[1] + p[2] for p in pts], color=color, alpha=0.12, lw=0
        )
    ax.set_xlabel("arrival time of the task (s; 30 s windows)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    n = max(int(num(r["n_seeds"])) for r in rows)
    ax.set_title(
        f"E2 brownout: success per arrival window, {n} seeds, 95% bands  [" + SIMULATED + "]",
        loc="left",
        fontsize=8.5,
    )
    ax.legend(loc="lower left", fontsize=8)
    save(fig, out, "e2_brownout_windows.png")


# -- E3 -----------------------------------------------------------------------------------------------


def e3(results: Path, out: Path) -> None:
    rows = {r["config"]: r for r in read_csv(results / "e3" / "aggregate.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, fail in zip(axes, ("f00", "f10"), strict=True):
        ref = rows.get(f"{fail}/react")
        if ref:
            ax.axhline(num(ref["latency_p50_ms"]) / 1000, color=INK2, lw=1.2, ls="--")
            ax.text(
                8,
                num(ref["latency_p50_ms"]) / 1000,
                "react ",
                ha="right",
                va="bottom",
                fontsize=8,
                color=INK2,
            )
        for mode, color, marker in (
            ("react_parallel", BLUE, "o"),
            ("plan_execute", ORANGE, "s"),
            ("plan_replan", AQUA, "^"),
        ):
            ks, ys, es = [], [], []
            for k in (1, 2, 4, 8):
                r = rows.get(f"{fail}/{mode}/k{k}")
                if r:
                    ks.append(k)
                    ys.append(num(r["latency_p50_ms"]) / 1000)
                    es.append(num(r["latency_p50_ms_ci"]) / 1000)
            if ks:
                ax.errorbar(
                    ks,
                    ys,
                    yerr=es,
                    color=color,
                    marker=marker,
                    ms=7,
                    mec=SURFACE,
                    mew=1.5,
                    capsize=3,
                    label=mode,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("concurrency limit k")
        ax.set_title(f"tool transient failure rate {0 if fail == 'f00' else 10}%", fontsize=9)
    axes[0].set_ylabel("makespan P50 (s, virtual)")
    axes[0].legend(fontsize=8, loc="center right")
    n = max(int(num(r["n_seeds"])) for r in rows.values())
    fig.suptitle(
        f"E3: makespan against k, wide tasks, {n} seeds, 95% intervals  [" + SIMULATED + "]",
        x=0.01,
        ha="left",
        fontsize=8.5,
    )
    save(fig, out, "e3_makespan_vs_k.png")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    framework(out)
    timeline(out)
    for name, fn in (("e1", e1), ("e2", e2), ("e3", e3)):
        try:
            fn(results, out)
        except FileNotFoundError as e:
            print(f"skipped {name}: {e.filename} missing", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
