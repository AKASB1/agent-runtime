"""Write the small example traces under examples/traces/ (deterministic simulated runs).

    python scripts/make_example_traces.py

``plan_execute_wide.jsonl``: one ``plan_execute`` run of a wide task (k = 4) on the simulated
``medium`` model; the timeline figure is drawn from it. ``react_chain.jsonl``: one ``react`` run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_runtime.evals.simrun import Job, run_job  # noqa: E402
from agent_runtime.telemetry.trace import load_jsonl, validate_trace  # noqa: E402


def pick(out: dict, min_tools: int) -> str:
    for rid in sorted(out["traces"]):
        spans = load_jsonl(out["traces"][rid])
        tools = [s for s in spans if s["kind"] == "tool"]
        if len(tools) >= min_tools and spans[0]["status"] == "ok":
            return out["traces"][rid]
    raise SystemExit("no suitable run")


def main() -> int:
    dest = ROOT / "examples" / "traces"
    dest.mkdir(parents=True, exist_ok=True)
    wide = run_job(
        Job(
            "example",
            "plan_execute",
            1,
            workflow="plan_execute",
            n_tasks=20,
            shape="wide",
            router={"type": "fixed", "target": "sim-a/medium"},
            run_config={"concurrency": 4},
            keep_traces=True,
        )
    )
    chain = run_job(
        Job(
            "example",
            "react",
            1,
            workflow="react",
            n_tasks=20,
            router={"type": "fixed", "target": "sim-a/medium"},
            keep_traces=True,
        )
    )
    for name, out, n in (("plan_execute_wide.jsonl", wide, 5), ("react_chain.jsonl", chain, 4)):
        text = pick(out, n)
        validate_trace(load_jsonl(text))
        (dest / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote examples/traces/{name} ({len(text.splitlines())} spans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
