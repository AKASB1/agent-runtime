"""``python -m agent_runtime run`` and ``replay``: one run through the service's environments (on a
system clock scaled by 50, as in the demo), and the replay of a stored trace."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any

from agent_runtime.api.service import Service, Settings
from agent_runtime.runtime.runner import RunSpec, replay_spans
from agent_runtime.telemetry.trace import load_jsonl, validate_trace


def _service(tmp: str, root: str) -> Service:
    return Service(
        Settings(
            db=os.path.join(tmp, "cli.db"),
            workspace_root=os.path.join(tmp, "ws"),
            time_scale=50.0,
            examples_dir=os.path.join(root, "examples"),
        )
    )


def run_one(args: Any) -> int:
    root = os.environ.get("AGENT_RUNTIME_ROOT", os.getcwd())
    body = {"workflow": args.workflow, "input": json.loads(args.input), "config": {"seed": args.seed}}
    if args.workflow not in ("optimize", "agent", "chat"):
        body["config"]["router"] = {"type": "fixed", "target": args.target}

    async def go() -> tuple[dict, str]:
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, root)
            try:
                rid = await svc.start_run(None, body)
                await svc.wait(rid)
                return svc.get_run(rid), svc.trace(rid)
            finally:
                await svc.aclose()

    rec, trace = asyncio.run(go())
    validate_trace(load_jsonl(trace))
    print(
        json.dumps(
            {k: rec.get(k) for k in ("id", "status", "failure_reason", "result", "cost_ucu", "latency_s")},
            indent=2,
        )
    )
    if args.trace_out:
        with open(args.trace_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(trace)
        print(f"trace written to {args.trace_out}")
    return 0 if rec.get("status") in ("succeeded", "failed") else 1


def replay_file(path: str) -> int:
    root = os.environ.get("AGENT_RUNTIME_ROOT", os.getcwd())
    with open(path, encoding="utf-8") as f:
        spans = load_jsonl(f.read())
    with tempfile.TemporaryDirectory() as tmp:
        svc = _service(tmp, root)
        try:
            out = replay_spans(
                spans, lambda spec: svc.build_env(RunSpec.model_validate(spec.model_dump()), live=False)
            )
        finally:
            svc.store.close()
    print(json.dumps(out, indent=2))
    return 0 if out["divergence"] is None else 1
