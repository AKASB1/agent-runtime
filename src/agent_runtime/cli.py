"""Command line: ``python -m agent_runtime <command>``.

Commands: ``serve`` (the HTTP API), ``run`` (one workflow run), ``eval`` (simulated experiments),
``replay`` (re-run a trace without calling a provider or a tool), ``gen`` (generate simulated tasks),
``report`` (figures and tables from committed results), ``openapi`` (write docs/openapi.json),
``demo`` (scripts/demo.py)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _root() -> str:
    return os.environ.get("AGENT_RUNTIME_ROOT", os.getcwd())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agent_runtime", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="run simulated experiments")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--full", action="store_true")
    e.add_argument("--no-sensitivity", action="store_true")
    e.add_argument(
        "--experiment",
        choices=[
            "e1-tune",
            "informativeness",
            "e1",
            "e1-sensitivity",
            "e2",
            "e3",
            "e4",
            "e4b",
            "e5-tune",
            "e5",
            "e6",
        ],
    )
    e.add_argument("--live", action="store_true")
    e.add_argument("--workers", type=int, default=int(os.environ.get("AGENT_RUNTIME_WORKERS", "4")))
    e.add_argument("--load-note", default="not recorded")
    e.add_argument("--root", default=_root())

    g = sub.add_parser("gen", help="print simulated tasks as JSON lines")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("-n", type=int, default=5)
    g.add_argument("--mix", default="mixed")
    g.add_argument("--shape", default="chain")

    r = sub.add_parser("replay", help="replay a stored trace (JSON Lines) of a simulated or demo run")
    r.add_argument("trace")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=18500)
    s.add_argument("--db", default="tmp/agent-runtime.db")
    s.add_argument("--workspace-root", default="tmp/workspaces")
    s.add_argument("--time-scale", type=float, default=1.0)
    s.add_argument("--max-sandbox", default="workspace_write", choices=["read_only", "workspace_write"])
    s.add_argument("--max-approval", default="never", choices=["ask", "auto", "never"])
    s.add_argument(
        "--allow-exec",
        action="append",
        default=[],
        help='an argument-vector prefix as JSON, e.g. \'["python", "-c"]\'',
    )
    s.add_argument("--allow-hosts", action="append", default=[])
    s.add_argument(
        "--openai-base-url",
        default=None,
        help="an OpenAI-compatible endpoint exposed as target 'openai/<model>'",
    )
    s.add_argument("--openai-models", default="echo")
    s.add_argument("--openai-dialect", default="openai")
    s.add_argument("--examples-dir", default="examples")
    s.add_argument(
        "--port-file", default=None, help="write the bound port to this file (useful with --port 0)"
    )

    run = sub.add_parser("run", help="run one workflow against the simulated models and print the result")
    run.add_argument("workflow")
    run.add_argument("--input", default="{}", help='JSON input, e.g. \'{"task": "..."}\'')
    run.add_argument("--target", default="sim-a/medium")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--trace-out", default=None)

    rep = sub.add_parser("report", help="write figures and tables from committed results")
    rep.add_argument("--root", default=_root())

    o = sub.add_parser("openapi", help="write docs/openapi.json")
    o.add_argument("--out", default="docs/openapi.json")

    d = sub.add_parser("demo", help="run scripts/demo.py")
    d.add_argument("rest", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)
    if args.cmd == "eval":
        from agent_runtime.evals.commands import main as eval_main

        return eval_main(args)
    if args.cmd == "gen":
        from agent_runtime.sim.tasks import gen

        for t in gen(args.seed, args.n, args.mix, args.shape):
            print(
                json.dumps(
                    {
                        "id": t.id,
                        "tier": t.tier,
                        "hint": t.hint,
                        "text": t.text,
                        "answer": t.answer,
                        "steps": [
                            {"id": s.id, "tool": s.tool, "args": s.args, "depends_on": list(s.depends_on)}
                            for s in t.steps
                        ],
                    }
                )
            )
        return 0
    if args.cmd == "replay":
        from agent_runtime.tools_cli import replay_file

        return replay_file(args.trace)
    if args.cmd == "run":
        from agent_runtime.tools_cli import run_one

        return run_one(args)
    if args.cmd == "serve":
        from agent_runtime.api.service import serve

        return serve(args)
    if args.cmd == "report":
        from agent_runtime.evals.report import main as report_main

        return report_main(Path(args.root))
    if args.cmd == "openapi":
        from agent_runtime.api.app import write_openapi

        write_openapi(args.out)
        return 0
    if args.cmd == "demo":
        import runpy

        sys.argv = ["scripts/demo.py", *args.rest]
        runpy.run_path(str(Path(_root()) / "scripts" / "demo.py"), run_name="__main__")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
