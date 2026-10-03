"""The examples run as documented."""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    env = {
        k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "WINDIR")
    }
    env.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"})
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=120, env=env
    )


def test_local_run_example():
    out = run("examples/local_run.py")
    assert out.returncode == 0 and out.stdout.strip() == "hello"


def test_scripted_agent_example_in_a_temporary_workspace():
    out = run("examples/agent/run_agent.py")
    assert out.returncode == 0, out.stderr
    assert "deny  rule=path_outside_workspace" in out.stdout
    assert "subagent   lookup depth=1" in out.stdout
    assert "escape.txt exists outside the workspace: False" in out.stdout
    assert "replay: divergence=None provider_calls=0 tool_calls=0" in out.stdout


def test_report_writes_the_tables_after_the_plot_script_exits(tmp_path):
    from agent_runtime.evals.report import main

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "plot_results.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    assert main(tmp_path) == 0
    assert (tmp_path / "results" / "tables.md").read_text(encoding="utf-8").startswith("# Result tables")


def test_a_manifest_records_the_commit_at_the_start_of_the_run(tmp_path):
    from agent_runtime.evals.commands import Ctx, manifest

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    (tmp_path / "out.csv").write_text("a\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "out.csv"], ["commit", "-q", "-m", "x"]):
        subprocess.run([*git, *args], cwd=tmp_path, check=True, capture_output=True)
    Ctx(tmp_path, 1, "test")
    (tmp_path / "out.csv").write_text("b\n", encoding="utf-8")  # the run rewrites a tracked result
    m = manifest(tmp_path, "x", seeds=[1], config={}, workers=1, wall=1.0, load_note="test", tasks=1)
    assert len(m["commit"]) == 40 and not m["commit"].endswith("-dirty")
