# Contributing

This is a portfolio repository with one maintainer. Issues and pull requests are welcome; this page lists what a change needs before it is merged.

## Environment

- Python 3.12 (`pyproject.toml`). `scripts/setup_env.ps1` (PowerShell) or `bash scripts/setup_env.sh` creates `.venv` and installs the package with all extras: `api` (FastAPI, uvicorn), `mcp` (the `mcp` SDK, pinned), `eval` (NumPy, SciPy, matplotlib), and `dev` (pytest, ruff, pytest-repeat).
- No hosted model is needed: the tests and the evaluation use scripted and simulated models and a local stand-in server.

## Checks

Run these before opening a pull request. CI runs the same steps (`.github/workflows/ci.yml`):

```bash
ruff check src tests scripts examples
ruff format --check src tests scripts examples
python -m pytest -q
python -m agent_runtime eval --quick --workers 2
```

The API's OpenAPI document must match `docs/openapi.json`; `python -m pytest -q tests/test_api.py -k openapi` checks it.

## Results

The results in the README and in [docs/results.md](docs/results.md) come from the committed files in `results/`, each experiment with its manifest; `results/tables.md` is generated from the CSV files. Selections are tuned on the tuning seeds and frozen in `configs/` before an evaluation ([docs/evaluation.md](docs/evaluation.md)). A change that moves a result reruns the affected experiments once on the committed change and commits their result files and manifests. The README's highlights and tables are written by hand from the results.

## Figures and diagrams

`python scripts/plot_results.py` draws the figures in `docs/figures/` from the committed results, including the two README versions. The architecture diagram is drawn from `docs/diagrams/architecture.d2` with [D2](https://d2lang.com) (`bash scripts/render_diagrams.sh` or `powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1`); the committed SVG was drawn with D2 0.9.0. `python scripts/make_assets.py` writes the icon. CI needs none of these.

## Style

- Every model input is derived from the session log; nothing else feeds a model call ([docs/contracts.md](docs/contracts.md)).
- A router or planner sees only what the runtime knew at that point, never a task's true tier or an outcome that has not happened.
- Every tool call passes the permission layer; new tools declare their effects in the typed registry.
- Documentation states what was measured, under which assumptions, and names the result file.
