## What this change does

## Checklist

- [ ] `ruff check src tests scripts examples` and `ruff format --check src tests scripts examples` are clean
- [ ] `python -m pytest -q` passes, including the OpenAPI drift test
- [ ] `python -m agent_runtime eval --quick --workers 2` runs
- [ ] Tests added or updated for the changed behavior
- [ ] If the change moves a result: the affected experiments rerun, result files and manifests committed
- [ ] Documentation updated; every new number names its result file
