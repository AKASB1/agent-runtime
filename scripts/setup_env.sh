#!/usr/bin/env bash
# Create .venv in the repository and install the package with all extras (Linux, macOS, Git Bash).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
if ! "$PY" --version >/dev/null 2>&1; then PY=python3; fi
"$PY" -m venv .venv
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi
"$VPY" -m pip install --disable-pip-version-check -q -e ".[api,mcp,eval,dev]"
"$VPY" -c "import agent_runtime, mcp, fastapi, scipy; print('agent_runtime', agent_runtime.__version__, 'ready in .venv')"
echo "activate with: source .venv/bin/activate   (Git Bash on Windows: source .venv/Scripts/activate)"
