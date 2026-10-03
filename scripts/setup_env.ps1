# Create .venv in the repository and install the package with all extras (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $py -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -e ".[api,mcp,eval,dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv\Scripts\python.exe -c "import agent_runtime; print('agent_runtime', agent_runtime.__version__, 'ready in .venv')"
Write-Output "activate with: .venv\Scripts\Activate.ps1  (or call .venv\Scripts\python.exe directly)"
