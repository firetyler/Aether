Param(
    [string]$Args = ""
)

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Virtualenv not found. Run scripts\setup_env.ps1 first."
    exit 1
}

. .\.venv\Scripts\Activate.ps1
python -m ai_Model.aether2 $Args
