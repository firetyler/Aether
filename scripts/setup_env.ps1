Param()

Write-Host "Creating virtual environment in .venv..."
python -m venv .venv

Write-Host "Activating virtual environment and installing requirements..."
. .\.venv\Scripts\Activate.ps1
if (Test-Path "ai_Model\requirements.txt") {
    pip install --upgrade pip
    pip install -r ai_Model\requirements.txt
} else {
    Write-Host "ai_Model/requirements.txt not found. Please check path."
}

Write-Host "Install package in editable mode..."
pip install -e .

Write-Host "Setup complete. To activate the venv run: .\\.venv\\Scripts\\Activate.ps1"
