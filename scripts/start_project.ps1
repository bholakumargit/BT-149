Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating local Python environment..."
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.12 was not found; falling back to the default python command."
        python -m venv .venv
    }
}

Write-Host "Installing dashboard and inference packages..."
& ".venv\Scripts\python.exe" -m pip install -r requirements-app.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard package installation failed. Check the pip error above."
}

Write-Host "Starting Malware AI on http://127.0.0.1:8001"
Write-Host "Press Ctrl+C to stop the server."
& ".venv\Scripts\python.exe" -m uvicorn malware_classifier.app:app --app-dir src --port 8001 --reload
