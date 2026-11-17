Set-Location $PSScriptRoot\..
uvicorn malware_classifier.app:app --app-dir src --port 8001 --reload
