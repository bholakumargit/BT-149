ST@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
)

echo Installing dashboard and inference packages...
".venv\Scripts\python.exe" -m pip install -r requirements-app.txt
if errorlevel 1 goto install_failed

echo.
echo Starting Malware AI on http://127.0.0.1:8001
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" -m uvicorn malware_classifier.app:app --app-dir src --port 8001 --reload
exit /b %errorlevel%

:install_failed
echo Package installation failed. Check the error above and try again.
pause
exit /b 1
