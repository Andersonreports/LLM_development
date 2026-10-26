@echo off
rem Double-click to start the WES AI Analyser and open it in the browser.
rem Close this window (or press Ctrl+C) to stop it.
setlocal
title WES AI Analyser
cd /d "%~dp0"

echo.
echo    WES AI Analyser
echo    ===============
echo.

if not exist ".venv\Scripts\uvicorn.exe" (
  echo    [X] Python environment not set up yet. Run this once:
  echo.
  echo        python -m venv .venv
  echo        .venv\Scripts\pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

rem Already running? Just open the browser instead of starting a second copy.
powershell -NoProfile -Command "try{$null=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ -TimeoutSec 2;exit 0}catch{exit 1}" >nul 2>&1
if not errorlevel 1 (
  echo    Already running - opening http://localhost:8000
  start "" http://localhost:8000
  timeout /t 2 >nul
  exit /b 0
)

rem Use the full knowledge base when it has been downloaded, otherwise the demo one.
if exist "resources\hpo\hp.obo" (
  echo    Knowledge base : full
) else (
  if exist "demo\resources\hpo\hp.obo" (
    set "WES_RESOURCES_DIR=%CD%\demo\resources"
    echo    Knowledge base : DEMO only ^(6 genes^) - upload demo\demo_proband.vcf
  ) else (
    echo    Knowledge base : MISSING - analysis will have no phenotype matching
    echo                     fix: .venv\Scripts\python scripts\make_demo_data.py
  )
)

rem The interpretation model runs locally; without it cases still finish with a rules-only draft.
powershell -NoProfile -Command "try{$null=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/tags -TimeoutSec 2;exit 0}catch{exit 1}" >nul 2>&1
if errorlevel 1 (
  echo    AI model       : NOT running - start Ollama for AI drafts
) else (
  echo    AI model       : ready
)

echo.
echo    Opening http://localhost:8000
echo    Keep this window open while you use the tool.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 4; Start-Process 'http://localhost:8000'"
.venv\Scripts\uvicorn.exe wes.web.app:app --host 127.0.0.1 --port 8000
