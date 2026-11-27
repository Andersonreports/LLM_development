@echo off
REM ============================================================
REM  ADVAT LLM - chat tool launcher
REM  Double-click to start the chat and open it in your browser.
REM  Close the black window to stop it.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment not found at .venv\Scripts\python.exe
    echo Set up the project first, then run this again.
    pause
    exit /b 1
)

echo Starting the ADVAT LLM chat server...
start "ADVAT LLM chat server" ".venv\Scripts\python.exe" run_webui.py

echo Waiting for the server to come up...
timeout /t 4 /nobreak >nul

echo Opening the chat in your browser...
start "" http://127.0.0.1:8080

echo.
echo ============================================================
echo  Chat running at  http://127.0.0.1:8080
echo  A separate black window runs the server - keep it open.
echo  Close that window to stop the chat.
echo ============================================================
timeout /t 6 /nobreak >nul
