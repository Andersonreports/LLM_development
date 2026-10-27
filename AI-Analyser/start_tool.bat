@echo off
rem Double-click to open the WES AI Analyser.
rem
rem The tool runs inside WSL (Ubuntu), because bcftools and Ensembl VEP are Linux-only.
rem systemd keeps it running there, so this script only has to wake WSL and open the
rem browser - closing this window does NOT stop the tool.
setlocal
title WES AI Analyser

echo.
echo    WES AI Analyser
echo    ===============
echo.

wsl -d Ubuntu -- true >nul 2>&1
if errorlevel 1 (
  echo    [X] Cannot reach the Ubuntu WSL distribution.
  echo        Try:  wsl --list --verbose
  echo.
  pause
  exit /b 1
)

rem WSL shuts the whole Linux environment down once no Windows session is attached,
rem which stops the service and would kill a running VEP annotation. vmIdleTimeout in
rem .wslconfig is ignored from WSL 2.7, so hold a session open instead: one hidden
rem sleeping process is enough to keep the distro alive.
wsl -d Ubuntu -- pgrep -f "sleep 99999999" >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -WindowStyle Hidden -Command ^
    "Start-Process -WindowStyle Hidden wsl -ArgumentList '-d','Ubuntu','--','sleep','99999999'"
)

echo    Starting the analysis service...
wsl -d Ubuntu -- systemctl start wes ollama >nul 2>&1

rem The first request loads the knowledge base (about 18,000 genes and 20,000 HPO
rem terms), which takes up to a minute on a cold start - be patient here.
echo    Loading the knowledge base, this can take a minute...
set READY=
for /l %%i in (1,1,40) do (
  if not defined READY (
    wsl -d Ubuntu -- curl -s -m 10 -o /dev/null http://127.0.0.1:8000/system >nul 2>&1 && set READY=1
  )
)

if not defined READY (
  echo    [X] The service did not start. Recent log:
  wsl -d Ubuntu -- systemctl status wes --no-pager -n 15
  echo.
  pause
  exit /b 1
)

for /f "delims=" %%m in ('wsl -d Ubuntu -- bash -lc "systemctl is-active ollama 2>/dev/null"') do set OLLAMA=%%m
echo    Analysis service : running
if "%OLLAMA%"=="active" (echo    AI model         : running) else (echo    AI model         : NOT running - cases get a rules-only draft)
echo.
echo    Opening http://localhost:8000
echo    The tool keeps running after you close this window.
echo    To stop it:  wsl -d Ubuntu -- systemctl stop wes
echo.

start "" http://localhost:8000
rem brief pause so the browser launches before this window closes
rem (ping, not timeout: timeout fails when the script is run non-interactively)
ping -n 4 127.0.0.1 >nul
