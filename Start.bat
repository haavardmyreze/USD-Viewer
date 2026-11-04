@echo off
setlocal EnableExtensions
title USD Viewer

rem ---------------------------------------------------------------------------
rem  USD Viewer launcher for Windows.
rem  Double-click to start. The first run creates a Python environment in .venv
rem  and installs the dependencies (takes a minute); later runs start at once.
rem  Optional: Start.bat 8080   (use another port; default 8000)
rem ---------------------------------------------------------------------------

cd /d "%~dp0"

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

set "VENV_PY=.venv\Scripts\python.exe"
set "STAMP=.venv\requirements.installed"

if exist "%VENV_PY%" goto :check_deps

rem --- find Python 3.10+ ------------------------------------------------------
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if defined PY goto :make_venv
where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
if defined PY goto :make_venv

echo.
echo  Python 3.10 or newer was not found.
echo  Install it from https://www.python.org/downloads/
echo  and tick "Add python.exe to PATH" during setup, then run Start.bat again.
echo.
pause
exit /b 1

:make_venv
echo Creating Python environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto :fail

:check_deps
rem --- install / update dependencies when requirements.txt changes ----------
if exist "%STAMP%" (
    fc /b requirements.txt "%STAMP%" >nul 2>nul && goto :run
)
echo Installing dependencies (first run only, this can take a minute) ...
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
copy /y requirements.txt "%STAMP%" >nul

:run
rem --- open the browser once the server answers -----------------------------
start "" /min powershell -NoProfile -WindowStyle Hidden -Command ^
  "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/api/health' | Out-Null; Start-Process 'http://localhost:%PORT%'; break } catch { Start-Sleep -Milliseconds 500 } }"

echo.
echo  USD Viewer is running at http://localhost:%PORT%
echo  Close this window (or press Ctrl+C) to stop it.
echo.
"%VENV_PY%" -m uvicorn server.app:app --host 127.0.0.1 --port %PORT%
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo  Something went wrong - see the messages above.
echo  If port %PORT% is busy, run:  Start.bat 8080
echo.
pause
exit /b 1
