@echo off
REM ============================================================
REM  LesionIQ Research Pipeline — Windows bootstrap
REM
REM  First run: creates .venv, installs requirements.
REM  Subsequent runs: reuses .venv and forwards all args to run.py.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "PYEXE=python"

if not exist "%VENV%\Scripts\python.exe" (
    echo [LesionIQ] Creating virtual environment at %VENV% ...
    %PYEXE% -m venv "%VENV%"
    if errorlevel 1 (
        echo [LesionIQ] Failed to create venv. Is Python 3.10+ on PATH?
        exit /b 1
    )
    echo [LesionIQ] Installing requirements... ^(this is one-time, ~5 minutes^)
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [LesionIQ] pip install failed. Inspect the error above.
        exit /b 1
    )
)

REM No args -> print usage
if "%~1"=="" (
    echo Usage:
    echo   lesioniq.bat verify   --data-root PATH
    echo   lesioniq.bat preprocess --data-root PATH --out-root PATH
    echo   lesioniq.bat split    --pre-root PATH --raw-root PATH --out PATH --datasets isic2019 ...
    echo   lesioniq.bat train    --variant V0 --split-dir PATH --out-dir PATH
    echo   lesioniq.bat evaluate --variant V0 --checkpoint best.pt --split-dir PATH --out-dir PATH
    echo   lesioniq.bat full     --config pipeline.yaml
    echo.
    echo See README.md for full options.
    exit /b 0
)

"%VENV%\Scripts\python.exe" run.py %*
exit /b %errorlevel%
