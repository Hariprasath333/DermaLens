# ============================================================
# DermaLens — Quick Start (Backend + Optional Tunnel)
# ============================================================
#
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
# ============================================================

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

Write-Host ""
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  DermaLens — Starting Backend Server"                     -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Find venv (.venv or venv)
$venvActivate = $null
if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1")) {
    $venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
} elseif (Test-Path (Join-Path $ProjectRoot "venv\Scripts\Activate.ps1")) {
    $venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
}

if ($venvActivate) {
    Write-Host "[1/3] Activating virtual environment: $venvActivate" -ForegroundColor Yellow
    & $venvActivate
} else {
    Write-Host "[1/3] No venv found — using active Python environment" -ForegroundColor Yellow
}

# Start uvicorn
Write-Host "[2/3] Starting FastAPI backend on http://localhost:8000..." -ForegroundColor Yellow
$startCmd = if ($venvActivate) { ". `"$venvActivate`"; python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000" } else { "python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000" }
$backendJob = Start-Process -PassThru powershell -ArgumentList "-Command", "`"$startCmd`""

Start-Sleep -Seconds 2

# Check ngrok
$ngrokInstalled = (Get-Command ngrok -ErrorAction SilentlyContinue)
if ($ngrokInstalled) {
    Write-Host "[3/3] ngrok detected. Starting tunnel on port 8000..." -ForegroundColor Green
    Write-Host "  Press Ctrl+C to stop both backend and tunnel." -ForegroundColor Gray
    try {
        ngrok http 8000
    } finally {
        if ($backendJob -and !$backendJob.HasExited) {
            Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  Backend stopped." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[3/3] ngrok not installed or not in PATH." -ForegroundColor Yellow
    Write-Host "  Backend is running locally at http://localhost:8000" -ForegroundColor Green
    Write-Host "  To run the frontend: cd frontend && npm run dev" -ForegroundColor Cyan
    Write-Host "  Press Enter in this window to stop the backend..." -ForegroundColor Gray
    [void][System.Console]::ReadLine()
    if ($backendJob -and !$backendJob.HasExited) {
        Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  Backend stopped." -ForegroundColor Yellow
    }
}
