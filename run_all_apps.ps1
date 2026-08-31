# =============================================================================
# run_all_apps.ps1
# ----------------
# Launches all three Alzheimer's FL applications in parallel:
# 1. Streamlit Dashboard (http://localhost:8501)
# 2. API Gateway (http://localhost:8000)
# 3. Federated Learning Simulation
#
# Usage:
#   .\run_all_apps.ps1
#
# =============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ALZHEIMER'S FL - Application Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the project root directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Set Python path
$env:PYTHONPATH = "."

Write-Host "[1/3] Starting Streamlit Dashboard..." -ForegroundColor Green
Write-Host "       → http://localhost:8501" -ForegroundColor Gray
Start-Process -FilePath "python" -ArgumentList "-m streamlit run streamlit_app/app.py" -NoNewWindow

Write-Host "[2/3] Starting API Gateway..." -ForegroundColor Green
Write-Host "       → http://localhost:8000" -ForegroundColor Gray
Start-Process -FilePath "python" -ArgumentList "api_gateway/main.py" -NoNewWindow

Write-Host "[3/3] Starting Federated Learning Simulation..." -ForegroundColor Green
Write-Host "       → Training with 3 clients, 5 rounds" -ForegroundColor Gray
Start-Process -FilePath "python" -ArgumentList "federated_core/run_simulation_standalone.py" -NoNewWindow

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All applications launched!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  📊 Streamlit Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "  🔌 API Gateway:         http://localhost:8000" -ForegroundColor Cyan
Write-Host "  🤖 FL Simulation:       Training in progress" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C in each terminal to stop the applications." -ForegroundColor Gray
