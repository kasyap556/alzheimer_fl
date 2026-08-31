@echo off
REM =============================================================================
REM run_all_apps.bat
REM ----------------
REM Batch file to launch all Alzheimer's FL applications in parallel
REM
REM Usage:
REM   run_all_apps.bat
REM
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   ALZHEIMER'S FL - Application Launcher
echo ========================================
echo.

REM Get the project root directory
cd /d "%~dp0"

REM Set Python path
set PYTHONPATH=.

echo [1/3] Starting Streamlit Dashboard...
echo        - http://localhost:8501
start "Streamlit Dashboard" cmd /k "python -m streamlit run streamlit_app/app.py"

echo [2/3] Starting API Gateway...
echo        - http://localhost:8000
start "API Gateway" cmd /k "python api_gateway/main.py"

echo [3/3] Starting Federated Learning Simulation...
echo        - Training with 3 clients, 5 rounds
start "FL Simulation" cmd /k "python federated_core/run_simulation_standalone.py"

echo.
echo ========================================
echo   All applications launched!
echo ========================================
echo.
echo Services:
echo   Streamlit Dashboard: http://localhost:8501
echo   API Gateway:         http://localhost:8000
echo   FL Simulation:       Training in progress
echo.
echo Press Ctrl+C in each terminal to stop.
echo.
pause
