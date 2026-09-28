@echo off
TITLE graphforANN Enterprise AML Platform Launcher
COLOR 0A

echo ===============================================================================
echo   graphforANN: Heterogeneous GNN Platform for Enterprise AML Surveillance
echo ===============================================================================
echo [1/3] Checking Python Environment...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

echo [2/3] Starting FastAPI Secure Backend on http://127.0.0.1:8000 ...
start "graphforANN Backend (Port 8000)" cmd /k "python backend/api.py"

timeout /t 3 /nobreak >nul

echo [3/3] Starting Streamlit Research Portal on http://localhost:8501 ...
start "graphforANN Frontend (Port 8501)" cmd /k "streamlit run frontend/app.py --server.headless false"

echo ===============================================================================
echo   Platform successfully initialized!
echo   - Backend API: http://127.0.0.1:8000/docs
echo   - Research UI: http://localhost:8501
echo ===============================================================================
pause
