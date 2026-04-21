@echo off
REM =============================================================================
REM  EarlyMind — Windows Launcher
REM  Starts FastAPI (port 8000) and Streamlit (port 8501) concurrently.
REM  Run from the project root: start_windows.bat
REM =============================================================================

setlocal

echo.
echo ======================================================
echo   EarlyMind -- Starting Services
echo ======================================================
echo.

REM Activate conda environment
call conda activate infant_id
if errorlevel 1 (
    echo [ERROR] Could not activate 'infant_id' conda env.
    echo         Run setup.bat first.
    pause
    exit /b 1
)

REM ── Check ports are free ──────────────────────────────────────────────────────
netstat -an | findstr ":8000 " | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8000 is already in use. FastAPI may not start.
)
netstat -an | findstr ":8501 " | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8501 is already in use. Streamlit may not start.
)

REM ── Launch FastAPI (Uvicorn) in a separate window ─────────────────────────────
echo [EarlyMind] Starting FastAPI on http://localhost:8000 ...
start "EarlyMind - FastAPI" cmd /k "call conda activate infant_id && uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info"

REM ── Wait 4 seconds for FastAPI to be ready ───────────────────────────────────
timeout /t 4 /nobreak >nul

REM ── Launch Streamlit in a separate window ─────────────────────────────────────
echo [EarlyMind] Starting Streamlit on http://localhost:8501 ...
start "EarlyMind - Streamlit" cmd /k "call conda activate infant_id && streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"

echo.
echo ======================================================
echo   Both services are starting in separate windows.
echo.
echo   FastAPI  (API + Swagger): http://localhost:8000/docs
echo   Streamlit (UI Dashboard): http://localhost:8501
echo.
echo   Close those windows to stop the services.
echo ======================================================
echo.

REM ── Open browser ─────────────────────────────────────────────────────────────
timeout /t 6 /nobreak >nul
start http://localhost:8501

pause
