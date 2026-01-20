@echo off
REM Startup script to run both frontend and backend with live logs
echo ========================================
echo Starting Face-Body Swap Application
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "src\api\main.py" (
    echo Error: Please run this script from the face-body-swap directory
    pause
    exit /b 1
)

REM Set environment variables for logging
set LOG_LEVEL=DEBUG
set PYTHONUNBUFFERED=1

echo [1/2] Starting Backend Server...
echo Backend will run on http://localhost:8000
echo.
start "Backend Server" cmd /k "python -m src.api.main"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend Development Server...
echo Frontend will run on http://localhost:5173
echo.
cd frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)
start "Frontend Server" cmd /k "npm run dev"
cd ..

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo All logs will appear in the terminal windows.
echo Close the windows to stop the servers.
echo.
pause

