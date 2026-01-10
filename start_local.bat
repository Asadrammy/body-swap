@echo off
REM Face-Body Swap - Local Startup Script for Windows
REM This script sets up and starts the local server

echo ============================================================
echo Face-Body Swap - Local Server Startup
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [2/4] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo [2/4] No virtual environment found (skipping)
    echo To create one: python -m venv venv
)
echo.

REM Run setup verification
echo [3/4] Running setup verification...
python setup_local.py
if errorlevel 1 (
    echo.
    echo WARNING: Setup verification found issues
    echo The server may still start, but some features may not work
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo Setup cancelled
        pause
        exit /b 1
    )
)
echo.

REM Start the server
echo [4/4] Starting server...
echo.
echo ============================================================
echo Server starting...
echo Open your browser to: http://localhost:8000
echo API docs available at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python app.py

pause












