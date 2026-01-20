# PowerShell startup script to run both frontend and backend with live logs
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Face-Body Swap Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "src\api\main.py")) {
    Write-Host "Error: Please run this script from the face-body-swap directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set environment variables for logging
$env:LOG_LEVEL = "DEBUG"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[1/2] Starting Backend Server..." -ForegroundColor Yellow
Write-Host "Backend will run on http://localhost:8000" -ForegroundColor Green
Write-Host ""

# Start backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:LOG_LEVEL='DEBUG'; `$env:PYTHONUNBUFFERED='1'; python -m src.api.main" -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep -Seconds 3

Write-Host "[2/2] Starting Frontend Development Server..." -ForegroundColor Yellow
Write-Host "Frontend will run on http://localhost:5173" -ForegroundColor Green
Write-Host ""

# Check if node_modules exists
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location frontend
    npm install
    Set-Location ..
}

# Start frontend in new window
Set-Location frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm run dev" -WindowStyle Normal
Set-Location ..

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Both servers are starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "All logs will appear in the PowerShell windows." -ForegroundColor Cyan
Write-Host "Close the windows to stop the servers." -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit this script (servers will continue running)"

