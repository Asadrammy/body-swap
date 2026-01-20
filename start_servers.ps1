# PowerShell script to start both frontend and backend servers

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Face-Body Swap - Frontend & Backend" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$projectDir = "D:\projects\image\face-body-swap"
Set-Location $projectDir

# Check if frontend node_modules exists
$frontendNodeModules = Join-Path $projectDir "frontend\node_modules"
if (-not (Test-Path $frontendNodeModules)) {
    Write-Host "[WARN] Frontend node_modules not found. Installing..." -ForegroundColor Yellow
    Set-Location "$projectDir\frontend"
    npm install
    Set-Location $projectDir
}

Write-Host "[INFO] Starting Backend Server..." -ForegroundColor Green
Write-Host "  Port: 8000" -ForegroundColor Gray
Write-Host "  API: http://localhost:8000/api/v1" -ForegroundColor Gray
Write-Host "  Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""

# Start backend in background
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:projectDir
    python app.py
}

Start-Sleep -Seconds 2

Write-Host "[INFO] Starting Frontend Dev Server..." -ForegroundColor Green
Write-Host "  Port: 5173 (default Vite port)" -ForegroundColor Gray
Write-Host "  URL: http://localhost:5173" -ForegroundColor Gray
Write-Host ""

# Start frontend in background
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "$using:projectDir\frontend"
    npm run dev
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Servers Starting..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host ""

# Wait for jobs
try {
    Receive-Job -Job $backendJob, $frontendJob -Wait
} catch {
    Write-Host "Stopping servers..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob, $frontendJob
    Remove-Job -Job $backendJob, $frontendJob
}




