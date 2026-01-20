# Restart Backend Server Script
# Run this script as Administrator to properly restart the backend

Write-Host "Stopping backend servers on port 8000..." -ForegroundColor Yellow

# Find all processes using port 8000
$processes = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | 
    Select-Object -ExpandProperty OwningProcess -Unique

foreach ($pid in $processes) {
    try {
        $proc = Get-Process -Id $pid -ErrorAction Stop
        Write-Host "Stopping process: $($proc.ProcessName) (PID: $pid)" -ForegroundColor Cyan
        Stop-Process -Id $pid -Force -ErrorAction Stop
        Write-Host "  ✓ Stopped" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Could not stop PID $pid : $_" -ForegroundColor Red
        Write-Host "  You may need to run this script as Administrator" -ForegroundColor Yellow
    }
}

Start-Sleep -Seconds 2

# Verify port is free
$stillRunning = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($stillRunning) {
    Write-Host "`n⚠️  Port 8000 is still in use. Please manually stop the process or run as Administrator." -ForegroundColor Red
    Write-Host "Processes still using port 8000:" -ForegroundColor Yellow
    $stillRunning | ForEach-Object {
        $pid = $_.OwningProcess
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        Write-Host "  - $($proc.ProcessName) (PID: $pid)" -ForegroundColor Yellow
    }
    exit 1
}

Write-Host "`nStarting backend server..." -ForegroundColor Green
Set-Location $PSScriptRoot
python app.py

