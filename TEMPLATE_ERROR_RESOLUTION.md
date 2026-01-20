# Template Loading Error - Resolution Guide

## Problem Summary
Frontend was getting 404 errors when trying to access `/api/v1/templates` endpoint.

## Root Causes Identified

1. **Port Mismatch**: Backend config had port 8001, frontend expected 8000
   - ✅ **Fixed**: Updated `configs/default.yaml` to use port 8000

2. **Multiple Backend Processes**: Conflicting Python processes on port 8000
   - ✅ **Fixed**: Stopped all old processes and started clean backend

3. **Server Startup Time**: Backend takes time to fully initialize (model loading)
   - ⚠️ **Solution**: Wait 10-15 seconds after starting backend

## Verification Steps

### 1. Check Routes Are Registered
```powershell
cd D:\projects\image\face-body-swap
python -c "from src.api.main import app; routes = [r.path for r in app.routes]; print('/api/v1/templates' in routes)"
```
Should return: `True`

### 2. Test Backend Directly
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Templates endpoint
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/templates"
```

### 3. Check Vite Proxy
The frontend Vite config (`frontend/vite.config.ts`) proxies `/api` to `http://localhost:8000`

## Solution Steps

1. **Stop all backend processes**:
   ```powershell
   Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
   ```

2. **Start backend cleanly**:
   ```powershell
   cd D:\projects\image\face-body-swap
   python -m src.api.main
   ```
   OR
   ```powershell
   python app.py
   ```

3. **Wait 10-15 seconds** for backend to fully start

4. **Verify backend is running**:
   - Check http://localhost:8000/health
   - Check http://localhost:8000/api/v1/templates

5. **Start frontend** (if not already running):
   ```powershell
   cd frontend
   npm run dev
   ```

6. **Refresh browser** at http://localhost:3000

## Registered Routes

The backend should have these routes:
- `/api/v1/templates` ✅
- `/api/v1/swap` ✅
- `/api/v1/jobs` ✅
- `/api/v1/jobs/{job_id}` ✅
- `/health` ✅
- `/docs` ✅

## Configuration Files

### `configs/default.yaml`
```yaml
api:
  host: 0.0.0.0
  port: 8000  # Must be 8000 for frontend proxy
```

### `frontend/vite.config.ts`
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    secure: false,
    ws: true,
  },
}
```

## Troubleshooting

### Still Getting 404?
1. Wait 10-15 more seconds (model loading takes time)
2. Check backend PowerShell window for errors
3. Verify port 8000 is not blocked by firewall
4. Try accessing backend directly: http://localhost:8000/api/v1/templates

### Templates Endpoint Returns Empty?
- Check `templates/catalog.yaml` exists
- Verify template files are in `templates/` directory
- Check backend logs for template catalog errors

### Frontend Can't Connect?
- Verify Vite dev server is running on port 3000
- Check browser console for CORS errors
- Ensure backend CORS middleware is enabled

---

**Status**: Routes are correctly registered. If still getting 404, wait for backend to fully start (10-15 seconds).




