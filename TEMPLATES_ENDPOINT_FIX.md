# Templates Endpoint 404 Fix

## Issue
Frontend getting 404 when accessing `/api/v1/templates` endpoint.

## Root Cause
The backend server needs to be restarted to pick up route registrations, or there may be an exception during route initialization.

## Solution

### Step 1: Restart Backend Server

**Stop the current backend:**
- Press `Ctrl+C` in the terminal where `python app.py` is running

**Start it again:**
```bash
cd face-body-swap
python app.py
```

### Step 2: Verify Backend is Running

Check that the backend is accessible:
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test templates endpoint
curl http://localhost:8000/api/v1/templates
```

### Step 3: Check Backend Logs

When you start the backend, you should see:
```
✓ Static files mounted at /static from: ...
Project root: ...
```

If you see any import errors or exceptions, those need to be fixed first.

### Step 4: Verify Vite Proxy

The Vite dev server proxy should automatically forward `/api` requests to `http://localhost:8000`.

**Check `frontend/vite.config.ts`:**
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}
```

If you modified `vite.config.ts`, restart the frontend dev server:
```bash
cd frontend
npm run dev
```

## Verification

1. **Backend running**: `http://localhost:8000/docs` should show API documentation
2. **Templates endpoint**: `http://localhost:8000/api/v1/templates` should return JSON
3. **Frontend**: `http://localhost:3000` should load templates without 404

## Common Issues

### Issue 1: Backend Not Running
**Symptom**: All API calls return connection errors
**Fix**: Start backend with `python app.py`

### Issue 2: Port Already in Use
**Symptom**: `Address already in use` error
**Fix**: 
```bash
# Find process using port 8000
netstat -ano | findstr :8000
# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Issue 3: Import Errors
**Symptom**: Backend fails to start with import errors
**Fix**: Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue 4: Template Catalog Not Loading
**Symptom**: Templates endpoint returns empty list or error
**Fix**: Check that `data/templates/catalog.json` exists and is valid

## Testing

After restarting:

1. **Test from browser console:**
   ```javascript
   fetch('http://localhost:8000/api/v1/templates')
     .then(r => r.json())
     .then(console.log)
   ```

2. **Test from terminal:**
   ```bash
   curl http://localhost:8000/api/v1/templates
   ```

3. **Check frontend**: Templates should load in Step 2

---

**Status**: ✅ Fix Applied - Restart Backend Required




