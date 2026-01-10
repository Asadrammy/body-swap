# Server Access Guide

**Status:** ✅ **SERVER IS RUNNING**

## Access URLs

### Main Interface
```
http://localhost:8000
```

### API Documentation (Interactive)
```
http://localhost:8000/docs
```

### Alternative API Docs
```
http://localhost:8000/redoc
```

## Available Endpoints

### Health & Monitoring
- **Health Check:** `http://localhost:8000/health`
- **Metrics:** `http://localhost:8000/metrics`

### API Endpoints
- **Templates:** `http://localhost:8000/api/v1/templates`
- **Create Swap Job:** `POST http://localhost:8000/api/v1/swap`
- **Job Status:** `GET http://localhost:8000/api/v1/jobs/{job_id}`
- **Job Result:** `GET http://localhost:8000/api/v1/jobs/{job_id}/result`
- **Job Bundle:** `GET http://localhost:8000/api/v1/jobs/{job_id}/bundle`
- **Refine Job:** `POST http://localhost:8000/api/v1/jobs/{job_id}/refine`

## Testing Instructions

### 1. Open Browser
Navigate to: **http://localhost:8000**

### 2. Test API Documentation
Navigate to: **http://localhost:8000/docs**

This provides an interactive interface where you can:
- View all available endpoints
- Test endpoints directly
- See request/response schemas
- Upload files for testing

### 3. Test Health Endpoint
Navigate to: **http://localhost:8000/health**

Should return JSON with system status.

### 4. Test Templates Endpoint
Navigate to: **http://localhost:8000/api/v1/templates**

Should return list of available templates.

### 5. Test Swap Endpoint (via /docs)
1. Go to `http://localhost:8000/docs`
2. Find the `/api/v1/swap` endpoint
3. Click "Try it out"
4. Upload customer photos and template
5. Execute and see the response

## Server Information

- **Port:** 8000
- **Host:** 0.0.0.0 (accessible from all interfaces)
- **Status:** Running
- **Process IDs:** 1880, 9608, 14012

## Quick Test Commands

### Using PowerShell:
```powershell
# Test health
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Get templates
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/templates"
```

### Using curl (if available):
```bash
# Test health
curl http://localhost:8000/health

# Get templates
curl http://localhost:8000/api/v1/templates
```

## Stopping the Server

To stop the server, press `Ctrl+C` in the terminal where it's running, or:

```powershell
# Find and stop Python processes
Get-Process python | Stop-Process
```

## Troubleshooting

If endpoints return 404:
- Wait a few seconds for server to fully initialize
- Check that server is running: `netstat -ano | findstr ":8000"`
- Check logs for any errors

If server won't start:
- Check if port 8000 is already in use
- Verify Python dependencies are installed
- Check `logs/app.log` for errors

---

**Server is ready for manual testing!** 🚀






