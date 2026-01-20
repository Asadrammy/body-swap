# Quick Start - Live Logs for Testing Stability AI API

## 🚀 Start Everything with One Command

### Windows (PowerShell)
```powershell
.\start_with_logs.ps1
```

This will:
1. ✅ Start backend server on http://localhost:8000
2. ✅ Start frontend server on http://localhost:5173
3. ✅ Open separate terminal windows for each
4. ✅ Show all logs in real-time

## 📺 Where to See Logs

### Backend Logs (Terminal Window)
- All API requests from frontend
- **Stability AI API calls with full details**
- Pipeline processing steps
- Job status updates

### Frontend Logs (Browser Console)
1. Open browser: http://localhost:5173
2. Press **F12** to open Developer Tools
3. Click **Console** tab
4. You'll see:
   - All API requests
   - Button clicks
   - Form submissions
   - API responses

## 🧪 Testing Stability AI API

1. **Start servers**: Run `.\start_with_logs.ps1`
2. **Open frontend**: http://localhost:5173
3. **Open browser console**: Press F12 → Console tab
4. **Upload photos** and click "Next"
5. **Select template** and click "Submit"
6. **Watch the logs**:
   - **Browser console**: Shows frontend API calls
   - **Backend terminal**: Shows Stability AI API calls with:
     - Request URL
     - Full prompts
     - Image parameters
     - Response status
     - Credit consumption

## ✅ What to Look For

### Successful API Call
```
🔑 STABILITY AI API CALL - LIVE LOGS
📤 API Endpoint: https://api.stability.ai/v2beta/stable-image/edit/inpaint
✅ STABILITY AI API CALL SUCCESSFUL - CREDITS CONSUMED
```

### Error - Need Credits
```
❌ STABILITY AI API ERROR - Status: 402
❌ STABILITY AI CREDITS REQUIRED
Please purchase credits at: https://platform.stability.ai/account/credits
```

### Error - Invalid API Key
```
❌ STABILITY AI API ERROR - Status: 401
❌ STABILITY AI API KEY INVALID or EXPIRED
```

## 📋 All Logged Events

- ✅ Frontend button clicks
- ✅ File uploads
- ✅ API requests (POST /api/v1/swap)
- ✅ **Stability AI API calls** (with full details)
- ✅ Job status polling
- ✅ Result downloads

## 🛑 Stop Servers

Close the terminal windows or press `Ctrl+C` in each window.

---

**Note**: All logs are in real-time. You'll see everything as it happens!
