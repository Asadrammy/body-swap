# Live Logs Setup - Testing Stability AI API

This setup enables **live logging** in the terminal to monitor all API calls, especially Stability AI API calls, when you interact with the frontend.

## 🚀 Quick Start

### Windows (PowerShell - Recommended)
```powershell
.\start_with_logs.ps1
```

### Windows (Command Prompt)
```cmd
start_with_logs.bat
```

### Linux/Mac
```bash
chmod +x start_with_logs.sh
./start_with_logs.sh
```

## 📋 What You'll See

### Backend Terminal Logs
- ✅ All incoming API requests from frontend
- ✅ Stability AI API calls with full details:
  - Request URL and headers
  - Image size and parameters
  - Full prompts (positive and negative)
  - Response status codes
  - Response time
  - Credit consumption status
- ✅ Pipeline processing steps
- ✅ Job status updates

### Frontend Terminal Logs
- ✅ All API requests sent to backend
- ✅ API responses received
- ✅ User interactions (button clicks, form submissions)
- ✅ Error messages

## 🔍 Testing Stability AI API

1. **Start the servers** using one of the scripts above
2. **Open the frontend** in your browser: http://localhost:5173
3. **Open browser console** (F12 → Console tab) to see frontend logs
4. **Watch the backend terminal** for detailed API logs
5. **Upload photos and submit** - you'll see:
   - Frontend logs in browser console
   - Backend logs in terminal showing:
     - Request received
     - Files saved
     - Pipeline started
     - **Stability AI API calls with full details**
     - Response received
     - Job completion

## 📊 Log Details

### Stability AI API Logs Include:
```
🔑 STABILITY AI API CALL - LIVE LOGS
📤 API Endpoint: https://api.stability.ai/v2beta/stable-image/edit/inpaint
🔑 API Key (first 20 chars): sk-VgJt8yVm3qX4GqLw...
📐 Image size: (1024, 1024) (1048576 pixels)
🎭 Mask provided: True/False
💬 Full Prompt: [your prompt]
🚫 Negative Prompt: [negative prompt]
⚙️  Strength: 0.8
⏱️  Sending request...
📥 API Response received after X.XX seconds
📊 Response Status Code: 200
✅ STABILITY AI API CALL SUCCESSFUL - CREDITS CONSUMED
```

### Error Logs Include:
- ❌ 402: Insufficient credits
- ❌ 401: Invalid API key
- ❌ 429: Rate limit exceeded
- ❌ Network errors

## 🛠️ Manual Start (Alternative)

If you prefer to start manually:

### Backend
```bash
cd face-body-swap
set LOG_LEVEL=DEBUG
set PYTHONUNBUFFERED=1
python -m src.api.main
```

### Frontend (in another terminal)
```bash
cd face-body-swap/frontend
npm run dev
```

## 📝 Notes

- **All logs are real-time** - you'll see them as they happen
- **DEBUG level logging** is enabled for maximum detail
- **Frontend logs** appear in browser console (F12)
- **Backend logs** appear in the terminal where you started the server
- **Stability AI API calls** are logged with full request/response details

## ✅ Verifying API Key Works

When you submit a job, watch for:
1. ✅ "STABILITY AI API CALL SUCCESSFUL" message
2. ✅ Status code 200
3. ✅ "CREDITS CONSUMED" confirmation
4. ✅ Generated image received

If you see errors:
- ❌ 402: Need to purchase credits at https://platform.stability.ai/account/credits
- ❌ 401: Check your API key in `.env` file
- ❌ 429: Too many requests, wait a moment

## 🎯 Testing All APIs

The logs will show:
- ✅ Template API calls (`/api/v1/templates`)
- ✅ Job creation (`POST /api/v1/swap`)
- ✅ Job status polling (`GET /api/v1/jobs/{id}`)
- ✅ **Stability AI API calls** (during refinement)
- ✅ Result download (`GET /api/v1/jobs/{id}/result`)

All API interactions are logged in real-time!

