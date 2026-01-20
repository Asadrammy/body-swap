#!/bin/bash
# Startup script to run both frontend and backend with live logs

echo "========================================"
echo "Starting Face-Body Swap Application"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "src/api/main.py" ]; then
    echo "Error: Please run this script from the face-body-swap directory"
    exit 1
fi

# Set environment variables for logging
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

echo "[1/2] Starting Backend Server..."
echo "Backend will run on http://localhost:8000"
echo ""

# Start backend in background
python -m src.api.main &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

echo "[2/2] Starting Frontend Development Server..."
echo "Frontend will run on http://localhost:5173"
echo ""

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "Both servers are starting!"
echo "========================================"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "All logs will appear in this terminal."
echo "Press Ctrl+C to stop both servers."
echo ""

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Wait for processes
wait

