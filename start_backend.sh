#!/bin/bash

echo "Starting Flood Damage Assessment Backend"
echo "========================================"

# Kill existing processes on port 8000
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true

# Wait a moment
sleep 2

# Go to project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Virtual environment not found. Run: ./setup.sh"
    exit 1
fi

# Change to backend directory
cd backend

# Start the backend
echo "Starting backend server..."
echo "Backend will run on: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo ""

python app.py