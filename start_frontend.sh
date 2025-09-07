#!/bin/bash

echo "Starting Flood Damage Assessment Frontend"
echo "========================================="

# Kill existing processes on port 3000
echo "Cleaning up existing processes..."
lsof -ti:3000 | xargs -r kill -9 2>/dev/null || true
pkill -f "python.*http.server.*3000" 2>/dev/null || true

# Wait a moment
sleep 2

# Go to project directory and frontend folder
cd "$(dirname "$0")/frontend"

# Start the frontend server
echo "Starting frontend server..."
echo "Frontend available at: http://localhost:3000"
echo "Press Ctrl+C to stop"
echo ""

python -m http.server 3000