#!/bin/bash

# FAO Flood Damage Assessment Tool - Unified Start Script
# This script starts both backend and frontend services

echo "Starting FAO Flood Damage Assessment Tool"
echo "=========================================="

# Go to project directory
cd "$(dirname "$0")"

# Kill existing processes
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8080 | xargs -r kill -9 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "python.*http.server" 2>/dev/null || true

sleep 2

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Run: ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated"

# Start backend in background
echo "Starting backend server..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Backend failed to start. Check backend logs."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend in background
echo "Starting frontend server..."
cd frontend
python -m http.server 8080 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

# Check if frontend started successfully
if ! curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "Frontend failed to start."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "Application started successfully!"
echo ""
echo "Services running:"
echo "  Backend API:     http://localhost:8000"
echo "  API Docs:        http://localhost:8000/docs"
echo "  Frontend Web:    http://localhost:8080"
echo ""
echo "To stop all services, run: ./kill_app.sh"
echo "Press Ctrl+C to view logs, or close terminal to run in background"
echo ""

# Keep script running and show logs
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

# Show combined logs
tail -f /dev/null