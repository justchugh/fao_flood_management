#!/bin/bash

echo "Stopping Flood Damage Assessment Application"
echo "============================================"

# Kill backend (port 8000)
echo "Killing backend processes..."
lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true

# Kill frontend (port 3000)  
echo "Killing frontend processes..."
lsof -ti:3000 | xargs -r kill -9 2>/dev/null || true
pkill -f "python.*http.server.*3000" 2>/dev/null || true

# Kill any other related processes
pkill -f "uvicorn" 2>/dev/null || true

echo "All application processes stopped"

# Show what's still running (optional)
echo ""
echo "Checking ports 8000 and 3000:"
lsof -i:8000,3000 2>/dev/null || echo "   No processes found on ports 8000 or 3000"