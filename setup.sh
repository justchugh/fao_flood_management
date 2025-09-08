#!/bin/bash

# FAO Flood Damage Assessment Tool - Setup Script
# This script installs all dependencies and downloads required models

set -e

echo "Setting up FAO Flood Damage Assessment Tool..."

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]]; then
    echo "Error: Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create models directory
mkdir -p backend/models

# Download SAM model if not exists
SAM_MODEL="backend/models/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_MODEL" ]; then
    echo "Downloading SAM model (2.4GB)..."
    cd backend/models
    wget -O sam_vit_h_4b8939.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    cd ../..
else
    echo "SAM model already exists."
fi

# Make scripts executable
chmod +x start_app.sh
chmod +x start_backend.sh
chmod +x start_frontend.sh
chmod +x kill_app.sh

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  ./start_app.sh        # Start both services (recommended)"
echo "  ./kill_app.sh         # Stop all services"
echo ""
echo "Alternative (separate services):"
echo "  ./start_backend.sh    # Start API server only (port 8000)"
echo "  ./start_frontend.sh   # Start web interface only (port 8080)"
echo ""
echo "Access the application at: http://localhost:8080"