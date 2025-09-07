#!/bin/bash

echo "🚀 Flood Damage Assessment Tool - Installation Script"
echo "======================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install fastapi uvicorn python-multipart opencv-python pillow

# Install ML dependencies
echo "🧠 Installing ML dependencies..."
pip install torch torchvision numpy matplotlib pandas

# Install segment-anything
echo "🎯 Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Create directories
echo "📁 Creating directories..."
mkdir -p backend/models uploads

# Download SAM model
echo "📥 Downloading SAM model (2.6GB)..."
cd backend/models
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    echo "✅ SAM model downloaded"
else
    echo "✅ SAM model already exists"
fi

cd ../..

echo ""
echo "🎉 Installation completed!"
echo ""
echo "🚀 To start the application:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Start backend: python backend/app.py"
echo "3. Open frontend: open frontend/index.html"