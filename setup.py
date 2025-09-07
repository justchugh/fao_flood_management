#!/usr/bin/env python3
"""
Setup script for Flood Damage Assessment Tool
Downloads SAM model and sets up the environment
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['backend/models', 'uploads', 'frontend']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_sam_model():
    """Download SAM ViT-H model if not exists"""
    model_path = Path('backend/models/sam_vit_h_4b8939.pth')
    model_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    
    if model_path.exists():
        print(f"✅ SAM model already exists: {model_path}")
        return True
    
    print("📥 Downloading SAM ViT-H model (2.6GB)...")
    print("This may take several minutes depending on your connection...")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            
            # Update progress bar
            bar_length = 30
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f'\r📊 [{bar}] {percent:3.0f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')
        
        urllib.request.urlretrieve(model_url, model_path, progress_hook)
        print(f"\n✅ SAM model downloaded successfully: {model_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download SAM model: {e}")
        print("Please download manually:")
        print(f"URL: {model_url}")
        print(f"Save to: {model_path}")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    requirements_path = Path('backend/requirements.txt')
    
    if not requirements_path.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)
        ])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_torch_installation():
    """Check if PyTorch is properly installed"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
        return True
    except ImportError:
        print("❌ PyTorch not found")
        return False

def test_sam_loading():
    """Test if SAM can be loaded"""
    try:
        from segment_anything import sam_model_registry
        model_path = Path('backend/models/sam_vit_h_4b8939.pth')
        
        if not model_path.exists():
            print("⚠️  SAM model not found - skipping load test")
            return True
            
        print("🧪 Testing SAM model loading...")
        sam = sam_model_registry['vit_h'](checkpoint=str(model_path))
        print("✅ SAM model loads successfully")
        return True
        
    except Exception as e:
        print(f"❌ SAM loading test failed: {e}")
        return False

def create_startup_scripts():
    """Create convenient startup scripts"""
    
    # Backend startup script
    backend_script = """#!/bin/bash
cd backend
source venv/bin/activate 2>/dev/null || echo "Virtual environment not found - using global Python"
python app.py
"""
    
    with open('start_backend.sh', 'w') as f:
        f.write(backend_script)
    os.chmod('start_backend.sh', 0o755)
    
    # Windows batch file
    backend_batch = """@echo off
cd backend
call venv\\Scripts\\activate.bat 2>nul || echo Virtual environment not found - using global Python
python app.py
pause
"""
    
    with open('start_backend.bat', 'w') as f:
        f.write(backend_batch)
    
    print("✅ Created startup scripts: start_backend.sh, start_backend.bat")

def main():
    """Main setup function"""
    print("🚀 Flood Damage Assessment Tool - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_python_dependencies():
        return False
    
    # Check PyTorch
    if not check_torch_installation():
        return False
    
    # Download SAM model
    if not download_sam_model():
        print("⚠️  Setup completed with warnings - model download failed")
    
    # Test SAM loading
    test_sam_loading()
    
    # Create startup scripts
    create_startup_scripts()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: ./start_backend.sh (or python backend/app.py)")
    print("2. Open frontend/index.html in your browser")
    print("3. Upload before/after flood images and analyze!")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)