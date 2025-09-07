# 🚀 Quick Setup Guide

Due to Python 3.13 compatibility issues, here are the simplified setup commands:

## Option 1: Use the Install Script (Recommended)
```bash
./install.sh
```

## Option 2: Manual Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies (Step by Step)
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Core web framework
pip install fastapi uvicorn python-multipart

# Image processing
pip install opencv-python pillow numpy

# ML dependencies  
pip install torch torchvision matplotlib pandas

# Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. Download SAM Model (2.6GB)
```bash
mkdir -p backend/models
cd backend/models
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```

### 4. Start the Application

**Terminal 1 - Backend:**
```bash
source venv/bin/activate  # Activate environment
python backend/app.py     # Start server
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python -m http.server 3000  # Start local server
```

**Browser:**
Open `http://localhost:3000`

## Quick Test
1. **Backend health**: Visit `http://localhost:8000/health`
2. **Upload images**: Use the web interface
3. **Process**: Click "Process Images" button
4. **Analyze results**: Review in Analysis tab

## Troubleshooting

### If pip install fails:
```bash
# Try with --no-cache-dir
pip install --no-cache-dir torch torchvision

# Or use conda instead
conda install pytorch torchvision -c pytorch
```

### If SAM model download fails:
- Download manually from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- Save to: `backend/models/sam_vit_h_4b8939.pth`

### If backend won't start:
- Check if all dependencies installed: `pip list`
- Verify SAM model exists: `ls -la backend/models/`
- Check Python version: `python --version` (3.8+ required)

## Success Indicators
- ✅ Backend: "SAM model loaded successfully"
- ✅ Frontend: Interface loads with three tabs
- ✅ Processing: Images upload and analyze successfully