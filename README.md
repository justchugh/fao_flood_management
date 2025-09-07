# 🌊 Flood Damage Assessment Tool

AI-powered agricultural damage analysis using Segment Anything Model (SAM) for before/after flood imagery comparison and financial impact calculation.

## Features

- **Image Upload**: Upload before and after flood images
- **AI Segmentation**: Automatic parcel detection using SAM ViT-H model
- **Change Detection**: IoU-based comparison to identify lost/damaged areas
- **Financial Analysis**: Calculate crop revenue and land value losses in NPR
- **Export Results**: Generate reports and export analysis data

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js (for development server, optional)
- At least 8GB RAM (for SAM model)
- CUDA GPU recommended (optional but faster)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/justchugh/fao_flood_management.git
   cd fao_flood_management
   ```

2. **Set up Python environment**:
   ```bash
   cd backend
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SAM model** (2.6GB):
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download ViT-H checkpoint
   wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

5. **Run the backend**:
   ```bash
   python app.py
   ```

6. **Open the frontend**:
   - Open `frontend/index.html` in your web browser
   - Or serve with a local server:
     ```bash
     cd ../frontend
     python -m http.server 3000
     # Then visit http://localhost:3000
     ```

## Usage

### 1. Image Upload
- Upload before and after flood images (JPG, PNG supported)
- Images are processed locally - no data leaves your system

### 2. Analysis
- Click "Process Images" to run SAM segmentation
- View detected parcels with IDs and status (Present/Lost)
- Review change detection results

### 3. Financial Impact
- Select parcels for assessment
- Choose crop type and land classification
- Get detailed financial loss calculation in NPR

## API Endpoints

### Backend Server (http://localhost:8000)

- `GET /health` - Check server and SAM model status
- `POST /segment` - Process before/after images
- `POST /calculate` - Calculate financial impact
- `GET /crop-types` - Available crop types and revenues
- `GET /land-types` - Available land types and values

## Configuration

### Crop Revenues (NPR per m²)
- Sugarcane: 66.06
- Potatoes: 40.18
- Jute: 31.34
- Maize: 15.18
- Paddy Rice: 12.02
- Wheat: 6.93
- Lentils: 6.14

### Land Values (NPR per m²)
- Prime Agricultural: 2,950.00
- Standard Agricultural: 1,475.00
- Rural/Remote Agricultural: 590.00

### Area Conversion
- Conversion factor: 0.0771 m²/pixel (calibrated from reference data)

## File Structure

```
disaster_app/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── requirements.txt    # Python dependencies
│   └── models/
│       └── sam_vit_h_4b8939.pth  # SAM model (download required)
├── frontend/
│   ├── index.html          # Main application
│   ├── main.js            # JavaScript functionality
│   └── style.css          # Styling
├── uploads/               # Temporary image storage
└── README.md             # This file
```

## Troubleshooting

### Model Loading Issues
- Ensure SAM model file exists in `backend/models/`
- Check available RAM (model requires ~2.6GB)
- Verify PyTorch installation with CUDA support (if using GPU)

### CORS Errors
- If accessing from different ports, update CORS settings in `app.py`
- Use a local server instead of opening HTML directly

### Out of Memory
- Use CPU instead of CUDA if GPU memory is insufficient
- Process smaller images or reduce batch size

## Performance

### Expected Processing Times
- Image upload: < 2 seconds
- SAM segmentation: 15-60 seconds (depending on image size and hardware)
- Financial calculation: < 1 second
- Report generation: < 5 seconds

### Hardware Recommendations
- **Minimum**: 8GB RAM, CPU-only processing
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage**: 5GB free space (including model)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is developed for FAO flood management research purposes.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review GitHub issues
3. Create a new issue with detailed description

---

**Powered by Segment Anything Model (SAM) - Meta AI Research**