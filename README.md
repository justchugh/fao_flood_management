# FAO Flood Damage Assessment Tool

An AI-powered web application for assessing agricultural flood damage using satellite imagery and the Segment Anything Model (SAM). This tool helps organizations quantify flood impact on agricultural land and calculate financial losses.

## Features

- **AI Image Segmentation**: Uses Meta's SAM model for precise agricultural plot identification
- **Flood Damage Analysis**: Compares before/after images to assess damage levels
- **Financial Impact Calculation**: Estimates crop and land value losses in NPR
- **Interactive Visualizations**: View segmented areas, damage maps, and water detection
- **Export Capabilities**: Generate PDF reports and export analysis data
- **Area Calibration**: Fine-tune measurements using known reference areas

## System Requirements

- Python 3.8 or higher
- 8GB+ RAM (recommended for SAM model)
- 4GB free disk space (for model files)
- Modern web browser
- Internet connection (for model download)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/disaster_app.git
cd disaster_app
```

2. Run the setup script:
```bash
./setup.sh
```

This will automatically:
- Create a virtual environment
- Install all Python dependencies
- Download the SAM model (2.4GB)
- Set up the application

## Usage

1. Start the application:
```bash
./start_app.sh
```

2. Access the application at: http://localhost:8080

3. To stop all services:
```bash
./start_app.sh -k
```

## Workflow

1. **Upload Images**: Provide before and after flood satellite images
2. **Process Analysis**: AI segments agricultural plots and detects flood areas
3. **Review Results**: Examine damage assessment and statistics
4. **Financial Impact**: Calculate losses using crop and land values
5. **Export Report**: Generate comprehensive PDF reports

## API Endpoints

- `GET /health` - Check API status and model availability
- `POST /segment` - Process image segmentation and damage analysis
- `POST /calibrate` - Recalibrate area measurements
- `GET /docs` - Interactive API documentation

## Configuration

The application uses predefined crop and land values for Nepal (NPR):

**Crop Revenue (per m²):**
- Sugarcane: 66.06 NPR/m²
- Potatoes: 40.18 NPR/m²
- Jute: 31.34 NPR/m²
- Maize: 15.18 NPR/m²
- Rice: 12.02 NPR/m²
- Wheat: 6.93 NPR/m²
- Lentils: 6.14 NPR/m²

**Land Values (per m²):**
- Prime Agricultural: 2,950 NPR/m²
- Standard Agricultural: 1,475 NPR/m²
- Rural/Remote: 590 NPR/m²

## Technology Stack

- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI Model**: Meta SAM (Segment Anything Model)
- **Computer Vision**: OpenCV, PIL
- **Web Server**: Uvicorn

## File Structure

```
disaster_app/
├── backend/
│   ├── app.py              # FastAPI server
│   └── models/             # AI model files
├── frontend/
│   ├── index.html          # Main interface
│   ├── main.js            # Application logic
│   ├── style.css          # Styling
│   └── banner.png         # Header image
├── setup.sh               # Installation script
├── start_app.sh          # Start/stop application (./start_app.sh -k to stop)
└── requirements.txt       # Python dependencies
```

## Deployment Options

### Server Deployment
1. Clone repository on server
2. Run `./setup.sh`
3. Configure firewall for ports 8000 and 8080
4. Use process manager (PM2, systemd) for production

### Docker Deployment
The application can be containerized for cloud deployment on platforms like AWS, GCP, or Azure.

## Troubleshooting

**Model Loading Issues:**
- Ensure 8GB+ RAM is available
- Check internet connection for model download
- Verify Python version compatibility

**Port Conflicts:**
- Use `./start_app.sh -k` to stop existing processes
- Modify ports in start_app.sh if needed

**Virtual Environment Issues:**
- Re-run `./setup.sh` to recreate environment
- Ensure Python 3.8+ is installed

## Support

For technical support or questions about deployment, please refer to the API documentation at http://localhost:8000/docs when the backend is running.

## License

This project is developed for agricultural disaster management and assessment purposes.