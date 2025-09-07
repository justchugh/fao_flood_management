from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import os

# Disable MPS to avoid float64 compatibility issues with SAM
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import json
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

app = FastAPI(title="Flood Damage Assessment API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for SAM model
sam_model = None
mask_generator = None
current_device = "cpu"  # Track which device we're using

# Configuration
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"  # Relative to backend/ directory
MODEL_TYPE = "vit_h"
UPLOADS_DIR = "../uploads"  # Relative to backend/ directory

# Crop revenues (NPR per square meter)
CROP_REVENUES = {
    "Sugarcane": 66.06,
    "Potatoes": 40.18,
    "Jute": 31.34,
    "Maize": 15.18,
    "Paddy (Rice)": 12.02,
    "Wheat": 6.93,
    "Lentils": 6.14,
}

# Land values (NPR per square meter)
LAND_VALUES = {
    "Prime Agricultural (Near Road/Irrigation)": 2950.00,
    "Standard Agricultural": 1475.00,
    "Rural/Remote Agricultural": 590.00,
}

# Conversion factor from notebook
M2_PER_PIXEL = 0.0771

class SegmentationRequest(BaseModel):
    image_before: str  # base64 encoded
    image_after: str   # base64 encoded

class FinancialRequest(BaseModel):
    pre_flood_area: float
    post_flood_area: float
    crop_type: str
    land_type: str
    custom_crop_revenue: Optional[float] = None
    custom_land_value: Optional[float] = None

class CalibrationRequest(BaseModel):
    parcel_id: int
    actual_area_m2: float
    analysis_results: dict  # The original analysis results to recalibrate

@app.on_event("startup")
async def startup_event():
    """Load SAM model on startup"""
    global sam_model, mask_generator, current_device
    
    try:
        if not os.path.exists(SAM_CHECKPOINT):
            raise FileNotFoundError(f"SAM model not found at {SAM_CHECKPOINT}")
        
        print(f"Loading SAM model from {SAM_CHECKPOINT}...")
        sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        
        # Use CUDA GPU if available, fallback to CPU
        # Note: MPS (Apple Silicon) has float64 compatibility issues with SAM
        if torch.cuda.is_available():
            device = "cuda"
            print("Using NVIDIA CUDA GPU acceleration")
        else:
            device = "cpu"
            if torch.backends.mps.is_available():
                print("MPS available but using CPU (SAM has float64 compatibility issues with MPS)")
            else:
                print("Using CPU (no GPU acceleration available)")
        
        try:
            sam_model.to(device=device)
            current_device = device  # Store the successful device
        except Exception as e:
            print(f"Failed to move model to {device}, falling back to CPU: {e}")
            device = "cpu"
            current_device = "cpu"
            sam_model.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(sam_model)
        print(f"SAM model loaded successfully on {current_device}")
        
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        raise e

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return opencv_image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union for two boolean masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0

def show_masks_with_ids(image: np.ndarray, masks: List[Dict]) -> np.ndarray:
    """Overlay masks on image with ID labels"""
    if not masks:
        return image
    
    overlay = image.copy()
    
    for i, ann in enumerate(masks):
        mask = ann['segmentation']
        
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Apply random color
        color = np.random.random(3) * 255
        overlay[mask_resized] = overlay[mask_resized] * 0.6 + color * 0.4
        
        # Add text label
        y, x = np.where(mask_resized)
        if len(x) > 0 and len(y) > 0:
            cx, cy = int(x.mean()), int(y.mean())
            cv2.putText(
                overlay, str(i), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
            )
    
    return overlay.astype(np.uint8)

def visualize_changes(image: np.ndarray, matches: List[Dict]) -> np.ndarray:
    """Create visualization with change detection colors"""
    overlay = image.copy()
    
    for match in matches:
        if match['status'] == 'Present':
            mask_id = match['best_match_id']
            if mask_id >= 0:
                # This would need the mask data - simplified for now
                pass
    
    return overlay

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Flood Damage Assessment API is running"}

@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy",
        "sam_loaded": sam_model is not None,
        "device": current_device,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload and save image file"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Save file
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"filename": file.filename, "message": "File uploaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/segment")
async def segment_images(request: SegmentationRequest):
    """Perform SAM segmentation on before and after images"""
    try:
        print(f"Received segment request")
        if mask_generator is None:
            raise HTTPException(status_code=503, detail="SAM model not loaded")
        
        # Decode images
        image_before = decode_base64_image(request.image_before)
        image_after = decode_base64_image(request.image_after)
        
        # Convert to RGB for SAM (ensure numpy arrays, not tensors)
        image_before_rgb = cv2.cvtColor(image_before, cv2.COLOR_BGR2RGB)
        image_after_rgb = cv2.cvtColor(image_after, cv2.COLOR_BGR2RGB)
        
        # Ensure images are numpy arrays (not torch tensors)
        if hasattr(image_before_rgb, 'cpu'):
            image_before_rgb = image_before_rgb.cpu().numpy()
        if hasattr(image_after_rgb, 'cpu'):
            image_after_rgb = image_after_rgb.cpu().numpy()
        
        # Generate masks
        print("Generating masks for before image...")
        masks_before = mask_generator.generate(image_before_rgb)
        print(f"Generated {len(masks_before)} masks for before image")
        
        print("Generating masks for after image...")
        masks_after = mask_generator.generate(image_after_rgb)
        print(f"Generated {len(masks_after)} masks for after image")
        
        # Calculate areas and match masks using IoU
        matches = []
        iou_threshold = 0.5
        
        for i, ann_before in enumerate(masks_before):
            mask_before = ann_before['segmentation']
            best_iou = 0
            best_match_id = -1
            
            for j, ann_after in enumerate(masks_after):
                mask_after = ann_after['segmentation']
                iou = calculate_iou(mask_before, mask_after)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = j
            
            # Determine status
            status = "Present" if best_iou >= iou_threshold else "Lost"
            
            # Calculate areas
            area_before_px = int(ann_before['area'])
            area_before_m2 = area_before_px * M2_PER_PIXEL
            
            area_after_px = 0
            area_after_m2 = 0
            loss_percentage = 100.0
            
            if status == "Present" and best_match_id >= 0:
                area_after_px = int(masks_after[best_match_id]['area'])
                area_after_m2 = area_after_px * M2_PER_PIXEL
                loss_percentage = ((area_before_m2 - area_after_m2) / area_before_m2) * 100
            
            matches.append({
                "parcel_id": i,
                "area_before_px": area_before_px,
                "area_before_m2": round(area_before_m2, 2),
                "area_after_px": area_after_px,
                "area_after_m2": round(area_after_m2, 2),
                "status": status,
                "loss_percentage": round(loss_percentage, 1),
                "iou_score": round(best_iou, 3),
                "best_match_id": best_match_id if best_match_id >= 0 else None
            })
        
        # Create visualizations
        before_viz = show_masks_with_ids(image_before_rgb, masks_before)
        after_viz = show_masks_with_ids(image_after_rgb, masks_after)
        
        # Convert back to base64
        before_viz_b64 = encode_image_to_base64(cv2.cvtColor(before_viz, cv2.COLOR_RGB2BGR))
        after_viz_b64 = encode_image_to_base64(cv2.cvtColor(after_viz, cv2.COLOR_RGB2BGR))
        
        return {
            "masks_before_count": len(masks_before),
            "masks_after_count": len(masks_after),
            "matches": matches,
            "before_visualization": before_viz_b64,
            "after_visualization": after_viz_b64,
            "conversion_factor": M2_PER_PIXEL
        }
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

@app.post("/calculate")
async def calculate_financial_impact(request: FinancialRequest):
    """Calculate financial impact of flood damage"""
    try:
        # Get crop revenue
        crop_revenue = request.custom_crop_revenue or CROP_REVENUES.get(request.crop_type)
        if not crop_revenue:
            raise HTTPException(status_code=400, detail=f"Unknown crop type: {request.crop_type}")
        
        # Get land value
        land_value = request.custom_land_value or LAND_VALUES.get(request.land_type)
        if not land_value:
            raise HTTPException(status_code=400, detail=f"Unknown land type: {request.land_type}")
        
        # Calculate crop revenue impact
        pre_flood_crop_revenue = request.pre_flood_area * crop_revenue
        post_flood_crop_revenue = request.post_flood_area * crop_revenue
        crop_loss = pre_flood_crop_revenue - post_flood_crop_revenue
        
        # Calculate land value impact
        pre_flood_land_value = request.pre_flood_area * land_value
        post_flood_land_value = request.post_flood_area * land_value
        land_loss = pre_flood_land_value - post_flood_land_value
        
        # Total loss
        total_loss = crop_loss + land_loss
        area_lost = request.pre_flood_area - request.post_flood_area
        
        return {
            "area_lost_m2": round(area_lost, 2),
            "crop_revenue_loss": round(crop_loss, 2),
            "land_value_loss": round(land_loss, 2),
            "total_loss": round(total_loss, 2),
            "pre_flood_crop_revenue": round(pre_flood_crop_revenue, 2),
            "post_flood_crop_revenue": round(post_flood_crop_revenue, 2),
            "pre_flood_land_value": round(pre_flood_land_value, 2),
            "post_flood_land_value": round(post_flood_land_value, 2),
            "crop_type": request.crop_type,
            "land_type": request.land_type,
            "crop_revenue_per_m2": crop_revenue,
            "land_value_per_m2": land_value
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial calculation failed: {e}")

@app.get("/crop-types")
async def get_crop_types():
    """Get available crop types and their revenues"""
    return CROP_REVENUES

@app.get("/land-types")
async def get_land_types():
    """Get available land types and their values"""
    return LAND_VALUES

@app.post("/calibrate")
async def calibrate_areas(request: CalibrationRequest):
    """Recalibrate all area measurements based on user-provided reference parcel"""
    try:
        # Find the reference parcel
        reference_parcel = None
        for match in request.analysis_results.get('matches', []):
            if match.get('parcel_id') == request.parcel_id:
                reference_parcel = match
                break
        
        if not reference_parcel:
            raise HTTPException(status_code=400, detail=f"Parcel ID {request.parcel_id} not found")
        
        # Calculate new conversion factor from the reference parcel
        reference_pixels = reference_parcel.get('area_before_px', 0)
        if reference_pixels == 0:
            raise HTTPException(status_code=400, detail="Reference parcel has zero pixel area")
        
        # New conversion factor: user_area_m2 / pixel_area
        new_conversion_factor = request.actual_area_m2 / reference_pixels
        
        # Recalculate all areas with new conversion factor
        recalibrated_matches = []
        for match in request.analysis_results.get('matches', []):
            area_before_px = match.get('area_before_px', 0)
            area_after_px = match.get('area_after_px', 0)
            
            # Recalculate areas
            area_before_m2 = area_before_px * new_conversion_factor
            area_after_m2 = area_after_px * new_conversion_factor
            
            # Recalculate loss percentage
            loss_percentage = 100.0
            if area_before_m2 > 0:
                loss_percentage = ((area_before_m2 - area_after_m2) / area_before_m2) * 100
            
            # Create updated match
            updated_match = match.copy()
            updated_match['area_before_m2'] = round(area_before_m2, 2)
            updated_match['area_after_m2'] = round(area_after_m2, 2)
            updated_match['loss_percentage'] = round(loss_percentage, 1)
            
            recalibrated_matches.append(updated_match)
        
        # Create recalibrated results
        recalibrated_results = request.analysis_results.copy()
        recalibrated_results['matches'] = recalibrated_matches
        recalibrated_results['conversion_factor'] = new_conversion_factor
        recalibrated_results['calibration_info'] = {
            'reference_parcel_id': request.parcel_id,
            'reference_actual_area_m2': request.actual_area_m2,
            'reference_pixel_area': reference_pixels,
            'old_conversion_factor': M2_PER_PIXEL,
            'new_conversion_factor': new_conversion_factor
        }
        
        return recalibrated_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)