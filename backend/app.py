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
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
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

# Global variables for models
sam_model = None
mask_generator = None
flood_model = None
current_device = "cpu"  # Track which device we're using

# Configuration
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"  # Relative to backend/ directory
FLOOD_MODEL_PATH = "models/flood_deeplabv3_resnet50.pth"  # Add flood model path
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

# Water detection preprocessing
flood_preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_flood_model(num_classes=2, use_aux=True):
    """Build DeepLabV3 model for flood detection"""
    return deeplabv3_resnet50(weights=None, num_classes=num_classes, aux_loss=use_aux)

def load_flood_weights(model, ckpt_path):
    """Load flood detection model weights"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            sd = ckpt

        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "").replace("model.", "")
            new_sd[nk] = v

        model.load_state_dict(new_sd, strict=False)
        return model
    except Exception as e:
        print(f"Warning: Could not load flood model weights: {e}")
        return model

def detect_water_flood(img_pil, threshold=0.5, model=None):
    """Model-based water detection using DeepLabV3"""
    if model is None:
        return None, np.zeros((img_pil.size[1], img_pil.size[0]), dtype=bool)

    W, H = img_pil.size
    x = flood_preprocess(img_pil).unsqueeze(0).to(current_device)

    with torch.no_grad():
        try:
            out = model(x)["out"]
            out_up = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            prob = out_up.softmax(1)[0, 1].detach().cpu().numpy()
        except Exception as e:
            print(f"Flood model inference error: {e}")
            return None, np.zeros((H, W), dtype=bool)

    water_mask = (prob > threshold)
    return prob, water_mask

def detect_water_by_edges(img_array):
    """Optimized edge-based water detection with medium cleaning"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    no_edge_mask = ~edges_dilated.astype(bool)

    # Remove small regions
    num_labels, labels = cv2.connectedComponents(no_edge_mask.astype(np.uint8))
    raw_mask = np.zeros_like(no_edge_mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if np.sum(labels == i) > 2000:
            raw_mask[labels == i] = 1

    # Medium cleaning
    num_labels2, labels2 = cv2.connectedComponents(raw_mask)
    cleaned_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    for i in range(1, num_labels2):
        if np.sum(labels2 == i) >= 500:
            cleaned_mask[labels2 == i] = 1

    # Fill holes
    inverted = ~cleaned_mask.astype(bool)
    num_labels_inv, labels_inv = cv2.connectedComponents(inverted.astype(np.uint8))
    for i in range(1, num_labels_inv):
        if np.sum(labels_inv == i) < 1000:
            cleaned_mask[labels_inv == i] = 1

    # Morphological smoothing
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_smooth)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_smooth)

    return cleaned_mask.astype(bool)

def detect_water_hybrid(img_array, img_pil, flood_model_ref):
    """Hybrid detection: 50% model + 50% edges"""
    prob_model, mask_model = detect_water_flood(img_pil, threshold=0.5, model=flood_model_ref)
    mask_edges = detect_water_by_edges(img_array)

    if prob_model is not None:
        combined_prob = (prob_model * 0.50 + mask_edges.astype(float) * 0.50)
    else:
        # Fallback to edges only if model fails
        combined_prob = mask_edges.astype(float)
    
    final_mask = combined_prob > 0.4
    return combined_prob, final_mask

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
    """Load SAM and flood models on startup"""
    global sam_model, mask_generator, flood_model, current_device
    
    try:
        # Load SAM model
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
        
        # Load flood detection model
        print("Loading flood detection model...")
        try:
            flood_model = build_flood_model(num_classes=2, use_aux=True)
            if os.path.exists(FLOOD_MODEL_PATH):
                flood_model = load_flood_weights(flood_model, FLOOD_MODEL_PATH)
                print(f"Flood model weights loaded from {FLOOD_MODEL_PATH}")
            else:
                print(f"Warning: Flood model weights not found at {FLOOD_MODEL_PATH}, using untrained model")
            
            flood_model = flood_model.to(device=current_device).eval()
            print(f"Flood detection model loaded successfully on {current_device}")
        except Exception as e:
            print(f"Warning: Failed to load flood detection model: {e}")
            flood_model = None
        
    except Exception as e:
        print(f"Error loading models: {e}")
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

def calculate_parcel_flood_damage(before_masks, water_mask, image_shape):
    """Calculate flood damage by overlaying before parcels on water mask"""
    results = []
    
    for i, mask_data in enumerate(before_masks):
        parcel_mask = mask_data['segmentation']
        parcel_area = mask_data['area']
        
        # Check overlap with water
        overlap_mask = parcel_mask & water_mask
        flooded_area = np.sum(overlap_mask)
        
        # Calculate percentages
        flood_percentage = (flooded_area / parcel_area * 100) if parcel_area > 0 else 0
        remaining_area = parcel_area - flooded_area
        remaining_percentage = 100 - flood_percentage
        
        # Status determination
        if flood_percentage >= 50:
            status = "Heavily Damaged"
        elif flood_percentage >= 20:
            status = "Moderately Damaged"
        elif flood_percentage > 0:
            status = "Lightly Damaged"
        else:
            status = "Undamaged"
        
        results.append({
            'parcel_id': i,
            'original_area_px': int(parcel_area),
            'flooded_area_px': int(flooded_area),
            'remaining_area_px': int(remaining_area),
            'original_area_m2': round(parcel_area * M2_PER_PIXEL, 2),
            'flooded_area_m2': round(flooded_area * M2_PER_PIXEL, 2),
            'remaining_area_m2': round(remaining_area * M2_PER_PIXEL, 2),
            'flood_percentage': round(flood_percentage, 1),
            'remaining_percentage': round(remaining_percentage, 1),
            'damage_status': status
        })
    
    return results

def create_water_overlay(image, water_mask):
    """Create water detection visualization"""
    overlay = image.copy()
    overlay[water_mask] = overlay[water_mask] * 0.5 + np.array([0, 100, 200]) * 0.5
    return overlay

def create_land_only_image(image, water_mask):
    """Create land-only visualization by masking water areas"""
    land_only = image.copy()
    land_only[water_mask] = [0, 0, 0]
    return land_only

def visualize_parcel_damage(image, before_masks, water_mask, damage_results, max_parcels=50):
    """Visualize parcels with color-coded damage levels"""
    overlay = image.copy().astype(np.float32)
    
    damage_colors = {
        'Undamaged': [0, 255, 0],        # Green
        'Lightly Damaged': [255, 255, 0], # Yellow
        'Moderately Damaged': [255, 165, 0], # Orange
        'Heavily Damaged': [255, 0, 0]    # Red
    }
    
    for i in range(min(max_parcels, len(before_masks))):
        mask = before_masks[i]['segmentation']
        status = damage_results[i]['damage_status']
        color = damage_colors[status]
        
        # Apply color with transparency
        overlay[mask] = overlay[mask] * 0.7 + np.array(color) * 0.3
        
        # Add parcel ID
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = int(x_indices.mean())
            center_y = int(y_indices.mean())
            
            overlay = cv2.putText(overlay.astype(np.uint8), str(i), (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            overlay = cv2.putText(overlay, str(i), (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return overlay.astype(np.uint8)

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
    """Enhanced flood damage assessment: SAM on before image + hybrid water detection on after image"""
    try:
        print("Received enhanced segment request")
        if mask_generator is None:
            raise HTTPException(status_code=503, detail="SAM model not loaded")
        
        # Decode images
        image_before = decode_base64_image(request.image_before)
        image_after = decode_base64_image(request.image_after)
        
        # Convert to RGB for processing
        image_before_rgb = cv2.cvtColor(image_before, cv2.COLOR_BGR2RGB)
        image_after_rgb = cv2.cvtColor(image_after, cv2.COLOR_BGR2RGB)
        
        # Ensure images match dimensions
        if image_before_rgb.shape != image_after_rgb.shape:
            print(f"Resizing images to match: {image_before_rgb.shape} -> {image_after_rgb.shape}")
            before_pil = Image.fromarray(image_before_rgb)
            after_pil = Image.fromarray(image_after_rgb)
            if before_pil.size != after_pil.size:
                after_pil = after_pil.resize(before_pil.size, Image.LANCZOS)
                image_after_rgb = np.array(after_pil)
        
        # Step 1: Run SAM on BEFORE image only
        print("Running SAM on before image only...")
        masks_before = mask_generator.generate(image_before_rgb)
        print(f"Generated {len(masks_before)} parcels from before image")
        
        # Step 2: Apply hybrid water detection on AFTER image
        print("Applying hybrid water detection on after image...")
        after_pil = Image.fromarray(image_after_rgb)
        prob_hybrid, water_mask_hybrid = detect_water_hybrid(image_after_rgb, after_pil, flood_model)
        water_coverage = (water_mask_hybrid.sum() / water_mask_hybrid.size) * 100
        print(f"Water coverage: {water_coverage:.1f}%")
        
        # Step 3: Calculate flood damage by overlaying parcels on water mask
        print("Calculating flood damage...")
        damage_results = calculate_parcel_flood_damage(masks_before, water_mask_hybrid, image_after_rgb.shape)
        
        # Create 4 visualizations
        print("Creating visualizations...")
        
        # 1. Before image with SAM parcels
        before_with_parcels = show_masks_with_ids(image_before_rgb, masks_before)
        
        # 2. Water detection overlay
        water_overlay = create_water_overlay(image_after_rgb, water_mask_hybrid)
        
        # 3. Land-only after image
        land_only = create_land_only_image(image_after_rgb, water_mask_hybrid)
        
        # 4. Flood damage assessment
        damage_viz = visualize_parcel_damage(image_after_rgb, masks_before, water_mask_hybrid, damage_results)
        
        # Convert to base64
        before_viz_b64 = encode_image_to_base64(cv2.cvtColor(before_with_parcels, cv2.COLOR_RGB2BGR))
        water_viz_b64 = encode_image_to_base64(cv2.cvtColor(water_overlay, cv2.COLOR_RGB2BGR))
        land_viz_b64 = encode_image_to_base64(cv2.cvtColor(land_only, cv2.COLOR_RGB2BGR))
        damage_viz_b64 = encode_image_to_base64(cv2.cvtColor(damage_viz, cv2.COLOR_RGB2BGR))
        
        # Calculate statistics
        damage_stats = {}
        for result in damage_results:
            status = result['damage_status']
            damage_stats[status] = damage_stats.get(status, 0) + 1
        
        total_original_area_m2 = sum(r['original_area_m2'] for r in damage_results)
        total_flooded_area_m2 = sum(r['flooded_area_m2'] for r in damage_results)
        total_remaining_area_m2 = sum(r['remaining_area_m2'] for r in damage_results)
        
        return {
            "parcels_detected": len(masks_before),
            "water_coverage_percentage": round(water_coverage, 1),
            "damage_results": damage_results,
            "damage_statistics": damage_stats,
            "total_areas": {
                "original_m2": round(total_original_area_m2, 2),
                "flooded_m2": round(total_flooded_area_m2, 2),
                "remaining_m2": round(total_remaining_area_m2, 2)
            },
            "visualizations": {
                "before_sam_results": before_viz_b64,
                "water_detection": water_viz_b64,
                "land_only": land_viz_b64,
                "flood_damage_assessment": damage_viz_b64
            },
            "conversion_factor": M2_PER_PIXEL,
            "processing_method": "Enhanced: SAM once + hybrid water detection"
        }
        
    except Exception as e:
        print(f"Enhanced segmentation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced segmentation failed: {e}")

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