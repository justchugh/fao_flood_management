#!/usr/bin/env python3
"""Test script to reproduce the segmentation error"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io

def create_test_image(width=100, height=100, color=(255, 0, 0)):
    """Create a simple test image"""
    img = Image.new('RGB', (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def test_segment():
    """Test the segment endpoint"""
    print("Creating test images...")
    
    # Create two simple test images
    before_image = create_test_image(100, 100, (255, 0, 0))  # Red
    after_image = create_test_image(100, 100, (0, 255, 0))   # Green
    
    # Prepare request
    data = {
        "image_before": f"data:image/png;base64,{before_image}",
        "image_after": f"data:image/png;base64,{after_image}"
    }
    
    print("Sending request to /segment endpoint...")
    try:
        response = requests.post(
            "http://localhost:8000/segment",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            print("SUCCESS: Segmentation completed")
            result = response.json()
            print(f"Found {len(result['matches'])} matches")
        else:
            print(f"ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"REQUEST ERROR: {e}")

if __name__ == "__main__":
    test_segment()