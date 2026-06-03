import os
import uuid
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIR = r'D:\Projects\ChestVision-AI\backend\uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image array (float 0-1) to base64 PNG string."""
    img_uint8 = (img_array * 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8)
    buffer    = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@router.post('/predict')
async def predict_xray(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image and get:
    - Disease predictions with confidence scores
    - Grad-CAM heatmap overlays (base64)
    - Structured report
    """
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save uploaded file
    file_id   = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    contents = await file.read()
    with open(save_path, 'wb') as f:
        f.write(contents)

    # Import model service (loaded at startup in main.py)
    from app.main import model_service
    if model_service is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    try:
        results = model_service.run_inference(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Convert heatmap overlays to base64 for JSON response
    heatmaps_b64 = {}
    for disease, data in results['heatmaps'].items():
        heatmaps_b64[disease] = numpy_to_base64(data['overlay'])

    # Convert original image to base64
    original_b64 = numpy_to_base64(results['img_float'])

    return JSONResponse({
        'scan_id':     file_id,
        'filename':    file.filename,
        'predictions': results['predictions'],
        'heatmaps':    heatmaps_b64,
        'original':    original_b64,
        'report':      results['report'],
    })