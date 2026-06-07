import os
import uuid
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIR         = r'D:\Projects\ChestVision-AI\backend\uploads'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def numpy_to_base64(img_array: np.ndarray) -> str:
    img_uint8 = (img_array * 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8)
    buffer    = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@router.post('/predict')
async def predict_xray(
    file: UploadFile = File(...),
    age:  float      = Form(default=60.0),
    sex:  str        = Form(default='Unknown')
):
    """
    Upload a chest X-ray and get predictions.
    Parameters:
    - file : X-ray image (JPG, PNG, or DICOM)
    - age  : Patient age (default 60 — dataset mean)
    - sex  : Male / Female / Unknown (default Unknown)
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    if not (0 <= age <= 120):
        raise HTTPException(status_code=400, detail="Age must be between 0 and 120")

    if sex not in ['Male', 'Female', 'Unknown']:
        sex = 'Unknown'

    file_id   = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    contents  = await file.read()
    with open(save_path, 'wb') as f:
        f.write(contents)

    from app.main import model_service
    if model_service is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    try:
        results = model_service.run_inference(save_path, age=age, sex=sex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    heatmaps_b64 = {
        disease: numpy_to_base64(data['overlay'])
        for disease, data in results['heatmaps'].items()
    }

    return JSONResponse({
        'scan_id':     file_id,
        'filename':    file.filename,
        'age':         age,
        'sex':         sex,
        'predictions': results['predictions'],
        'heatmaps':    heatmaps_b64,
        'original':    numpy_to_base64(results['img_float']),
        'report':      results['report'],
    })