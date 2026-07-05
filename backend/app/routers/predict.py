import os
import uuid
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.config import UPLOAD_DIR

router = APIRouter()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def is_valid_xray(img_array: np.ndarray) -> tuple:
    """
    Validates image as chest X-ray using radiological physics principles.
    Based on Beer-Lambert Law: X-rays have specific attenuation patterns.

    X-ray characteristics:
    - Grayscale by physics (single energy photon absorption)
    - High dynamic range (lung=dark/low attenuation, bone=bright/high attenuation)
    - Wide histogram spread (not clustered like regular photos)
    - Must have both dark regions (lungs) and bright regions (bones)
    """
    gray = np.mean(img_array, axis=2)

    # ── Check 1: X-rays are grayscale by physics ──────────────
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)
    color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))

    if color_diff > 20:
        return False, (
            "Image appears to be a color photograph, not a chest X-ray. "
            "Please upload a grayscale chest X-ray image."
        )

    # ── Check 2: Sufficient contrast (dynamic range) ──────────
    std_dev = np.std(gray)
    if std_dev < 25:
        return False, (
            "Image has insufficient contrast. "
            "A valid chest X-ray should have clear contrast between "
            "lung fields and bone structures."
        )

    # ── Check 3: Must have both dark AND bright regions ───────
    dark_ratio   = np.sum(gray < 80)  / gray.size   # lung fields
    bright_ratio = np.sum(gray > 180) / gray.size   # bones/spine

    if dark_ratio < 0.05:
        return False, (
            "Image does not show expected dark lung field regions. "
            "Please upload a frontal chest X-ray."
        )

    if bright_ratio < 0.03:
        return False, (
            "Image does not show expected bright bone structures. "
            "Please upload a frontal chest X-ray."
        )

    # ── Check 4: Histogram spread ─────────────────────────────
    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    non_zero_bins = np.sum(hist > 0)

    if non_zero_bins < 10:
        return False, (
            "Image intensity distribution is too narrow. "
            "This does not appear to be a medical X-ray."
        )

    # ── Check 5: Reasonable image dimensions ──────────────────
    h, w = gray.shape
    if h < 100 or w < 100:
        return False, "Image is too small. Please upload a full-resolution chest X-ray."

    if h > 10000 or w > 10000:
        return False, "Image is too large. Please upload a standard resolution chest X-ray."

    return True, "Valid chest X-ray"


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
    - file : Chest X-ray image (JPG, PNG, or DICOM)
    - age  : Patient age (default 60 — dataset mean)
    - sex  : Male / Female / Unknown (default Unknown)

    Returns:
    - predictions, heatmaps, report, scan metadata
    """
    # ── Validate file type ────────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed formats: JPG, JPEG, PNG, DICOM (.dcm)"
        )

    # ── Validate demographics ─────────────────────────────────
    if not (0 <= age <= 120):
        raise HTTPException(
            status_code=400,
            detail="Age must be between 0 and 120"
        )

    if sex not in ['Male', 'Female', 'Unknown']:
        sex = 'Unknown'

    # ── Save uploaded file ────────────────────────────────────
    file_id   = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    contents  = await file.read()

    with open(save_path, 'wb') as f:
        f.write(contents)

    # ── Validate X-ray (skip for DICOM — already medical format) ──
    if ext != '.dcm':
        try:
            img       = Image.open(save_path).convert('RGB')
            img_array = np.array(img)
            is_valid, message = is_valid_xray(img_array)

            if not is_valid:
                # Clean up the saved file
                os.remove(save_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image: {message}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process image: {str(e)}"
            )

    # ── Run inference ─────────────────────────────────────────
    from app.main import model_service
    if model_service is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet. Please try again.')

    try:
        results = model_service.run_inference(save_path, age=age, sex=sex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # ── Convert heatmaps to base64 ────────────────────────────
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