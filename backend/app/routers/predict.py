import os
import uuid
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import UPLOAD_DIR, MODEL_VERSION
from app.rate_limit import limiter, daily_counter
from app import db

router = APIRouter()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)


def is_valid_xray(img_array: np.ndarray) -> tuple:
    """
    Validates image as chest X-ray using radiological physics principles.
    Based on Beer-Lambert Law: X-rays have specific attenuation patterns.
    """
    gray = np.mean(img_array, axis=2)

    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)
    color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))

    if color_diff > 20:
        return False, (
            "Image appears to be a color photograph, not a chest X-ray. "
            "Please upload a grayscale chest X-ray image."
        )

    std_dev = np.std(gray)
    if std_dev < 25:
        return False, (
            "Image has insufficient contrast. "
            "A valid chest X-ray should have clear contrast between "
            "lung fields and bone structures."
        )

    dark_ratio   = np.sum(gray < 80)  / gray.size
    bright_ratio = np.sum(gray > 180) / gray.size

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

    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    non_zero_bins = np.sum(hist > 0)

    if non_zero_bins < 10:
        return False, (
            "Image intensity distribution is too narrow. "
            "This does not appear to be a medical X-ray."
        )

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
@limiter.limit("5/minute")
async def predict_xray(
    request: Request,
    file: UploadFile = File(...),
    age:  float      = Form(default=60.0),
    sex:  str        = Form(default='Unknown')
):
    """
    Upload a chest X-ray and get predictions.

    Parameters:
    - file : Chest X-ray image (JPG, PNG, or DICOM), max 15MB
    - age  : Patient age (default 60 — dataset mean)
    - sex  : Male / Female / Unknown (default Unknown)

    If an identical (image, age, sex) combination was already analyzed
    under the current model version, the cached result is returned
    immediately — no re-run of the ensemble or the Groq LLM call, and no
    daily-cap deduction.

    Rate limits:
    - 5 requests per minute per IP address
    - Global daily cap shared across all users (protects LLM API costs) —
      only counted against on an actual (non-cached) inference run

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

    # ── Read upload with an explicit size cap ─────────────────
    contents = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)  # 1MB at a time
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum upload size is "
                       f"{MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
            )
    contents = bytes(contents)

    # ── Save uploaded file ────────────────────────────────────
    file_id   = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(save_path, 'wb') as f:
        f.write(contents)

    # ── Validate X-ray (skip for DICOM — already medical format) ──
    img_array_for_hash = None
    if ext != '.dcm':
        try:
            img       = Image.open(save_path).convert('RGB')
            img_array = np.array(img)
            is_valid, message = is_valid_xray(img_array)

            if not is_valid:
                os.remove(save_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image: {message}"
                )
            img_array_for_hash = img_array
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process image: {str(e)}"
            )

    # ── Dedup cache check ──────────────────────────────────────
    # Hash based on decoded pixels (or raw DICOM bytes as a fallback), so
    # the same X-ray re-uploaded doesn't re-run the ensemble or burn a
    # Groq call. Cache key includes age/sex/model_version since the
    # prediction genuinely depends on all three.
    cache_hit = None
    try:
        if img_array_for_hash is not None:
            image_hash = db.hash_pixels(img_array_for_hash)
        else:
            image_hash = db.hash_pixels(np.frombuffer(contents, dtype=np.uint8))

        cache_hit = db.find_cached_scan(image_hash, age, sex, MODEL_VERSION)
    except Exception as e:
        # DB being unavailable should degrade gracefully to "always run
        # inference", not break the whole endpoint.
        print(f"[predict] Cache lookup failed (continuing without cache): {e}")

    if cache_hit:
        return JSONResponse({
            'scan_id':     file_id,
            'filename':    file.filename,
            'age':         age,
            'sex':         sex,
            'predictions': cache_hit['predictions'],
            'heatmaps':    {},  # heatmaps aren't cached (image-derived, regenerable on demand)
            'original':    None,
            'report':      cache_hit['report'],
            'daily_requests_remaining': daily_counter.remaining(),
            'from_cache':  True,
            'scan_db_id':  cache_hit['scan_id'],
        })

    # ── Global daily cap (protects Groq API costs) ────────────
    # Only enforced for actual inference runs, not cache hits.
    if not daily_counter.increment_and_check():
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily request limit reached ({daily_counter.daily_limit} requests). "
                f"This limit resets at midnight UTC. This protects the shared LLM API "
                f"budget for this research demo — thanks for your patience."
            )
        )

    # ── Run inference ─────────────────────────────────────────
    from app.main import model_service
    if model_service is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet. Please try again.')

    try:
        results = model_service.run_inference(save_path, age=age, sex=sex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # ── Persist to DB (audit log + future cache hits) ─────────
    scan_db_id = None
    try:
        client_ip = request.client.host if request.client else None
        scan_db_id = db.save_scan(
            image_hash=image_hash,
            age=age, sex=sex,
            model_version=MODEL_VERSION,
            predictions=results['predictions'],
            report=results['report'],
            requester_ip=client_ip,
        )
    except Exception as e:
        # Don't fail the whole request just because logging failed.
        print(f"[predict] Failed to save scan to DB: {e}")

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
        'daily_requests_remaining': daily_counter.remaining(),
        'from_cache':  False,
        'scan_db_id':  scan_db_id,
    })