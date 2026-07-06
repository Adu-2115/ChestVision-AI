import os
import uuid
import base64
import logging
import asyncio
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import UPLOAD_DIR, MODEL_VERSION
from app.rate_limit import limiter, daily_counter
from app import db

router = APIRouter()
logger = logging.getLogger("chestvision.predict")

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

# ── Decompression bomb protection ─────────────────────────
# Pillow already refuses to decode images above ~178 megapixels by
# default, but chest X-rays are never anywhere close to that size —
# tightening this to 50MP means a maliciously crafted "small file that
# decodes into a huge image" gets rejected quickly and cleanly, rather
# than relying on the library default as the only line of defense.
Image.MAX_IMAGE_PIXELS = 50_000_000

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Magic-byte signatures ──────────────────────────────────
# Checking actual file content, not just the claimed extension — someone
# can rename any file to .jpg and it would previously sail past the
# extension check before ever being opened.
_MAGIC_BYTES = {
    '.jpg':  [b'\xff\xd8\xff'],
    '.jpeg': [b'\xff\xd8\xff'],
    '.png':  [b'\x89PNG\r\n\x1a\n'],
    '.dcm':  None,  # verified separately below (DICOM preamble check)
}


def verify_file_signature(contents: bytes, ext: str) -> bool:
    """Returns True if the file's actual bytes match what its extension claims."""
    if ext == '.dcm':
        # Standard DICOM files have a 128-byte preamble followed by the
        # 4-byte magic string 'DICM' at offset 128. Not universally
        # required by the DICOM spec, but true for the vast majority of
        # real-world files and a solid quick check.
        return len(contents) > 132 and contents[128:132] == b'DICM'

    signatures = _MAGIC_BYTES.get(ext)
    if not signatures:
        return False
    return any(contents.startswith(sig) for sig in signatures)


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

    Rate limits:
    - 5 requests per minute per IP address
    - Global daily cap shared across all users (protects LLM API costs)

    Returns:
    - predictions, heatmaps, report, scan metadata
    """
    # ── Validate file type (extension) ────────────────────────
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

    # ── Read upload with an explicit size cap AND time limit ──
    # A deliberately slow upload (sending 1 byte every few seconds) would
    # otherwise hold this connection — and the worker handling it — open
    # indefinitely. 60s is generous for a 15MB file on any real connection.
    UPLOAD_READ_TIMEOUT_SECONDS = 60

    async def _read_upload():
        buf = bytearray()
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB at a time
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum upload size is "
                           f"{MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
                )
        return bytes(buf)

    try:
        contents = await asyncio.wait_for(_read_upload(), timeout=UPLOAD_READ_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Upload took too long. Please check your connection and try again."
        )

    # ── Verify actual file content matches claimed extension ──
    # Rejects a renamed arbitrary file before it's ever saved to disk or
    # opened by PIL/pydicom.
    if not verify_file_signature(contents, ext):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match its extension ({ext}). "
                   f"The file may be corrupted or mislabeled."
        )

    # ── Save uploaded file ────────────────────────────────────
    file_id   = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(save_path, 'wb') as f:
        f.write(contents)

    try:
        # ── Validate X-ray (skip for DICOM — already medical format) ──
        img_array_for_hash = None
        if ext != '.dcm':
            try:
                img       = Image.open(save_path).convert('RGB')
                img_array = np.array(img)
                is_valid, message = is_valid_xray(img_array)

                if not is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid image: {message}"
                    )
                img_array_for_hash = img_array
            except HTTPException:
                raise
            except Image.DecompressionBombError:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are unreasonably large for a chest X-ray "
                           "and were rejected as a safety precaution."
                )
            except Exception as e:
                logger.warning(f"Image processing failed for upload {file_id}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail="Could not process this image. Please ensure it's a valid "
                           "JPG or PNG file."
                )

        # ── Dedup cache check ──────────────────────────────────
        cache_hit = None
        try:
            if img_array_for_hash is not None:
                image_hash = db.hash_pixels(img_array_for_hash)
            else:
                image_hash = db.hash_pixels(np.frombuffer(contents, dtype=np.uint8))

            cache_hit = db.find_cached_scan(image_hash, age, sex, MODEL_VERSION)
        except Exception as e:
            logger.warning(f"Cache lookup failed (continuing without cache): {e}")

        if cache_hit:
            return JSONResponse({
                'scan_id':     file_id,
                'filename':    file.filename,
                'age':         age,
                'sex':         sex,
                'predictions': cache_hit['predictions'],
                'heatmaps':    {},
                'original':    None,
                'report':      cache_hit['report'],
                'daily_requests_remaining': daily_counter.remaining(),
                'from_cache':  True,
                'scan_db_id':  cache_hit['scan_id'],
            })

        # ── Global daily cap (protects Groq API costs) ────────
        if not daily_counter.increment_and_check():
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Daily request limit reached ({daily_counter.daily_limit} requests). "
                    f"This limit resets at midnight UTC. This protects the shared LLM API "
                    f"budget for this research demo — thanks for your patience."
                )
            )

        # ── Run inference ─────────────────────────────────────
        from app.main import model_service
        if model_service is None:
            raise HTTPException(status_code=503, detail='Model not loaded yet. Please try again.')

        try:
            results = model_service.run_inference(save_path, age=age, sex=sex)
        except Exception as e:
            # Full detail logged server-side only — the client never sees
            # internal paths, library tracebacks, or other implementation
            # details that could aid an attacker probing the system.
            logger.error(f"Inference failed for upload {file_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Analysis failed due to an internal error. Please try again, "
                       "and contact support if this persists."
            )

        # ── Persist to DB (audit log + future cache hits) ─────
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
            logger.warning(f"Failed to save scan to DB: {e}")

        # ── Convert heatmaps to base64 ─────────────────────────
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

    finally:
        # ── Always clean up the uploaded file ──────────────────
        # Whether inference succeeded, failed, or was rejected as invalid,
        # the raw uploaded image should never persist on disk indefinitely.
        # Predictions/report are what get retained (in the DB) — not the
        # original image itself.
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp upload {save_path}: {e}")
