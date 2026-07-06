"""
Tests for is_valid_xray() — the physics-based sanity checks that reject
non-X-ray uploads before they hit the (expensive) ensemble.

Save this as: backend/tests/test_xray_validation.py

Note: importing app.routers.predict has the side effect of creating the
UPLOAD_DIR directory (os.makedirs with exist_ok=True) — harmless in a
test environment, just creates a local folder if run outside Docker.
"""
import numpy as np
import pytest
from app.routers.predict import is_valid_xray


def make_fake_xray(size=256):
    """
    Builds a synthetic array with X-ray-like statistical properties:
    grayscale (R=G=B), high contrast, dark lung fields + bright bone
    regions, spread histogram.
    """
    arr = np.random.randint(40, 220, size=(size, size), dtype=np.uint8)
    # Force some genuinely dark and genuinely bright regions
    arr[0:size//3, :] = np.random.randint(0, 60, size=(size//3, size))       # dark "lung" band
    arr[2*size//3:, :] = np.random.randint(190, 255, size=(size - 2*size//3, size))  # bright "bone" band
    rgb = np.stack([arr, arr, arr], axis=-1)  # grayscale replicated across channels
    return rgb


def test_valid_xray_passes():
    img = make_fake_xray()
    is_valid, message = is_valid_xray(img)
    assert is_valid is True


def test_color_photo_rejected():
    """A real color photo has very different R/G/B channels — should fail
    the grayscale check."""
    size = 256
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # strong red channel
    img[:, :, 1] = 50
    img[:, :, 2] = 50
    is_valid, message = is_valid_xray(img)
    assert is_valid is False
    assert 'color photograph' in message.lower()


def test_flat_low_contrast_image_rejected():
    """A blank/flat image has near-zero standard deviation."""
    size = 256
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    is_valid, message = is_valid_xray(img)
    assert is_valid is False
    assert 'contrast' in message.lower()


def test_too_small_image_rejected():
    img = make_fake_xray(size=50)
    is_valid, message = is_valid_xray(img)
    assert is_valid is False
    assert 'too small' in message.lower()


def test_missing_dark_regions_rejected():
    """An image that's uniformly mid-bright with no dark lung-field
    region shouldn't pass, even if it has some contrast."""
    size = 256
    arr = np.random.randint(100, 200, size=(size, size), dtype=np.uint8)
    img = np.stack([arr, arr, arr], axis=-1)
    is_valid, message = is_valid_xray(img)
    assert is_valid is False
