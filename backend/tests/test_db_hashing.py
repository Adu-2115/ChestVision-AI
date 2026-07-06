"""
Tests for the hashing functions in app/db.py — pure functions, no actual
database connection needed (get_pool() is only called by the DB-hitting
functions, not by hash_pixels/hash_ip).

Save this as: backend/tests/test_db_hashing.py
"""
import numpy as np
from app.db import hash_pixels, hash_ip


def test_identical_pixel_arrays_produce_identical_hash():
    arr = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    assert hash_pixels(arr) == hash_pixels(arr.copy())


def test_different_pixel_arrays_produce_different_hash():
    arr1 = np.zeros((224, 224, 3), dtype=np.uint8)
    arr2 = np.ones((224, 224, 3), dtype=np.uint8)
    assert hash_pixels(arr1) != hash_pixels(arr2)


def test_hash_is_deterministic_across_calls():
    arr = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    h1 = hash_pixels(arr)
    h2 = hash_pixels(arr)
    assert h1 == h2


def test_hash_ip_never_returns_raw_ip():
    ip = "203.0.113.42"
    hashed = hash_ip(ip)
    assert ip not in hashed
    assert len(hashed) == 64  # sha256 hex digest length


def test_hash_ip_deterministic():
    ip = "198.51.100.7"
    assert hash_ip(ip) == hash_ip(ip)


def test_hash_ip_different_ips_differ():
    assert hash_ip("10.0.0.1") != hash_ip("10.0.0.2")
