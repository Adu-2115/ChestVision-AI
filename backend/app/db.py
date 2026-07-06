"""
Database layer for ChestVision AI — Supabase-hosted Postgres.

Three responsibilities:
1. Dedup cache: skip re-running the ensemble + Groq call for an identical
   (image, age, sex, model_version) combination already seen before.
2. Audit log: every scan performed is recorded (hashed IP, not raw).
3. Feedback storage: lets a reviewer mark a scan's prediction correct/incorrect.

Uses a thread-safe connection pool (SimpleConnectionPool) since the ensemble
now runs 3 models concurrently via ThreadPoolExecutor — multiple requests
may hit the DB at overlapping times.

Save this as: app/db.py
"""
import os
import json
import hashlib
from psycopg2 import pool
from psycopg2.extras import Json

DATABASE_URL = os.getenv('DATABASE_URL')

_pool = None


def get_pool():
    """Lazily create the connection pool on first use."""
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set — caching/audit logging will not work. "
                "Set it as an env var / HF Space secret."
            )
        _pool = pool.SimpleConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
    return _pool


def hash_pixels(img_array) -> str:
    """
    Hash based on decoded pixel content, not raw file bytes — so the same
    X-ray re-saved/re-exported by different software still matches even if
    file metadata or compression differs.
    """
    return hashlib.sha256(img_array.tobytes()).hexdigest()


def hash_ip(ip_address: str) -> str:
    """One-way hash of the requester's IP for audit purposes without
    storing the raw address."""
    return hashlib.sha256(ip_address.encode('utf-8')).hexdigest()


def find_cached_scan(image_hash: str, age: float, sex: str, model_version: str):
    """
    Returns the cached (predictions, report) dict pair if an identical
    scan was already processed under the current model version, else None.
    """
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, predictions, report
                FROM scans
                WHERE image_hash = %s AND age = %s AND sex = %s AND model_version = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (image_hash, age, sex, model_version)
            )
            row = cur.fetchone()
            if row is None:
                return None
            scan_id, predictions, report = row
            return {'scan_id': scan_id, 'predictions': predictions, 'report': report}
    finally:
        get_pool().putconn(conn)


def save_scan(image_hash: str, age: float, sex: str, model_version: str,
              predictions: list, report: dict, requester_ip: str = None) -> int:
    """Insert a new scan record. Returns the new row's id (used to link feedback)."""
    ip_hash = hash_ip(requester_ip) if requester_ip else None

    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scans (image_hash, age, sex, model_version,
                                    predictions, report, requester_ip_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (image_hash, age, sex, model_version,
                 Json(predictions), Json(report), ip_hash)
            )
            scan_id = cur.fetchone()[0]
            conn.commit()
            return scan_id
    finally:
        get_pool().putconn(conn)


def save_feedback(scan_id: int, is_correct: bool,
                   corrected_diagnosis: str = None, comments: str = None):
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (scan_id, is_correct, corrected_diagnosis, comments)
                VALUES (%s, %s, %s, %s)
                """,
                (scan_id, is_correct, corrected_diagnosis, comments)
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)
