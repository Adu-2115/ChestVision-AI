"""
Save this as: backend/tests/conftest.py

Ensures `src` and `app` are importable when running pytest, matching the
same PYTHONPATH setup used at runtime (PYTHONPATH=/app in Docker,
$env:PYTHONPATH = "D:\\Projects\\ChestVision-AI" locally on Windows).
"""
import os
import sys

# Walk up from this file to find the project root (parent of backend/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

for _path in (_PROJECT_ROOT, _BACKEND_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
