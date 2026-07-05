import os

# ── Paths — work locally (Windows) and in Docker (Linux) ─
BASE_DIR              = os.getenv('BASE_DIR',        '/app')
CHECKPOINT            = os.getenv('CHECKPOINT_PATH', os.path.join(BASE_DIR, 'checkpoints', 'best_model_calibrated.pth'))
MOBILENET_CHECKPOINT  = os.getenv('MOBILENET_CHECKPOINT_PATH', os.path.join(BASE_DIR, 'checkpoints_mobilenet', 'best_model_calibrated.pth'))
UPLOAD_DIR            = os.getenv('UPLOAD_DIR',      os.path.join(BASE_DIR, 'uploads'))
REPORTS_DIR           = os.getenv('REPORTS_DIR',     os.path.join(BASE_DIR, 'reports'))
FRONTEND_URL          = os.getenv('FRONTEND_URL',    'http://localhost:3000')

# ── Local Windows override (dev only) ────────────────────
if os.name == 'nt':  # Windows
    _win_base            = r'D:\Projects\ChestVision-AI'
    CHECKPOINT           = os.getenv('CHECKPOINT_PATH',
                              os.path.join(_win_base, 'checkpoints_efficientnet', 'best_model_calibrated.pth'))
    MOBILENET_CHECKPOINT = os.getenv('MOBILENET_CHECKPOINT_PATH',
                              os.path.join(_win_base, 'checkpoints_mobilenet', 'best_model_calibrated.pth'))
    UPLOAD_DIR           = os.getenv('UPLOAD_DIR',
                              os.path.join(_win_base, 'backend', 'uploads'))
    REPORTS_DIR          = os.getenv('REPORTS_DIR',
                              os.path.join(_win_base, 'backend', 'reports'))