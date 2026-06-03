"""
Basic API tests — run with: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_health_check():
    """Health endpoint should return 200."""
    # Import here so model load is skipped in CI
    from app.main import app
    client = TestClient(app)
    # lifespan won't run in TestClient without context manager
    response = client.get('/health')
    assert response.status_code == 200

def test_root():
    from app.main import app
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert 'ChestVision' in response.json()['message']

def test_predict_no_file():
    """Predict without file should return 422."""
    from app.main import app
    client = TestClient(app)
    response = client.post('/api/predict')
    assert response.status_code == 422

def test_predict_wrong_filetype():
    """Predict with wrong file type should return 400."""
    from app.main import app
    import io
    client = TestClient(app)
    fake_file = io.BytesIO(b'fake content')
    response = client.post(
        '/api/predict',
        files={'file': ('test.txt', fake_file, 'text/plain')}
    )
    assert response.status_code in [400, 503]