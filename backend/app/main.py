import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, report
from app.services.model_service import ModelService
from app.config import CHECKPOINT

# ── Global model instance ─────────────────────────────────
model_service: ModelService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release at shutdown."""
    global model_service
    print("Loading ChestVision model...")
    model_service = ModelService(checkpoint_path=CHECKPOINT)
    print("Model loaded and ready.")
    yield
    print("Shutting down...")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title='ChestVision AI API',
    description='Chest X-Ray Disease Detection with Explainable AI',
    version='1.0.0',
    lifespan=lifespan
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:3000',
        os.getenv('FRONTEND_URL', 'https://your-frontend.vercel.app')
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Routers ───────────────────────────────────────────────
app.include_router(predict.router, prefix='/api', tags=['Prediction'])
app.include_router(report.router,  prefix='/api', tags=['Report'])


@app.get('/health')
def health():
    return {
        'status':       'ok',
        'model_loaded': model_service is not None
    }


@app.get('/')
def root():
    return {
        'message': 'ChestVision AI API',
        'docs':    '/docs'
    }