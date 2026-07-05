import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, report
from app.services.model_service import ModelService
from app.config import CHECKPOINT, MOBILENET_CHECKPOINT

from dotenv import load_dotenv

# ── .env loading — portable across Windows dev and Linux/Docker ──
if os.name == 'nt':
    load_dotenv(r'D:\Projects\ChestVision-AI\.env')
else:
    load_dotenv()

# ── Global model instance ─────────────────────────────────
model_service: ModelService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load 3-model ensemble once at startup, release at shutdown."""
    global model_service
    print("Loading ChestVision ensemble (EfficientNet-B0 + MobileNetV2 + TorchXRayVision)...")
    model_service = ModelService(
        efficientnet_checkpoint=CHECKPOINT,
        mobilenet_checkpoint=MOBILENET_CHECKPOINT
    )
    print("Ensemble loaded and ready.")
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
        'https://chest-vision-ai.vercel.app',
        'https://chest-vision-ai-git-main-advait-gujar.vercel.app',
        os.getenv('FRONTEND_URL', '*')
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