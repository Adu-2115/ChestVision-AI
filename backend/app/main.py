import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.rate_limit import limiter, daily_counter
from app.routers import predict, report, feedback
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

# ── Rate limiting ─────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────
# IMPORTANT: never include '*' here. allow_credentials=True + '*' in the
# origin list means Starlette echoes back whatever Origin header the
# caller sends, which effectively allows any website to make credentialed
# requests to this API — that defeats the purpose of an allowlist entirely.
# Previously this list fell back to os.getenv('FRONTEND_URL', '*'), which
# silently included '*' whenever FRONTEND_URL wasn't set as an env var
# (true on the HF Space, since it was never explicitly configured there).
_allowed_origins = [
    'http://localhost:3000',
    'https://chest-vision-ai.vercel.app',
    'https://chest-vision-ai-git-main-advait-gujar.vercel.app',
]
_extra_origin = os.getenv('FRONTEND_URL')
if _extra_origin and _extra_origin != '*':
    _allowed_origins.append(_extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Routers ───────────────────────────────────────────────
app.include_router(predict.router, prefix='/api', tags=['Prediction'])
app.include_router(report.router,  prefix='/api', tags=['Report'])
app.include_router(feedback.router, prefix='/api', tags=['Feedback'])


@app.get('/health')
def health():
    return {
        'status':       'ok',
        'model_loaded': model_service is not None,
        'daily_requests_remaining': daily_counter.remaining(),
    }


@app.get('/')
def root():
    return {
        'message': 'ChestVision AI API',
        'docs':    '/docs'
    }