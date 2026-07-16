FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface-hub

# Download model from Hugging Face
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import shutil, os; \
path = hf_hub_download(repo_id='Adu2115/chessvision-densenet121', filename='best_model.pth'); \
os.makedirs('/app/checkpoints', exist_ok=True); \
shutil.copy(path, '/app/checkpoints/best_model.pth'); \
print('Model downloaded successfully')"

# Pre-download MobileCLIP-S1 weights (OOD detection) at build time —
# mirrors the same pre-bake step in chessvision-api's Dockerfile (the
# repo that actually deploys). Note: unlike chessvision-api, this
# Dockerfile does not pin a CPU-only torch build before installing
# requirements.txt — see chessvision-api/Dockerfile for why that matters.
RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr'); \
print('MobileCLIP-S1 weights cached')"

# Copy source modules
COPY src/ /app/src/
COPY backend/ /app/backend/

# Set working directory to backend
WORKDIR /app/backend

RUN mkdir -p /app/uploads /app/reports

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]