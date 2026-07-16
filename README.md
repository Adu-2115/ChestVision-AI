# ChestVision AI 🩻

<div align="center">

**An explainable, AI-powered chest X-ray disease detection and clinical reporting platform**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3_70B-F55036?style=for-the-badge)](https://groq.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Spaces-Backend-yellow?style=for-the-badge)](https://huggingface.co/spaces/Adu2115/chessvision-api)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Live Demo](https://chest-vision-ai.vercel.app) · [API Docs](https://adu2115-chessvision-api.hf.space/docs) · [GitHub](https://github.com/Adu-2115/ChestVision-AI)

</div>

---

## Overview

ChestVision AI is a research-grade clinical decision-support platform that analyzes chest X-ray images to simultaneously detect multiple thoracic diseases. The system is built on four pillars:

- **3-Model Ensemble** — EfficientNet-B0 and MobileNetV2 (both trained on CheXpert with age/sex fusion), averaged with TorchXRayVision's pretrained CheXpert model for a more robust, variance-reduced prediction, with all three run concurrently for lower latency
- **Confidence Calibration** — Temperature scaling applied post-training to both custom models so reported probabilities better reflect true likelihood, not just raw model confidence
- **Explainable AI** — Grad-CAM heatmaps mapped to 7 clinical anatomical zones, visually highlighting the lung regions driving each prediction
- **LLM-Powered Reports** — LLaMA3-70B (via Groq) generates detailed, clinician-style radiology reports with differential diagnoses, urgency assessment, and spatial grounding from Grad-CAM data

> **Medical Disclaimer:** This is a research and decision-support tool. It is not a substitute for professional radiological diagnosis and has not been approved for clinical use.

---

## Key Features

| Feature | Description |
|---|---|
| 3-Model Ensemble | EfficientNet-B0 + MobileNetV2 + TorchXRayVision averaged per-disease, run concurrently, with per-model score breakdown and inter-model agreement/disagreement surfaced in the UI |
| Multi-label Detection | Simultaneously detects 5 thoracic conditions from a single X-ray |
| Confidence Calibration | Temperature scaling post-training, applied at inference via `forward_calibrated()` |
| Demographic Fusion | Patient age and sex fused into both custom models' predictions |
| X-Ray Physics Validation | 5 radiological sanity checks (contrast, dynamic range, histogram spread, dimensions) reject non-X-ray uploads before inference |
| Grad-CAM Explainability | Interactive heatmaps mapped to 7 anatomical zones (upper/mid/lower left & right lung, central mediastinum) |
| LLM Clinical Reports | LLaMA3-70B generates findings, differential diagnosis, and urgency assessment, grounded in Grad-CAM spatial data |
| DICOM Support | Accepts `.dcm` files alongside JPG/PNG, with VOI LUT and photometric interpretation handling |
| PDF Export | Professionally formatted downloadable radiology reports |
| Disease Knowledge Panel | Symptoms, causes, and specialist referrals for each detected condition |
| Rate Limiting | Per-IP (5 req/min) and global daily request caps to protect shared LLM API budget |
| Network Hardening | Strict CORS allowlist (no wildcard), upload size cap, and LLM call timeout |
| REST API | Fully documented FastAPI backend with Swagger UI |

---

## System Architecture

```
+-------------------------------------------------------------+
|                      React Frontend                          |
|    Upload · Predictions · Heatmaps · Reports · Disease Info |
+------------------------+------------------------------------+
                         | REST API (rate limited, CORS restricted)
+------------------------v------------------------------------+
|                     FastAPI Backend                          |
|                                                              |
|  +-------------+   +------------------------------------+  |
|  |   Image     |   |     3-Model Ensemble (concurrent)   |  |
|  | Preprocess  +-->|  EfficientNet-B0  (calibrated)      |  |
|  | + Physics   |   |  MobileNetV2      (calibrated)      |  |
|  |  Validation |   |  TorchXRayVision  (pretrained)      |  |
|  +-------------+   +--------------------+-----------------+  |
|                            | averaged per-disease           |
|                     Disease Scores + Agreement               |
|                            |                                 |
|                     +------v-------+   +----------------+   |
|                     |  Grad-CAM    |   |  LLaMA3-70B    |   |
|                     |  (Eff-Net-B0)+-->|  Report Gen    |   |
|                     +--------------+   +----------------+   |
+-------------------------------------------------------------+
                         |
+------------------------v------------------------------------+
|                    Infrastructure                            |
|      Docker · Hugging Face Spaces · Vercel · HF Model Hub    |
+-------------------------------------------------------------+
```

---

## Model Performance

Individual val AUC per model, trained on CheXpert-v1.0-small (223k images, 20 epochs, NVIDIA RTX 3050 6GB):

### EfficientNet-B0 (multimodal, calibrated)
58MB model size, ~350MB RAM. Temperature: **1.0786**

| Disease | Val AUC |
|---|---|
| Atelectasis | 0.712 |
| Cardiomegaly | 0.869 |
| Consolidation | 0.734 |
| Edema | 0.850 |
| Pleural Effusion | 0.868 |
| **Mean AUC** | **0.807** |

### MobileNetV2 (multimodal, calibrated)
~10MB smaller than EfficientNet-B0, added for architectural diversity in the ensemble. Temperature: **1.0535**

| Disease | Val AUC |
|---|---|
| Atelectasis | 0.698 |
| Cardiomegaly | 0.864 |
| Consolidation | 0.726 |
| Edema | 0.839 |
| Pleural Effusion | 0.864 |
| **Mean AUC** | **0.798** |

### TorchXRayVision DenseNet121 (pretrained, image-only)
Externally pretrained on CheXpert/NIH/MIMIC/PadChest — no training required, added as a third independent opinion. Uses its own normalization pipeline (not ImageNet stats) and does not receive age/sex input.

### Ensemble aggregation
Final probability per disease is an **equal-weight average** across whichever models cover that disease (TorchXRayVision's coverage is verified at runtime, not assumed). All three models run **concurrently** via a thread pool rather than sequentially, since PyTorch releases the GIL during tensor computation — this meaningfully cuts per-request latency on CPU. Per-model scores and an inter-model **disagreement metric** (max − min score) are surfaced in the API response and UI — high disagreement signals a case worth extra clinical scrutiny rather than being silently averaged away.

---

## AI-Powered Report Generation

Reports are generated by LLaMA3-70B (via Groq API) acting as a senior radiologist, grounded in Grad-CAM spatial activation data (7 anatomical zones) rather than generic disease descriptions. Each report includes:

- **Findings** — Detailed radiological observations per disease, referencing the specific anatomical zones Grad-CAM activated and their laterality (bilateral / left-sided / right-sided)
- **Differential Diagnosis** — 3-4 possible underlying conditions with clinical reasoning tied to the spatial distribution and patient demographics
- **Impression** — Overall clinical picture, significance, and most urgent next step
- **Recommendations** — Urgency level, specialist referrals, and follow-up investigations appropriate to patient age/sex

Example impression for a Pleural Effusion + Atelectasis + Edema finding:

> *"Findings are suggestive of heart failure, given the combination of pleural effusion, atelectasis, and edema. Urgent — refer to Cardiologist and Pulmonologist; recommend echocardiogram and BNP."*

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Models | PyTorch 2.5, EfficientNet-B0, MobileNetV2, TorchXRayVision | 3-model disease classification ensemble, run concurrently |
| Calibration | Temperature scaling (LBFGS post-training) | Confidence calibration on both custom models |
| Explainability | pytorch-grad-cam | Heatmap generation mapped to 7 anatomical zones |
| LLM | Groq API, LLaMA3-70B | Clinical report generation with spatial grounding |
| Backend | FastAPI, Uvicorn, slowapi | REST API server with rate limiting and CORS hardening |
| PDF Reports | ReportLab | Report export |
| Frontend | React 18, TypeScript, TailwindCSS | User interface with per-model consensus indicators |
| Dataset | CheXpert v1.0 (Stanford) | Training data |
| Containerization | Docker (CPU-only PyTorch build) | Deployment packaging |
| Hosting | Hugging Face Spaces (backend), Vercel (frontend) | Cloud deployment |
| Model Storage | Hugging Face Hub | Calibrated checkpoint hosting, pulled at Docker build time |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (recommended for training; inference runs on CPU in production)
- Groq API key (free at console.groq.com)

### 1. Clone and set up environment

```bash
git clone https://github.com/Adu-2115/ChestVision-AI.git
cd ChestVision-AI

conda create -n chessvision python=3.10
conda activate chessvision

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
CHECKPOINT_PATH=D:/Projects/ChestVision-AI/checkpoints_efficientnet/best_model_calibrated.pth
MOBILENET_CHECKPOINT_PATH=D:/Projects/ChestVision-AI/checkpoints_mobilenet/best_model_calibrated.pth
DAILY_REQUEST_LIMIT=200
FRONTEND_URL=https://chest-vision-ai.vercel.app
```

### 3. Start backend

```bash
$env:PYTHONPATH = "D:\Projects\ChestVision-AI"
cd backend
uvicorn app.main:app --reload --port 8000
```

### 4. Start frontend

```bash
cd frontend
npm install
npm start
```

Visit http://localhost:3000

API documentation at http://localhost:8000/docs

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — model load status, daily requests remaining |
| `POST` | `/api/predict` | Analyze X-ray via 3-model ensemble — rate limited to 5/min per IP, 15MB upload cap |
| `POST` | `/api/report/generate` | Generate PDF from report data |

### Sample Response

```json
{
  "scan_id": "8f6d05b2-...",
  "predictions": [
    {
      "disease": "Edema",
      "probability": 0.79,
      "positive": true,
      "model_scores": {
        "efficientnet_b0": 0.81,
        "mobilenet_v2": 0.76,
        "torchxrayvision": 0.80
      },
      "disagreement": 0.05,
      "n_models_used": 3
    }
  ],
  "heatmaps": {"Edema": "<base64_png>"},
  "report": {
    "llm_generated": true,
    "findings": "The chest X-ray demonstrates...",
    "impression": "Findings are suggestive of...",
    "recommendations": ["Urgency: Urgent", "Refer to Cardiologist..."]
  },
  "daily_requests_remaining": 187
}
```

---

## Project Structure

```
ChestVision-AI/
├── src/                          # Core ML pipeline
│   ├── dataset.py                # CheXpert data loader + physics-based augmentation
│   ├── train.py                  # EfficientNet-B0 training loop
│   ├── train_mobilenet.py        # MobileNetV2 training loop
│   ├── gradcam.py                # Grad-CAM + 7-zone spatial description
│   ├── report.py                 # LLM + rule-based report generation (Groq call, 30s timeout)
│   └── models/
│       ├── densenet.py           # EfficientNet-B0 + temperature scaling
│       ├── mobilenet.py          # MobileNetV2 + temperature scaling
│       ├── xrv_wrapper.py        # TorchXRayVision pretrained model wrapper
│       └── ensemble.py           # Concurrent 3-model averaging + disagreement scoring
├── backend/                      # FastAPI application
│   ├── app/
│   │   ├── main.py               # App entry, CORS allowlist, lifespan, rate limit registration
│   │   ├── config.py             # Environment-aware configuration (both checkpoints)
│   │   ├── rate_limit.py         # Per-IP + daily request limiting
│   │   ├── routers/
│   │   │   ├── predict.py        # /api/predict endpoint (chunked upload, size-capped)
│   │   │   └── report.py         # /api/report endpoints + PDF generation
│   │   └── services/
│   │       └── model_service.py  # Ensemble loading + inference + Grad-CAM pipeline
│   ├── Dockerfile                # CPU-only torch install
│   └── requirements.txt
├── frontend/                     # React application
│   └── src/
│       └── App.tsx               # Main UI — predictions, per-model breakdown, heatmaps, reports
├── notebooks/                    # Training, calibration, and exploration notebooks
└── checkpoints_efficientnet/ · checkpoints_mobilenet/   # Local training outputs (gitignored)
```

Note: `chessvision-api` is a separate, leaner deployment repo containing only `src/` and `app/` (no notebooks or frontend) — its git remote points directly at the Hugging Face Space, and pushing to it triggers an automatic rebuild. Its own `README.md` requires HF Spaces' YAML config frontmatter and is kept separate from this one.

---

## Deployment

The backend deploys to Hugging Face Spaces via direct git push (the `chessvision-api` repo's git remote *is* the Space); the frontend auto-deploys to Vercel on push to `ChestVision-AI`'s `main` branch.

```
Push to chessvision-api (main)
        |
        v
Hugging Face Spaces rebuild
        |-- Install CPU-only torch/torchvision (avoids unused CUDA libs)
        |-- Install remaining requirements (incl. torchxrayvision, slowapi)
        |-- Download both calibrated checkpoints from HF Model Hub
        |-- Pre-bake TorchXRayVision weights at build time
        |-- Pre-bake MobileCLIP-S1 weights at build time (OOD detection)
        +-- Start Uvicorn with 3-model ensemble + CLIP OOD checker loaded

Push to ChestVision-AI (main)
        |
        v
Vercel auto-deploy (frontend/ only)
```

**Live URLs:**
- Frontend: https://chest-vision-ai.vercel.app
- Backend API: https://adu2115-chessvision-api.hf.space
- API Docs: https://adu2115-chessvision-api.hf.space/docs

---

## Roadmap

- [x] Multi-label chest X-ray classification
- [x] Grad-CAM explainability with 7-zone anatomical mapping
- [x] LLM-powered clinical report generation, grounded in spatial data
- [x] PDF report export
- [x] Docker containerization
- [x] X-ray physics validation (rejects non-X-ray uploads)
- [x] Age and sex auxiliary inputs for both custom models
- [x] Confidence calibration via temperature scaling
- [x] DICOM file format support
- [x] 3-model ensemble (EfficientNet-B0 + MobileNetV2 + TorchXRayVision)
- [x] Per-model score breakdown + inter-model disagreement in API/UI
- [x] Rate limiting (per-IP + daily global cap)
- [x] Parallelize ensemble inference to reduce per-request latency
- [x] Network hardening (CORS allowlist, upload size cap, LLM call timeout)
- [x] Binary X-ray classifier for proper out-of-distribution detection (CLIP zero-shot check layered on top of physics validation)
- [ ] Symptom-based disease pre-screening module
- [ ] Uncertainty quantification beyond inter-model disagreement
- [ ] User feedback / correction mechanism
- [ ] Audit logging
- [ ] Expansion to 14 diseases on full CheXpert dataset
- [ ] Multi-view (frontal + lateral) support
- [ ] Active learning pipeline for continuous improvement

---

## Disclaimer

ChestVision AI is an experimental research tool built for educational and portfolio demonstration purposes.

- Not approved for clinical use
- Not a substitute for professional radiological diagnosis
- Not validated on external clinical datasets
- All outputs must be verified by a qualified radiologist or physician

---

## Acknowledgements

- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert) — Stanford ML Group
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) — Cohen et al.
- [Groq](https://groq.com) — Ultra-fast LLM inference

---

<div align="center">
Built by Advait Gujar
</div>