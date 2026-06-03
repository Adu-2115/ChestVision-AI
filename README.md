# ChestVision AI 🩻

> Explainable multi-label chest X-ray disease detection using DenseNet-121, Grad-CAM visualization, and automated clinical report generation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-18-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

ChestVision AI is an end-to-end clinical decision-support platform that analyzes chest X-ray images to detect multiple thoracic diseases simultaneously. The system combines deep learning classification with explainable AI techniques, providing radiologists and clinicians with both predictions and visual explanations of model reasoning.

**This is not a replacement for radiologists.** It is a decision-support tool designed to assist in screening and improve diagnostic consistency.

---

## 🏗️ System Architecture

```
User uploads X-Ray
        │
        ▼
  Image Preprocessing
  (resize, normalize, augment)
        │
        ▼
  DenseNet-121 Model
  (pretrained on ImageNet, fine-tuned on CheXpert)
        │
   ┌────┴────┐
   ▼         ▼
Disease    Grad-CAM
Scores     Heatmaps
   │         │
   └────┬────┘
        ▼
  Report Generator
  (findings, impression, recommendations)
        ▼
  FastAPI Backend
        ▼
  React Frontend
  (predictions, heatmaps, PDF report)
```

---

## ✨ Features

- **Multi-label disease detection** — detects 5 thoracic conditions simultaneously from a single X-ray
- **Confidence scores** — probability for each disease with positive/negative classification
- **Grad-CAM heatmaps** — visual explanation of which lung regions influenced each prediction
- **Automated clinical report** — structured radiology-style report with findings, impression, and recommendations
- **PDF download** — professionally formatted report downloadable as PDF
- **Disease knowledge panel** — symptoms, causes, and recommended specialists for each detected condition
- **REST API** — fully documented FastAPI backend with Swagger UI

---

## 🎯 Detected Diseases

| Disease | Description | Val AUC |
|---|---|---|
| Atelectasis | Partial/complete lung collapse | 0.716 |
| Cardiomegaly | Enlarged cardiac silhouette | 0.870 |
| Consolidation | Airspace opacification | 0.735 |
| Edema | Fluid in lung interstitium | 0.851 |
| Pleural Effusion | Fluid in pleural space | 0.872 |
| **Mean** | | **0.809** |

---

## 🧠 How It Works

### 1. Data Preparation
- Dataset: [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert) (Stanford) — 223k chest X-rays
- Frontal views only (AP/PA)
- Uncertain labels handled via U-Ones strategy (Atelectasis, Edema → positive)
- Train/val split: 162k / 29k samples
- Augmentation: horizontal flip, brightness/contrast, affine transforms, CLAHE

### 2. Model Training
- Base model: DenseNet-121 pretrained on ImageNet
- Custom classification head: Linear(1024→512) → ReLU → Dropout → Linear(512→5)
- Loss: BCEWithLogitsLoss with class weights (handles severe imbalance)
- Optimizer: Adam with cosine annealing scheduler
- Mixed precision training (float16) for memory efficiency
- Strategy: freeze backbone for 2 epochs, then unfreeze full model
- Best model checkpoint saved based on mean validation AUC

### 3. Explainability (Grad-CAM)
- Target layer: DenseNet final dense block (`denseblock4`)
- Generates per-class activation maps
- Heatmaps overlaid on original X-ray using jet colormap
- Shows which anatomical regions drove each prediction

### 4. Report Generation
- Rule-based template engine with disease-specific clinical text
- Structured sections: Findings → Impression → Recommendations
- Disease knowledge base: symptoms, causes, specialist referral
- PDF generation via ReportLab

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch 2.5, DenseNet-121 |
| Explainability | pytorch-grad-cam |
| Backend | FastAPI, Uvicorn |
| Report PDF | ReportLab |
| Frontend | React 18, TypeScript, TailwindCSS |
| Dataset | CheXpert v1.0 (Stanford) |
| Deployment | Docker, Render, Vercel |
| CI/CD | GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- Node.js 18
- CUDA-compatible GPU (recommended)

### Backend Setup

```bash
# Clone repo
git clone https://github.com/Adu-2115/ChestVision-AI.git
cd ChestVision-AI

# Create environment
conda create -n chessvision python=3.10
conda activate chessvision

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt

# Set environment variables
export CHECKPOINT_PATH=/path/to/best_model.pth

# Start backend
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

Visit `http://localhost:3000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/predict` | Upload X-ray, get predictions + heatmaps + report |
| `POST` | `/api/report/generate` | Generate PDF from report data |

### Example Request
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

### Example Response
```json
{
  "scan_id": "uuid",
  "predictions": [
    {"disease": "Edema", "probability": 0.79, "positive": true},
    {"disease": "Cardiomegaly", "probability": 0.58, "positive": true}
  ],
  "heatmaps": {"Edema": "base64_png..."},
  "report": {
    "findings": "...",
    "impression": "...",
    "recommendations": ["..."]
  }
}
```

---

## 📁 Project Structure

```
ChestVision-AI/
├── src/                        # Core ML modules
│   ├── dataset.py              # CheXpert data loader
│   ├── train.py                # Training pipeline
│   ├── gradcam.py              # Grad-CAM visualization
│   ├── report.py               # Report generation
│   └── models/
│       └── densenet.py         # Model architecture
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/
│   │   │   ├── predict.py
│   │   │   └── report.py
│   │   └── services/
│   │       └── model_service.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # React application
│   └── src/
│       └── App.tsx
├── notebooks/                  # Exploration & training notebooks
└── .github/workflows/          # CI/CD pipeline
    └── deploy.yml
```

---

## 🔬 Training Results

Training was performed on CheXpert-v1.0-small (223k images) for 20 epochs on NVIDIA RTX 3050 6GB.

- Best validation AUC: **0.8088** (epoch 12)
- Training time: ~3 hours
- Mixed precision (FP16) enabled

---

## ⚠️ Disclaimer

**IMPORTANT:** ChestVision AI is an experimental research tool built for educational and demonstration purposes. It is:

- **NOT** approved for clinical use
- **NOT** a substitute for professional radiological diagnosis
- **NOT** validated on external clinical datasets

All outputs must be verified by a qualified radiologist or physician before any clinical decision is made.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert) — Stanford ML Group
- [CheXNet Paper](https://arxiv.org/abs/1711.05225) — Rajpurkar et al.
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat