# ChestVision AI

Upload a chest X-ray and three models score it for five common findings. The result shows each score, a heatmap of where the model looked, where the models disagree, and a written summary.

**Research preview, not a diagnosis.** This is not a medical device and has not been validated for clinical use. Scores can be wrong. A qualified clinician must review any image and result.

- Site: https://chestvision.advaitgujar.com
- API: https://adu2115-chessvision-api.hf.space ([interactive docs](https://adu2115-chessvision-api.hf.space/docs))

## What it does

| Feature | Detail |
|---|---|
| Five conditions | Atelectasis, cardiomegaly, consolidation, edema, pleural effusion. Nothing else is assessed. |
| Three models | EfficientNet-B0 and MobileNetV2 (trained here on CheXpert, with age and sex as extra inputs) and TorchXRayVision's pretrained DenseNet121. They run concurrently and their scores are averaged per condition. |
| Disagreement shown | The response carries each model's score and the spread between them (highest minus lowest). The page flags a finding when the models disagree. |
| Heatmaps | Grad-CAM from EfficientNet-B0 for each flagged finding, described in terms of 7 chest zones. A heatmap shows what influenced the score; it is not an outline of disease. |
| Written summary | Drafted by a language model (`llama-3.3-70b-versatile` through Groq) from the scores and heatmap zones, with a rule-based template when the model is unavailable. It can contain mistakes. |
| File types | JPG, PNG and DICOM (`.dcm`), up to 15 MB. |
| Image checks | Uploads are checked for type, size and content, and a MobileCLIP-S1 check plus image-statistics checks reject images that do not look like a chest X-ray. |
| PDF | The written summary can be downloaded as a PDF. |
| Limits on use | 5 analyses per minute per address and a daily cap (200 by default) to protect the shared language-model quota. |

The uploaded image file is deleted from the server when the analysis finishes. Scores and the written summary may be stored when a database is configured.

## How accurate is it?

There are no trustworthy accuracy figures for the models currently deployed, so none are quoted here.

An earlier version of this README listed validation AUCs of about 0.80 for the two models trained here. A later audit of the data split found that most validation patients also appeared in the training set, which makes those numbers too optimistic. They have been removed. Retraining on a patient-level split with a held-out test set is under way; figures will be added here once they come from that test set.

Other limits to keep in mind:

- TorchXRayVision was itself trained on CheXpert, so it is not an independent check on CheXpert images.
- The models read a single front-view image and know nothing else about the person.
- Nothing has been validated on data from other hospitals or scanners.

## How it fits together

```
Browser (React, Vercel)
   |  upload image, age, sex
   v
FastAPI backend (Hugging Face Space, CPU)
   |-- validate the upload (type, size, content, "is this a chest X-ray?")
   |-- run three models concurrently and average the scores per condition
   |-- Grad-CAM heatmaps for flagged findings
   |-- written summary (language model, or template fallback)
   v
JSON: scores, per-model scores, disagreement, heatmaps, summary
```

## Repository layout

```
src/                     model and training code
  dataset.py             CheXpert loader and augmentation
  train.py               EfficientNet-B0 training
  train_mobilenet.py     MobileNetV2 training
  gradcam.py             Grad-CAM and the 7-zone description
  report.py              written summary (language model and rule-based)
  models/                the two trained models, the TorchXRayVision wrapper,
                         ensemble averaging, out-of-distribution check
backend/                 FastAPI application
  app/main.py            app entry, CORS, rate-limit registration
  app/routers/           predict, report (PDF), feedback
  app/services/          model loading and inference
  app/db.py              optional result cache and audit log (PostgreSQL)
  tests/                 pytest suite
frontend/                React + TypeScript + Tailwind
  src/App.tsx            screen switching (landing, upload, results)
  src/api.ts             API calls and response types
  src/components/        Landing, UploadScreen, Results, Viewer, FindingRow,
                         ConditionDrawer, Feedback, Header, shared UI parts
Dockerfile               backend image
```

The backend is deployed from a separate, smaller repository (`chessvision-api`) whose git remote is the Hugging Face Space. The frontend deploys to Vercel from this repository's `main` branch.

## Run it locally

You need Python 3.10+, Node.js 18+, and the two trained checkpoints. A Groq API key is optional; without it the written summary uses the template.

### Backend

```bash
git clone https://github.com/Adu-2115/ChestVision-AI.git
cd ChestVision-AI
pip install -r backend/requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here
CHECKPOINT_PATH=/path/to/checkpoints_efficientnet/best_model_calibrated.pth
MOBILENET_CHECKPOINT_PATH=/path/to/checkpoints_mobilenet/best_model_calibrated.pth
FRONTEND_URL=http://localhost:3000
DAILY_REQUEST_LIMIT=200
# DATABASE_URL=postgresql://...   optional: result cache, audit log, feedback
```

Start the server from the `backend` folder with the project root on `PYTHONPATH`:

```bash
cd backend
PYTHONPATH=.. uvicorn app.main:app --reload --port 8000
```

On Windows PowerShell, set the variable first: `$env:PYTHONPATH = ".."`.

### Frontend

```bash
cd frontend
npm install
npm start
```

The page opens at http://localhost:3000 and calls the hosted API by default. To use your local backend, create `frontend/.env.development.local` containing `REACT_APP_API_URL=http://localhost:8000`.

### Tests

```bash
cd backend && pytest          # backend
cd frontend && npm test       # frontend
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Whether the models are loaded, and analyses left today |
| `POST` | `/api/predict` | Analyse one image. Form fields: `file`, `age`, `sex` |
| `POST` | `/api/report/generate` | Build a PDF from a report object |
| `POST` | `/api/feedback` | Record whether a result was useful (needs a database) |

Shortened example of a `/api/predict` response:

```json
{
  "scan_id": "8f6d05b2-...",
  "predictions": [
    {
      "disease": "Edema",
      "probability": 0.53,
      "positive": true,
      "model_scores": { "efficientnet_b0": 0.40, "mobilenet_v2": 0.55, "torchxrayvision": 0.64 },
      "disagreement": 0.23,
      "n_models_used": 3
    }
  ],
  "heatmaps": { "Edema": "<base64 PNG>" },
  "original": "<base64 PNG>",
  "report": { "llm_generated": false, "findings": "...", "impression": "...", "recommendations": ["..."] },
  "daily_requests_remaining": 187
}
```

`positive` is true when the averaged score is 0.5 or higher. The page words this as "flagged for review", never as a diagnosis.

## Frontend design

Light page, one deep-teal accent, and a dark panel only around the X-ray. Colours, fonts and corner sizes are defined once in `frontend/tailwind.config.js`.

- **Landing page:** what the tool does, what it cannot do, and one button to start.
- **Upload:** one column; age and sex are optional; two confirmations before the first upload (it is a research preview, and the image carries no names or ID numbers). The confirmation is remembered in the browser only.
- **Results:** the X-ray stays in view beside the findings. Selecting a finding shows its heatmap and each model's score. The written summary, the PDF button and background notes on a condition are on the same page.
- Status is always shown with an icon and a word as well as colour: "Flagged for review", "Below threshold", "Models disagree".

## Not built yet

- Accounts, saved history and an About page with validation results
- Published accuracy figures from a held-out test set
- More than five conditions; side-view images
- Server-side record of consent

## Acknowledgements

- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert) (Stanford ML Group) for the training data
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) (Cohen et al.)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (Jacob Gildenblat)
- [Groq](https://groq.com) for language-model inference

Built by Advait Gujar.
