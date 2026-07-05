import sys
import os
import torch
import numpy as np
from PIL import Image

# ── Only needed on Windows dev — on Linux/Docker deployment,
#    `src/` should already be present inside /app (copied in via
#    Dockerfile from the chessvision-api repo itself), so this
#    hardcoded path would be wrong there and is skipped.
if os.name == 'nt':
    sys.path.append(r'D:\Projects\ChestVision-AI')

from src.dataset import DISEASE_COLS
from src.models.ensemble import EnsemblePredictor
from src.gradcam import (
    preprocess_image, generate_heatmap,
    overlay_heatmap, get_spatial_description
)
from src.report import build_report


class ModelService:
    def __init__(self, efficientnet_checkpoint: str, mobilenet_checkpoint: str):
        """
        Both paths should point to *_calibrated.pth checkpoints —
        see EnsemblePredictor's docstring for why uncalibrated
        checkpoints would skew the averaged ensemble score.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ensemble = EnsemblePredictor(
            efficientnet_checkpoint=efficientnet_checkpoint,
            mobilenet_checkpoint=mobilenet_checkpoint,
            device=self.device
        )

        # Grad-CAM still runs against EfficientNet-B0 specifically — it's
        # the model we have full architectural control over. Extending
        # Grad-CAM to MobileNetV2/XRV as well is a reasonable later step,
        # not required for the ensemble's predictions to work.
        self.gradcam_model = self.ensemble.efficientnet

        print(f"ModelService ready on {self.device} (3-model ensemble)")

    def run_inference(self, image_path: str, age: float = 60.0,
                      sex: str = 'Unknown', threshold: float = 0.5):
        """
        Full ensemble inference pipeline with spatial Grad-CAM data
        passed through to LLM for anatomically precise reports.
        """
        # Preprocess — same tensor feeds EfficientNet-B0 and MobileNetV2
        # (both ImageNet-normalized). XRV does its own preprocessing
        # internally from img_pil, inside EnsemblePredictor.
        tensor, img_float, img_pil = preprocess_image(image_path)

        # Ensemble prediction — averages all 3 models, keeps individual
        # scores + disagreement per disease
        predictions = self.ensemble.predict(
            img_pil, tensor, age=age, sex=sex, threshold=threshold
        )

        # Generate heatmaps + spatial descriptions (EfficientNet-B0 only)
        heatmaps     = {}
        spatial_data = {}

        positives = [p for p in predictions if p['positive']]
        if not positives:
            positives = predictions[:2]

        for pred in positives:
            class_idx = DISEASE_COLS.index(pred['disease'])

            heatmap = generate_heatmap(
                self.gradcam_model, tensor, self.device,
                class_idx, age=age, sex=sex
            )
            overlay = overlay_heatmap(img_float, heatmap)
            spatial = get_spatial_description(heatmap, pred['disease'])

            heatmaps[pred['disease']] = {
                'heatmap': heatmap,
                'overlay': overlay,
            }
            spatial_data[pred['disease']] = spatial

        # Build report with spatial context — predictions now also carry
        # model_scores + disagreement per disease, available to report.py
        # once you're ready to weave that into the LLM prompt
        image_filename = image_path.split('\\')[-1].split('/')[-1]
        report = build_report(
            predictions,
            image_filename=image_filename,
            patient_age=age,
            patient_sex=sex,
            spatial_data=spatial_data
        )

        return {
            'predictions':  predictions,
            'heatmaps':     heatmaps,
            'report':       report,
            'img_float':    img_float,
            'spatial_data': spatial_data,
        }