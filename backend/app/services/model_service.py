import sys
import torch
import numpy as np
from PIL import Image

sys.path.append(r'D:\Projects\ChestVision-AI')
from src.models.densenet import get_model
from src.dataset import DISEASE_COLS, get_transforms
from src.gradcam import (
    preprocess_image, predict, generate_heatmap,
    overlay_heatmap, load_model
)
from src.report import build_report, format_report_text


class ModelService:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = load_model(checkpoint_path, self.device)
        self.model.eval()
        print(f"ModelService ready on {self.device}")

    def run_inference(self, image_path: str, threshold: float = 0.5):
        """
        Full inference pipeline.
        Returns predictions, heatmaps as numpy arrays, and report dict.
        """
        # 1. Preprocess
        tensor, img_float, img_pil = preprocess_image(image_path)

        # 2. Predict
        predictions, probs = predict(
            self.model, tensor, self.device, threshold=threshold
        )

        # 3. Generate heatmaps for positive diseases
        heatmaps = {}
        positives = [p for p in predictions if p['positive']]
        if not positives:
            positives = predictions[:2]   # fallback: top 2

        for pred in positives:
            class_idx = DISEASE_COLS.index(pred['disease'])
            heatmap   = generate_heatmap(
                self.model, tensor, self.device, class_idx
            )
            overlay        = overlay_heatmap(img_float, heatmap)
            heatmaps[pred['disease']] = {
                'heatmap': heatmap,
                'overlay': overlay,
            }

        # 4. Build report
        image_filename = image_path.split('\\')[-1].split('/')[-1]
        report         = build_report(predictions, image_filename=image_filename)

        return {
            'predictions': predictions,
            'heatmaps':    heatmaps,
            'report':      report,
            'img_float':   img_float,
        }