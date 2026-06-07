import sys
import torch
import numpy as np
from PIL import Image

sys.path.append(r'D:\Projects\ChestVision-AI')
from src.models.densenet import get_model
from src.dataset import DISEASE_COLS, get_transforms, encode_demographics
from src.gradcam import (
    preprocess_image, predict, generate_heatmap,
    overlay_heatmap, load_model
)
from src.report import build_report


class ModelService:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = load_model(checkpoint_path, self.device)
        self.model.eval()
        print(f"ModelService ready on {self.device}")

    def run_inference(self, image_path: str, age: float = 60.0,
                      sex: str = 'Unknown', threshold: float = 0.5):
        """
        Full inference pipeline with demographics support.
        age: patient age (default 60 = dataset mean)
        sex: Male / Female / Unknown
        """
        # Preprocess image
        tensor, img_float, img_pil = preprocess_image(image_path)

        # Predict with demographics
        predictions, probs = predict(
            self.model, tensor, self.device,
            age=age, sex=sex, threshold=threshold
        )

        # Generate heatmaps
        heatmaps = {}
        positives = [p for p in predictions if p['positive']]
        if not positives:
            positives = predictions[:2]

        for pred in positives:
            class_idx = DISEASE_COLS.index(pred['disease'])
            heatmap   = generate_heatmap(
                self.model, tensor, self.device,
                class_idx, age=age, sex=sex
            )
            overlay = overlay_heatmap(img_float, heatmap)
            heatmaps[pred['disease']] = {
                'heatmap': heatmap,
                'overlay': overlay,
            }

        # Build report
        image_filename = image_path.split('\\')[-1].split('/')[-1]
        report = build_report(
            predictions,
            image_filename=image_filename,
            patient_age=age,
            patient_sex=sex
        )

        return {
            'predictions': predictions,
            'heatmaps':    heatmaps,
            'report':      report,
            'img_float':   img_float,
        }