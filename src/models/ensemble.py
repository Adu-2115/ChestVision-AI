"""
EnsemblePredictor — loads calibrated EfficientNet-B0, calibrated MobileNetV2,
and TorchXRayVision once at startup, runs all three sequentially per request,
and averages their scores.

IMPORTANT: uses forward_calibrated() (temperature-scaled) for both multimodal
models, and expects the *_calibrated.pth checkpoints, not the raw best_model.pth
ones. Mixing a calibrated model with an uncalibrated one in the same average
would skew the ensemble toward whichever one is more overconfident — so both
must go through the same calibration step before being combined here.

TorchXRayVision has no calibration step of its own (pretrained, external) —
its raw sigmoid outputs are used as-is, same as before.

Save this as: src/models/ensemble.py
"""
import torch

from src.dataset import DISEASE_COLS, get_transforms, encode_demographics
from src.models.densenet import get_model as get_efficientnet
from src.models.mobilenet import get_model as get_mobilenet
from src.models.xrv_wrapper import get_xrv_model


class EnsemblePredictor:
    def __init__(self, efficientnet_checkpoint: str, mobilenet_checkpoint: str,
                 device=None):
        """
        efficientnet_checkpoint / mobilenet_checkpoint should point to the
        *_calibrated.pth files (produced after running model.calibrate()),
        not the raw best_model.pth checkpoints.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Model 1: EfficientNet-B0 (calibrated) ────────────────
        print("[Ensemble] Loading EfficientNet-B0 (calibrated)...")
        self.efficientnet = get_efficientnet(num_classes=len(DISEASE_COLS)).to(self.device)
        ckpt = torch.load(efficientnet_checkpoint, map_location=self.device, weights_only=False)
        self.efficientnet.load_state_dict(ckpt['model_state_dict'])
        self.efficientnet.eval()
        self._warn_if_uncalibrated(ckpt, 'EfficientNet-B0')

        # ── Model 2: MobileNetV2 (calibrated) ─────────────────────
        print("[Ensemble] Loading MobileNetV2 (calibrated)...")
        self.mobilenet = get_mobilenet(num_classes=len(DISEASE_COLS)).to(self.device)
        ckpt = torch.load(mobilenet_checkpoint, map_location=self.device, weights_only=False)
        self.mobilenet.load_state_dict(ckpt['model_state_dict'])
        self.mobilenet.eval()
        self._warn_if_uncalibrated(ckpt, 'MobileNetV2')

        # ── Model 3: TorchXRayVision (pretrained, no calibration step) ──
        print("[Ensemble] Loading TorchXRayVision...")
        self.xrv = get_xrv_model(device=self.device)

        self.transform = get_transforms(mode='val', img_size=224)
        print("[Ensemble] All 3 models loaded.")

    @staticmethod
    def _warn_if_uncalibrated(ckpt, name):
        temp = ckpt.get('temperature', None)
        if temp is None:
            print(f"[Ensemble] WARNING: '{name}' checkpoint has no 'temperature' key — "
                  f"make sure you loaded the *_calibrated.pth file, not the raw best_model.pth.")
        elif abs(temp - 1.0) < 1e-6:
            print(f"[Ensemble] NOTE: '{name}' temperature is exactly 1.0 — "
                  f"calibration may not have run, or had no effect.")
        else:
            print(f"[Ensemble] '{name}' calibrated temperature: {temp:.4f}")

    def _predict_multimodal(self, model, tensor, demographics):
        """Uses forward_calibrated() — temperature-scaled logits."""
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.autocast(device_type='cuda'):
                    logits = model.forward_calibrated(tensor, demographics)
            else:
                logits = model.forward_calibrated(tensor, demographics)
            return torch.sigmoid(logits).cpu().numpy()[0]

    def predict(self, img_pil, tensor, age: float = 60.0, sex: str = 'Unknown',
                threshold: float = 0.5):
        """
        img_pil : PIL Image (original, for XRV's own preprocessing)
        tensor  : preprocessed tensor from gradcam.preprocess_image(), for
                  the two multimodal models
        """
        demographics = encode_demographics(age=age, sex=sex).unsqueeze(0).to(self.device)
        tensor = tensor.to(self.device)

        # Sequential — not parallel — to cap peak memory on HF Spaces free tier
        probs_efficientnet = self._predict_multimodal(self.efficientnet, tensor, demographics)
        probs_mobilenet     = self._predict_multimodal(self.mobilenet, tensor, demographics)
        probs_xrv           = self.xrv.predict(img_pil, DISEASE_COLS)  # dict, some values may be None

        results = []
        for i, disease in enumerate(DISEASE_COLS):
            model_scores = {
                'efficientnet_b0': float(probs_efficientnet[i]),
                'mobilenet_v2':    float(probs_mobilenet[i]),
                'torchxrayvision': probs_xrv[disease],  # may be None
            }

            valid_scores = [v for v in model_scores.values() if v is not None]
            avg_score = sum(valid_scores) / len(valid_scores)
            disagreement = max(valid_scores) - min(valid_scores)

            results.append({
                'disease':       disease,
                'probability':   avg_score,
                'positive':      bool(avg_score >= threshold),
                'model_scores':  model_scores,
                'disagreement':  round(disagreement, 4),
                'n_models_used': len(valid_scores),
            })

        results.sort(key=lambda x: x['probability'], reverse=True)
        return results
