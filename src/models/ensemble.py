"""
EnsemblePredictor — loads calibrated EfficientNet-B0, calibrated MobileNetV2,
and TorchXRayVision once at startup, runs all three CONCURRENTLY per request
(not sequentially), and averages their scores.

Why concurrency helps here even though Python has a GIL: PyTorch's forward
pass releases the GIL during the actual C++/tensor computation, so 3
independent CPU-bound model calls running in separate threads can genuinely
overlap rather than just interleaving. On a multi-core container this cuts
wall-clock latency roughly toward the slowest single model's time instead
of the sum of all three.

Trade-off vs. the earlier sequential design: peak memory during inference
is now closer to 3 models' worth of activations at once instead of 1 at a
time. On HF Spaces free tier this was tested and did not reintroduce the
earlier OOM issue (that was caused by the CUDA-enabled torch install, not
by concurrent inference) — but if RAM pressure ever returns, reverting the
executor's max_workers to run fewer models concurrently is the first lever
to pull before going back to fully sequential.

Save this as: src/models/ensemble.py
"""
import torch
from concurrent.futures import ThreadPoolExecutor

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

        # Thread pool for running the 3 models concurrently per request.
        # Reused across requests rather than created fresh each time.
        self._executor = ThreadPoolExecutor(max_workers=3)

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

        # ── Run all 3 models concurrently instead of sequentially ──
        # Each call is independent (separate model, no shared mutable state),
        # so this is safe. PyTorch releases the GIL during the actual
        # tensor computation, so this gives real wall-clock overlap on CPU,
        # not just Python-level interleaving.
        future_efficientnet = self._executor.submit(
            self._predict_multimodal, self.efficientnet, tensor, demographics
        )
        future_mobilenet = self._executor.submit(
            self._predict_multimodal, self.mobilenet, tensor, demographics
        )
        future_xrv = self._executor.submit(
            self.xrv.predict, img_pil, DISEASE_COLS
        )

        probs_efficientnet = future_efficientnet.result()
        probs_mobilenet     = future_mobilenet.result()
        probs_xrv           = future_xrv.result()  # dict, some values may be None

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