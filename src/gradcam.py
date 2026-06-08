import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(r'D:\Projects\ChestVision-AI')
from src.models.densenet import get_model
from src.dataset import DISEASE_COLS, get_transforms, encode_demographics


CHECKPOINT = r'D:\Projects\ChestVision-AI\checkpoints_efficientnet\best_model.pth'
SAVE_DIR   = r'D:\Projects\ChestVision-AI\checkpoints_efficientnet\gradcam_samples'
IMG_SIZE   = 224


def load_model(checkpoint_path: str, device):
    model      = get_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded — type: {checkpoint.get('model_type', 'unknown')}")
    print(f"Inputs: {checkpoint.get('inputs', ['image'])}")
    return model


def preprocess_image(image_path: str, img_size: int = 224):
    """
    Load and preprocess image. Supports JPEG, PNG and DICOM (.dcm).
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext == '.dcm':
        img_pil = _load_dicom(image_path)
    else:
        img_pil = Image.open(image_path).convert('RGB')

    img_np    = np.array(img_pil)
    transform = get_transforms(mode='val', img_size=img_size)
    tensor    = transform(image=img_np)['image'].unsqueeze(0)

    img_resized = cv2.resize(img_np, (img_size, img_size))
    img_float   = img_resized.astype(np.float32) / 255.0

    return tensor, img_float, img_pil


def _load_dicom(dicom_path: str) -> Image.Image:
    """Load DICOM file and convert to RGB PIL Image."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds        = pydicom.dcmread(dicom_path)
    pixel_arr = apply_voi_lut(ds.pixel_array.astype(float), ds)

    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            pixel_arr = pixel_arr.max() - pixel_arr

    pixel_arr = pixel_arr - pixel_arr.min()
    if pixel_arr.max() > 0:
        pixel_arr = pixel_arr / pixel_arr.max()
    pixel_arr = (pixel_arr * 255).astype(np.uint8)

    return Image.fromarray(pixel_arr).convert('RGB')


def predict(model, tensor, device, age: float = 60.0,
            sex: str = 'Unknown', threshold: float = 0.5):
    """
    Run inference. Age and sex are used by the multimodal model.
    Defaults: age=60 (dataset mean), sex='Unknown' → 0.5 (neutral)
    """
    demographics = encode_demographics(age=age, sex=sex).unsqueeze(0).to(device)
    tensor       = tensor.to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda'):
                logits = model(tensor, demographics)
        else:
            logits = model(tensor, demographics)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = []
    for i, (disease, prob) in enumerate(zip(DISEASE_COLS, probs)):
        results.append({
            'disease':     disease,
            'probability': float(prob),
            'positive':    bool(prob >= threshold)
        })

    results.sort(key=lambda x: x['probability'], reverse=True)
    return results, probs


class DemographicsWrapper(torch.nn.Module):
    """
    Wrapper to make model compatible with GradCAM.
    GradCAM expects model(image) but ours needs model(image, demographics).
    This wrapper stores demographics and passes them automatically.
    """
    def __init__(self, model, demographics):
        super().__init__()
        self.model        = model
        self.demographics = demographics

    def forward(self, x):
        return self.model(x, self.demographics)


def generate_heatmap(model, tensor, device, class_idx,
                     age: float = 60.0, sex: str = 'Unknown'):
    """Generate Grad-CAM heatmap for a specific disease class."""
    demographics  = encode_demographics(age=age, sex=sex).unsqueeze(0).to(device)
    wrapped_model = DemographicsWrapper(model, demographics)

    target_layers = [wrapped_model.model.backbone.features[-1]]
    cam           = GradCAM(model=wrapped_model, target_layers=target_layers)
    targets       = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=tensor.to(device), targets=targets)
    return grayscale_cam[0]


def overlay_heatmap(img_float, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    overlay         = (1 - alpha) * img_float + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def visualize_single(image_path, model, device, age=60.0,
                     sex='Unknown', save_path=None):
    """Full pipeline: load → predict → heatmaps → visualize."""
    tensor, img_float, img_pil = preprocess_image(image_path)
    predictions, probs         = predict(model, tensor, device,
                                         age=age, sex=sex)

    positive = [p for p in predictions if p['positive']]
    if not positive:
        positive = predictions[:2]

    n_cols  = len(positive) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_float)
    axes[0].set_title('Original X-Ray', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    heatmaps = {}
    for i, pred in enumerate(positive):
        class_idx = DISEASE_COLS.index(pred['disease'])
        heatmap   = generate_heatmap(model, tensor, device,
                                     class_idx, age=age, sex=sex)
        overlay   = overlay_heatmap(img_float, heatmap)

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(
            f"{pred['disease']}\n{pred['probability']*100:.1f}%",
            fontsize=10,
            color='red' if pred['positive'] else 'gray'
        )
        axes[i + 1].axis('off')
        heatmaps[pred['disease']] = heatmap

    plt.suptitle('Grad-CAM Visualization', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return predictions, heatmaps