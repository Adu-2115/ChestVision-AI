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
    def __init__(self, model, demographics):
        super().__init__()
        self.model        = model
        self.demographics = demographics

    def forward(self, x):
        return self.model(x, self.demographics)


def generate_heatmap(model, tensor, device, class_idx,
                     age: float = 60.0, sex: str = 'Unknown'):
    demographics  = encode_demographics(age=age, sex=sex).unsqueeze(0).to(device)
    wrapped_model = DemographicsWrapper(model, demographics)
    target_layers = [wrapped_model.model.backbone.features[-1]]
    cam           = GradCAM(model=wrapped_model, target_layers=target_layers)
    targets       = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=tensor.to(device), targets=targets)
    return grayscale_cam[0]


def get_spatial_description(heatmap: np.ndarray, disease: str) -> dict:
    """
    Convert Grad-CAM heatmap to anatomical spatial description.

    Divides chest X-ray into 7 clinical anatomical regions:
    - Upper/mid/lower left and right lung zones
    - Central mediastinum (heart, major vessels)

    Returns structured spatial data for LLM prompt enrichment.
    Based on Beer-Lambert attenuation zones from clinical X-ray physics.
    """
    h, w = heatmap.shape

    # 7 anatomical regions based on chest X-ray clinical zones
    regions = {
        'upper left lung zone':  heatmap[:h//3,       :w//2],
        'upper right lung zone': heatmap[:h//3,       w//2:],
        'mid left lung zone':    heatmap[h//3:2*h//3, :w//2],
        'mid right lung zone':   heatmap[h//3:2*h//3, w//2:],
        'lower left lung zone':  heatmap[2*h//3:,     :w//2],
        'lower right lung zone': heatmap[2*h//3:,     w//2:],
        'central mediastinum':   heatmap[h//4:3*h//4, w//3:2*w//3],
    }

    region_scores = {
        name: float(np.mean(region))
        for name, region in regions.items()
    }

    # Significantly activated regions (threshold 0.35)
    activated_regions = [
        name for name, score in region_scores.items()
        if score > 0.35
    ]

    peak_region = max(region_scores, key=region_scores.get)
    peak_score  = region_scores[peak_region]

    # Determine laterality
    left_score  = np.mean([
        region_scores['upper left lung zone'],
        region_scores['mid left lung zone'],
        region_scores['lower left lung zone']
    ])
    right_score = np.mean([
        region_scores['upper right lung zone'],
        region_scores['mid right lung zone'],
        region_scores['lower right lung zone']
    ])

    if abs(left_score - right_score) < 0.1:
        laterality = "bilateral"
    elif left_score > right_score:
        laterality = "left-sided predominant"
    else:
        laterality = "right-sided predominant"

    # Clinical description
    if not activated_regions:
        description = f"Diffuse low-level activation for {disease}"
    else:
        regions_text = ', '.join(activated_regions)
        description  = (
            f"{disease}: {laterality} pattern, "
            f"most prominent in {peak_region} "
            f"(score: {peak_score:.2f}). "
            f"Activated zones: {regions_text}."
        )

    return {
        'disease':           disease,
        'description':       description,
        'activated_regions': activated_regions,
        'peak_region':       peak_region,
        'laterality':        laterality,
        'region_scores':     region_scores,
    }


def overlay_heatmap(img_float, heatmap, alpha=0.4):
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    overlay         = (1 - alpha) * img_float + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def visualize_single(image_path, model, device, age=60.0,
                     sex='Unknown', save_path=None):
    tensor, img_float, img_pil = preprocess_image(image_path)
    predictions, probs         = predict(model, tensor, device,
                                         age=age, sex=sex)

    positive = [p for p in predictions if p['positive']]
    if not positive:
        positive = predictions[:2]

    n_cols    = len(positive) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_float)
    axes[0].set_title('Original X-Ray', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    heatmaps             = {}
    spatial_descriptions = {}

    for i, pred in enumerate(positive):
        class_idx = DISEASE_COLS.index(pred['disease'])
        heatmap   = generate_heatmap(model, tensor, device,
                                     class_idx, age=age, sex=sex)
        overlay   = overlay_heatmap(img_float, heatmap)
        spatial   = get_spatial_description(heatmap, pred['disease'])

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(
            f"{pred['disease']}\n{pred['probability']*100:.1f}%",
            fontsize=10,
            color='red' if pred['positive'] else 'gray'
        )
        axes[i + 1].axis('off')

        heatmaps[pred['disease']]             = heatmap
        spatial_descriptions[pred['disease']] = spatial

    plt.suptitle('Grad-CAM Visualization', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    plt.show()
    return predictions, heatmaps, spatial_descriptions