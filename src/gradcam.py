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
from src.dataset import DISEASE_COLS, get_transforms


# ── Config ────────────────────────────────────────────────
CHECKPOINT  = r'D:\Projects\ChestVision-AI\checkpoints\best_model.pth'
SAVE_DIR    = r'D:\Projects\ChestVision-AI\checkpoints\gradcam_samples'
IMG_SIZE    = 224
# ──────────────────────────────────────────────────────────


def load_model(checkpoint_path, device):
    model = get_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path, img_size=224):
    """Load and preprocess a single image for inference."""
    img_pil = Image.open(image_path).convert('RGB')
    img_np  = np.array(img_pil)

    transform = get_transforms(mode='val', img_size=img_size)
    tensor    = transform(image=img_np)['image']            # [3, H, W]
    tensor    = tensor.unsqueeze(0)                          # [1, 3, H, W]

    # Keep original for overlay (resized, 0-1 float RGB)
    img_resized = cv2.resize(img_np, (img_size, img_size))
    img_float   = img_resized.astype(np.float32) / 255.0

    return tensor, img_float, img_pil


def predict(model, tensor, device, threshold=0.5):
    """Run inference and return probabilities + binary predictions."""
    tensor = tensor.to(device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda'):
            logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # [5]

    results = []
    for i, (disease, prob) in enumerate(zip(DISEASE_COLS, probs)):
        results.append({
            'disease':    disease,
            'probability': float(prob),
            'positive':   bool(prob >= threshold)
        })

    # Sort by probability descending
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results, probs


def generate_heatmap(model, tensor, device, class_idx):
    """Generate Grad-CAM heatmap for a specific disease class."""
    target_layers = model.get_gradcam_layer()

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]

    # Grad-CAM returns [1, H, W] grayscale heatmap
    grayscale_cam = cam(
        input_tensor=tensor.to(device),
        targets=targets
    )
    return grayscale_cam[0]  # [H, W]  values in [0, 1]


def overlay_heatmap(img_float, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    # Convert heatmap to RGB using jet colormap
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # [H, W, 3]  drop alpha channel

    # Blend
    overlay = (1 - alpha) * img_float + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_single(image_path, model, device, save_path=None):
    """
    Full pipeline: load image → predict → generate heatmaps → visualize.
    Returns predictions and heatmap overlays.
    """
    tensor, img_float, img_pil = preprocess_image(image_path)
    predictions, probs         = predict(model, tensor, device)

    # Get positive diseases (or top-2 if none above threshold)
    positive = [p for p in predictions if p['positive']]
    if not positive:
        positive = predictions[:2]

    n_cols  = len(positive) + 1           # original + one per disease
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    # Original image
    axes[0].imshow(img_float)
    axes[0].set_title('Original X-Ray', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Heatmap per positive disease
    heatmaps = {}
    for i, pred in enumerate(positive):
        class_idx = DISEASE_COLS.index(pred['disease'])
        heatmap   = generate_heatmap(model, tensor, device, class_idx)
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


def run_samples(dataset_root, model, device, n_samples=6):
    """
    Run Grad-CAM on sample images from the validation set
    to visually verify heatmap quality.
    """
    import pandas as pd

    os.makedirs(SAVE_DIR, exist_ok=True)

    val_csv = os.path.join(dataset_root, 'valid.csv')
    df      = pd.read_csv(val_csv)
    df      = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

    print(f"Running Grad-CAM on {n_samples} sample images...\n")

    for idx in range(min(n_samples, len(df))):
        row        = df.iloc[idx]
        image_path = os.path.join(
            dataset_root,
            row['Path'].replace('CheXpert-v1.0-small/', '')
        )

        if not os.path.exists(image_path):
            continue

        print(f"Image {idx+1}: {os.path.basename(image_path)}")
        save_path = os.path.join(SAVE_DIR, f'gradcam_sample_{idx+1}.png')

        predictions, _ = visualize_single(
            image_path, model, device, save_path=save_path
        )

        print("  Predictions:")
        for p in predictions:
            status = '✓' if p['positive'] else ' '
            print(f"  {status} {p['disease']:20s}: {p['probability']*100:.1f}%")
        print()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(CHECKPOINT, device)
    print("Model loaded successfully")

    run_samples(
        dataset_root=r'D:\Datasets\CheXpert-v1.0-small',
        model=model,
        device=device,
        n_samples=6
    )