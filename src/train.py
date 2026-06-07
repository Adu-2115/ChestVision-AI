import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders, DISEASE_COLS
from src.models.densenet import get_model, get_class_weights


# ── Config ────────────────────────────────────────────────
DATASET_ROOT = r'D:\Datasets\CheXpert-v1.0-small'
SAVE_DIR     = r'D:\Projects\ChestVision-AI\checkpoints_efficientnet'
BATCH_SIZE   = 16
NUM_EPOCHS   = 20
LR           = 1e-4
IMG_SIZE     = 224
NUM_WORKERS  = 2
# ──────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_labels, all_preds = [], []

    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_labels.append(labels.cpu().detach().numpy())
        all_preds.append(torch.sigmoid(outputs).cpu().detach().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    avg_loss   = total_loss / len(loader)

    aucs = []
    for i in range(len(DISEASE_COLS)):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        except Exception:
            aucs.append(0.0)

    return avg_loss, float(np.mean(aucs)), aucs


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    avg_loss   = total_loss / len(loader)

    aucs = []
    for i in range(len(DISEASE_COLS)):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        except Exception:
            aucs.append(0.0)

    return avg_loss, float(np.mean(aucs)), aucs


def plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_auc'], label='Train')
    axes[1].plot(history['val_auc'],   label='Val')
    axes[1].set_title('Mean AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=100)
    plt.show()
    print(f"Plot saved to {save_dir}\\training_history.png")


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Data
    train_loader, val_loader, train_df, _ = get_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS
    )

    # Model
    model = get_model(num_classes=len(DISEASE_COLS)).to(device)

    # Loss with class weights
    pos_weights = get_class_weights(train_df, DISEASE_COLS, device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # Fixed: device param added to GradScaler
    scaler = torch.amp.GradScaler(device='cuda')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc':  [], 'val_auc':  []
    }
    best_auc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):

        # Freeze backbone for first 2 epochs — only train the head
        if epoch == 1:
            print("Epochs 1-2: Training head only (backbone frozen)\n")
            for param in model.backbone.features.parameters():
                param.requires_grad = False

        # Unfreeze full model from epoch 3
        if epoch == 3:
            print("Epoch 3+: Unfreezing full model\n")
            for param in model.backbone.features.parameters():
                param.requires_grad = True

        train_loss, train_auc, _         = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss,   val_auc,   val_aucs  = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}")
        print(f"  Train  loss={train_loss:.4f}  AUC={train_auc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  AUC={val_auc:.4f}")
        print(f"  Per-disease val AUC:")
        for d, a in zip(DISEASE_COLS, val_aucs):
            print(f"    {d:20s}: {a:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc':              val_auc,
                'disease_cols':         DISEASE_COLS
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"  *** Saved best model (AUC={val_auc:.4f}) ***")

        print()

    plot_history(history, SAVE_DIR)
    print(f"Training complete. Best val AUC: {best_auc:.4f}")
    print(f"Model saved to: {SAVE_DIR}\\best_model.pth")


if __name__ == '__main__':
    train()