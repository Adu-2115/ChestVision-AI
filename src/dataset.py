import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


DISEASE_COLS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]

U_ONES  = ['Atelectasis', 'Edema']
U_ZEROS = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']


def load_and_clean_csv(csv_path, dataset_root):
    df = pd.read_csv(csv_path)

    # Keep frontal only
    df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

    # Fix image paths
    df['Path'] = df['Path'].apply(
        lambda x: os.path.join(dataset_root, x.replace('CheXpert-v1.0-small/', ''))
    )

    # Fill NaN with 0
    df[DISEASE_COLS] = df[DISEASE_COLS].fillna(0)

    # Handle uncertain labels
    for col in U_ONES:
        df[col] = df[col].replace(-1, 1)
    for col in U_ZEROS:
        df[col] = df[col].replace(-1, 0)

    return df


def encode_demographics(age: float, sex: str):
    """
    Encode age and sex into normalized tensor.
    Age: clipped 0-100, normalized to 0-1
    Sex: Male=1.0, Female=0.0, Unknown=0.5
    """
    age_normalized = float(np.clip(age, 0, 100)) / 100.0

    if sex == 'Male':
        sex_encoded = 1.0
    elif sex == 'Female':
        sex_encoded = 0.0
    else:
        sex_encoded = 0.5

    return torch.tensor([age_normalized, sex_encoded], dtype=torch.float32)


def get_transforms(mode='train', img_size=224):
    """
    Preprocessing pipeline based on clinical X-ray image processing principles.

    Training pipeline includes:
    1. Noise reduction   — GaussianBlur (Wiener-style)
    2. Contrast enhance  — CLAHE (medical standard), RandomGamma
    3. Edge sharpening   — Sharpen (Laplacian-based)
    4. Augmentation      — Flip, Affine, BrightnessContrast
    5. Normalization     — Z-score (ImageNet stats, standard for transfer learning)

    Val/inference pipeline:
    1. CLAHE             — consistent contrast normalization
    2. Z-score normalize — same as training
    """
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),

            # ── Noise reduction (Wiener-style) ────────────────
            # Removes quantum noise inherent to X-ray photon capture
            # Small kernel to preserve fine detail
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),

            # ── Contrast enhancement ──────────────────────────
            # CLAHE: medical standard, processes tiles separately
            # prevents noise explosion in uniform regions
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),

            # Gamma correction: handles over/underexposed X-rays
            # gamma<1 brightens darks, gamma>1 darkens brights
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # ── Edge sharpening (Laplacian-based) ─────────────
            # Enhances rib margins, vessel markings, organ boundaries
            # Critical for disease boundary detection
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.8, 1.2), p=0.3),

            # ── Augmentation ──────────────────────────────────
            # Horizontal flip (left/right lung symmetry)
            A.HorizontalFlip(p=0.5),

            # Mild brightness/contrast variation
            # Simulates different X-ray machine exposure settings
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),

            # Affine transforms — small shifts/rotations only
            # X-rays have strict anatomical orientation
            A.Affine(
                translate_percent=0.03,
                scale=(0.97, 1.03),
                rotate=(-5, 5),
                p=0.3
            ),

            # ── Normalization (Z-score) ───────────────────────
            # ImageNet stats — standard for transfer learning
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    else:
        # Val/inference: only essential preprocessing, no augmentation
        return A.Compose([
            A.Resize(img_size, img_size),

            # CLAHE for consistent contrast across different X-ray sources
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),

            # Z-score normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class CheXpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image as RGB
        img = Image.open(row['Path']).convert('RGB')
        img = np.array(img)

        # Apply transforms
        if self.transform:
            img = self.transform(image=img)['image']

        # Labels
        labels = torch.tensor(
            row[DISEASE_COLS].values.astype(np.float32),
            dtype=torch.float32
        )

        # Demographics
        demographics = encode_demographics(
            age=row['Age'],
            sex=row['Sex']
        )

        return img, labels, demographics


def get_dataloaders(dataset_root, batch_size=16, val_split=0.15,
                    img_size=224, num_workers=2):
    csv_path = os.path.join(dataset_root, 'train.csv')
    df       = load_and_clean_csv(csv_path, dataset_root)

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=42,
        stratify=df['Pleural Effusion'].astype(int)
    )

    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")

    train_dataset = CheXpertDataset(
        train_df, transform=get_transforms('train', img_size)
    )
    val_dataset = CheXpertDataset(
        val_df, transform=get_transforms('val', img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_df, val_df