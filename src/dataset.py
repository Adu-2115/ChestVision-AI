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

    # Fill NaN with 0 (not mentioned = negative)
    df[DISEASE_COLS] = df[DISEASE_COLS].fillna(0)

    # Handle uncertain labels (-1)
    for col in U_ONES:
        df[col] = df[col].replace(-1, 1)
    for col in U_ZEROS:
        df[col] = df[col].replace(-1, 0)

    return df


def encode_demographics(age: float, sex: str):
    """
    Encode age and sex into a normalized tensor.

    Age: normalized to 0-1 range, clipped at 0-100
         (handles 3 pediatric outliers under 15)

    Sex: Male=1.0, Female=0.0, Unknown=0.5
         (only 1 Unknown in entire CheXpert dataset)
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
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(
                translate_percent=0.05,
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                p=0.3
            ),
            A.CLAHE(clip_limit=2.0, p=0.4),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
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

        # Image
        img = Image.open(row['Path']).convert('RGB')
        img = np.array(img)
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