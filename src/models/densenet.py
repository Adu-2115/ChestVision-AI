import torch
import torch.nn as nn
import torchvision.models as models


class ChestVisionModel(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        # Load pretrained DenseNet-121
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')

        # Replace classifier head
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
            # No sigmoid — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        return self.backbone(x)

    def get_gradcam_layer(self):
        # Last dense block — best layer for Grad-CAM
        return [self.backbone.features.denseblock4]


def get_model(num_classes=5, dropout=0.3):
    model = ChestVisionModel(num_classes=num_classes, dropout=dropout)
    return model


def get_class_weights(train_df, disease_cols, device):
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance"""
    weights = []
    for col in disease_cols:
        pos    = (train_df[col] == 1).sum()
        neg    = (train_df[col] == 0).sum()
        weight = neg / (pos + 1e-6)
        weights.append(weight)
        print(f"{col:20s} pos={pos:6d}  neg={neg:6d}  weight={weight:.2f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)