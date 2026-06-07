# import torch
# import torch.nn as nn
# import torchvision.models as models


# class ChestVisionModel(nn.Module):
#     def __init__(self, num_classes=5, dropout=0.3):
#         super().__init__()

#         # EfficientNet-B0 — 20MB, fits Render free tier
#         self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

#         # Replace classifier head
#         in_features = self.backbone.classifier[1].in_features  # 1280
#         self.backbone.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(in_features, num_classes)
#         )

#     def forward(self, x):
#         return self.backbone(x)

#     def get_gradcam_layer(self):
#         # Last conv block — best for Grad-CAM on EfficientNet
#         return [self.backbone.features[-1]]


# def get_model(num_classes=5, dropout=0.3):
#     return ChestVisionModel(num_classes=num_classes, dropout=dropout)


# def get_class_weights(train_df, disease_cols, device):
#     weights = []
#     for col in disease_cols:
#         pos    = (train_df[col] == 1).sum()
#         neg    = (train_df[col] == 0).sum()
#         weight = neg / (pos + 1e-6)
#         weights.append(weight)
#         print(f"{col:20s} pos={pos:6d}  neg={neg:6d}  weight={weight:.2f}")
#     return torch.tensor(weights, dtype=torch.float32).to(device)

import torch
import torch.nn as nn
import torchvision.models as models


class ChestVisionModel(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        self.backbone = models.densenet121(weights='IMAGENET1K_V1')

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_gradcam_layer(self):
        return [self.backbone.features.denseblock4]


def get_model(num_classes=5, dropout=0.3):
    return ChestVisionModel(num_classes=num_classes, dropout=dropout)


def get_class_weights(train_df, disease_cols, device):
    weights = []
    for col in disease_cols:
        pos    = (train_df[col] == 1).sum()
        neg    = (train_df[col] == 0).sum()
        weight = neg / (pos + 1e-6)
        weights.append(weight)
        print(f"{col:20s} pos={pos:6d}  neg={neg:6d}  weight={weight:.2f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)