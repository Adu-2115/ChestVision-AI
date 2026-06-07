import torch
import torch.nn as nn
import torchvision.models as models


class ChestVisionModel(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        # EfficientNet-B0 image backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        image_features = self.backbone.classifier[1].in_features  # 1280

        # Remove original classifier — we build our own fusion head
        self.backbone.classifier = nn.Identity()

        # Demographics embedding
        # Small on purpose — demographics add context, image always dominates
        # 1280 image features vs 32 demo features = image is 97.6% of signal
        self.demo_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fusion classifier
        fusion_features = image_features + 32  # 1280 + 32 = 1312
        self.classifier = nn.Sequential(
            nn.Linear(fusion_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
            # No sigmoid — BCEWithLogitsLoss handles it
        )

    def forward(self, x, demographics):
        # Image features — 1280 dim
        image_feat = self.backbone(x)

        # Demographic features — 32 dim
        demo_feat = self.demo_embedding(demographics)

        # Fuse and classify
        combined = torch.cat([image_feat, demo_feat], dim=1)
        return self.classifier(combined)

    def get_gradcam_layer(self):
        return [self.backbone.features[-1]]


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