"""
MobileNetV2 multimodal model — now matches densenet.py exactly, including
temperature scaling calibration. Save as: src/models/mobilenet.py
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ChestVisionMobileNet(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        # MobileNetV2 image backbone
        self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        image_features = self.backbone.classifier[1].in_features  # 1280

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Demographics embedding — identical to densenet.py
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
            # No sigmoid — BCEWithLogitsLoss handles it during training
        )

        # ── Temperature scaling for confidence calibration ────
        # Same as densenet.py: initialized to 1.0, frozen during training,
        # optimized post-training via calibrate()
        self.temperature = nn.Parameter(
            torch.ones(1) * 1.0,
            requires_grad=False
        )

    def forward(self, x, demographics):
        image_feat = self.backbone(x)
        demo_feat = self.demo_embedding(demographics)
        combined = torch.cat([image_feat, demo_feat], dim=1)
        return self.classifier(combined)

    def forward_calibrated(self, x, demographics):
        """
        Forward pass with temperature scaling applied.
        Use this during inference after calibration is done.
        """
        logits = self.forward(x, demographics)
        return logits / self.temperature

    def get_gradcam_layer(self):
        return [self.backbone.features[-1]]

    def calibrate(self, val_loader, device, max_iter=50):
        """
        Identical calibration procedure to densenet.py — find optimal
        temperature on validation set after training.
        """
        print("Running temperature scaling calibration...")

        self.temperature.requires_grad = True

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter
        )
        criterion = nn.BCEWithLogitsLoss()

        all_logits, all_labels = [], []

        self.eval()
        with torch.no_grad():
            for images, labels, demographics in val_loader:
                images       = images.to(device)
                demographics = demographics.to(device)
                logits       = self.forward(images, demographics)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        def eval_temp():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_temp)

        self.temperature.requires_grad = False

        print(f"Calibration complete. Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


def get_model(num_classes=5, dropout=0.3):
    return ChestVisionMobileNet(num_classes=num_classes, dropout=dropout)
