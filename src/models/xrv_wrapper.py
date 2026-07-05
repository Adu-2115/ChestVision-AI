"""
Wrapper around a pretrained TorchXRayVision model (chest-specific weights,
no training needed on our end).

IMPORTANT — why this needs its own preprocessing:
TorchXRayVision models do NOT use ImageNet normalization like our other two
models. They expect:
  - Single-channel (grayscale) input
  - Pixel values scaled to roughly [-1024, 1024], via xrv.datasets.normalize
  - A center-crop + resize via xrv.datasets.XRayResizer
Reusing our existing `get_transforms()` (ImageNet mean/std, 3-channel) would
silently produce garbage predictions from this model. Keep this preprocessing
path completely separate from dataset.py's transforms.

TorchXRayVision also predicts against a fixed universal pathology taxonomy —
not every model checkpoint covers every pathology (some entries are NaN for
a given checkpoint). We match by name at runtime and only use pathologies
this specific model actually supports, rather than assuming a hardcoded list.

Save this as: src/models/xrv_wrapper.py
"""
import numpy as np
import torch
import torchxrayvision as xrv
import skimage


# Our 5 target labels -> possible XRV taxonomy names for the same concept.
# XRV's universal taxonomy sometimes uses different naming than CheXpert's
# raw column names (e.g. "Effusion" vs "Pleural Effusion" in older releases).
LABEL_ALIASES = {
    'Atelectasis':      ['Atelectasis'],
    'Cardiomegaly':     ['Cardiomegaly'],
    'Consolidation':    ['Consolidation'],
    'Edema':            ['Edema'],
    'Pleural Effusion': ['Pleural Effusion', 'Effusion'],
}


class XRVWrapper:
    """
    Image-only pretrained model. No demographics input — XRV models were
    trained without a demographics fusion head, so we only use the image
    branch of the ensemble vote for this model.
    """

    def __init__(self, weights='densenet121-res224-chex', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = xrv.models.DenseNet(weights=weights).to(self.device)
        self.model.eval()

        # Build the mapping from our disease_cols -> index in this model's
        # pathologies list, skipping any disease this checkpoint doesn't cover.
        self.label_to_idx = {}
        for our_label, aliases in LABEL_ALIASES.items():
            for alias in aliases:
                if alias in self.model.pathologies:
                    idx = self.model.pathologies.index(alias)
                    self.label_to_idx[our_label] = idx
                    break

        missing = [d for d in LABEL_ALIASES if d not in self.label_to_idx]
        if missing:
            print(f"[XRVWrapper] WARNING: checkpoint '{weights}' does not cover: {missing}. "
                  f"These diseases will be excluded from this model's vote.")
        print(f"[XRVWrapper] Loaded '{weights}', covering: {list(self.label_to_idx.keys())}")

    def preprocess(self, img_pil):
        """
        img_pil: PIL Image (any mode). Returns a (1, 1, 224, 224) tensor
        ready for this model — completely separate pipeline from dataset.py.
        """
        img = np.array(img_pil.convert('L')).astype(np.float32)  # grayscale

        # XRV expects [-1024, 1024] scaling, not [0,1] or ImageNet norm
        img = xrv.datasets.normalize(img, 255)

        img = img[None, ...]  # add channel dim -> (1, H, W)
        transform = xrv.datasets.XRayCenterCrop()
        img = transform(img)
        resizer = xrv.datasets.XRayResizer(224)
        img = resizer(img)

        tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, 1, 224, 224)
        return tensor

    def predict(self, img_pil, disease_cols):
        """
        Returns a dict: {disease_name: probability_or_None}
        None means this model doesn't cover that disease — the ensemble
        should skip it for this model rather than average in a bad number.
        """
        tensor = self.preprocess(img_pil).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        result = {}
        for disease in disease_cols:
            if disease in self.label_to_idx:
                result[disease] = float(probs[self.label_to_idx[disease]])
            else:
                result[disease] = None
        return result


def get_xrv_model(device=None):
    return XRVWrapper(weights='densenet121-res224-chex', device=device)
