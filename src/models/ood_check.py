"""
Wrapper around a MobileCLIP zero-shot classifier used as a second, learned
line of defense against non-chest-X-ray uploads that slip past the
physics-based is_valid_xray() pixel-statistics check (e.g. a grayscale,
high-contrast portrait photo can satisfy those heuristics while still not
being an X-ray at all).

Why MobileCLIP-S1 specifically: true MobileCLIP-S0 (the smallest variant)
requires Apple's own `mobileclip` pip package, whose declared requirements
force torch>=2.8.0 — that would fight this project's pinned CPU-only
torch==2.5.1 (see Dockerfile comment; that CUDA-bloat mismatch caused a
real OOM crash-loop before). MobileCLIP-S1 is natively supported by
open_clip>=2.26.1,<3.2.0 (only requires torch>=1.9/2.0, no conflict) and
is still 44% smaller than standard CLIP ViT-B/32 (340MB vs 605MB).

Loaded once at startup via get_ood_checker(), same pattern as XRVWrapper's
get_xrv_model(). If loading fails for any reason (weights should already
be pre-baked into the Docker image at build time — see Dockerfile — but
this covers a corrupted cache, missing open_clip install, etc.), this
degrades gracefully: get_ood_checker() returns None and callers must fall
back to physics-only validation rather than crashing the app.

Save this as: src/models/ood_check.py
"""
import logging
import torch

logger = logging.getLogger("chestvision.ood_check")

MODEL_NAME = 'MobileCLIP-S1'
PRETRAINED_TAG = 'datacompdr'

# Zero-shot CLIP scores an image against ALL of these prompts and returns
# a softmax distribution over them. CHEST_XRAY_PROMPTS describes what we
# want to accept; NEGATIVE_PROMPTS covers common categories of things that
# should not reach the (expensive) disease-detection ensemble. Splitting
# the "accept" side across a few phrasings (rather than just one prompt)
# avoids the decision being overly sensitive to a single prompt's exact
# wording — see classify_probs() below for how the two sides are compared.
CHEST_XRAY_PROMPTS = [
    "a chest x-ray",
    "a radiograph of a human chest",
    "a black and white medical x-ray scan of the lungs and ribcage",
]
NEGATIVE_PROMPTS = [
    "a photograph of a person",
    "a photo of a human face",
    "an x-ray of a body part other than the chest",
    "a random photograph or object",
    "a natural photograph, not a medical scan",
]
ALL_PROMPTS = CHEST_XRAY_PROMPTS + NEGATIVE_PROMPTS
N_POSITIVE_PROMPTS = len(CHEST_XRAY_PROMPTS)

# Combined probability mass over CHEST_XRAY_PROMPTS needed to accept.
DEFAULT_THRESHOLD = 0.5


def classify_probs(probs, n_positive=N_POSITIVE_PROMPTS, prompts=ALL_PROMPTS,
                    threshold=DEFAULT_THRESHOLD):
    """
    Pure decision logic, pulled out of CLIPOODChecker.predict() so it's
    unit-testable without loading the actual CLIP model. `probs` is a
    softmax array over `prompts` (positive prompts first, then negative).

    Summing mass across all chest-x-ray prompts (rather than just checking
    whether the single top prompt is a positive one) means a genuine X-ray
    that splits its confidence across two similarly-worded positive
    prompts still gets counted correctly, instead of being penalized for
    "diluting" its own vote.
    """
    positive_mass = float(sum(probs[:n_positive]))
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        'is_chest_xray': positive_mass >= threshold,
        'confidence': positive_mass,
        'top_label': prompts[top_idx],
    }


class CLIPOODChecker:
    def __init__(self, device=None):
        import open_clip  # imported lazily so a broken/missing install only affects get_ood_checker(), not app startup as a whole

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_TAG
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        with torch.no_grad():
            text_tokens = tokenizer(ALL_PROMPTS).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        print(f"[CLIPOODChecker] Loaded {MODEL_NAME} ({PRETRAINED_TAG}) with {len(ALL_PROMPTS)} prompts")

    def predict(self, img_pil):
        """
        img_pil: PIL Image (any mode). Returns a dict — see classify_probs().
        """
        image = self.preprocess(img_pil.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()[0]

        return classify_probs(probs)


_checker_instance = None
_load_attempted = False


def get_ood_checker(device=None):
    """
    Loads the CLIP OOD checker once and caches the result (including a
    failed attempt, so a broken install doesn't retry-and-warn on every
    single request). Returns None if loading fails — callers must treat
    that as "OOD checking unavailable, fall back to physics-only X-ray
    validation" rather than erroring out.
    """
    global _checker_instance, _load_attempted
    if _load_attempted:
        return _checker_instance

    _load_attempted = True
    try:
        _checker_instance = CLIPOODChecker(device=device)
    except Exception as e:
        logger.warning(
            f"CLIP OOD checker failed to load — falling back to physics-only "
            f"X-ray validation. Error: {e}"
        )
        _checker_instance = None
    return _checker_instance
