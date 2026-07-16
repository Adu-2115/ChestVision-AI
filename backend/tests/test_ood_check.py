"""
Tests for classify_probs() — the CLIP zero-shot decision logic, tested in
isolation without loading the actual MobileCLIP-S1 model.

Save this as: backend/tests/test_ood_check.py
"""
import pytest
from src.models.ood_check import classify_probs, ALL_PROMPTS, N_POSITIVE_PROMPTS


def test_confident_chest_xray_accepted():
    """Mass concentrated on the positive prompts should be accepted."""
    probs = [0.70, 0.10, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]
    result = classify_probs(probs)

    assert result['is_chest_xray'] is True
    assert result['confidence'] == pytest.approx(0.85, abs=0.01)
    assert result['top_label'] == ALL_PROMPTS[0]


def test_confident_non_xray_rejected():
    """A portrait photo should push mass onto negative prompts."""
    probs = [0.02, 0.01, 0.01, 0.05, 0.60, 0.20, 0.06, 0.05]
    result = classify_probs(probs)

    assert result['is_chest_xray'] is False
    assert result['confidence'] == pytest.approx(0.04, abs=0.01)
    assert result['top_label'] == ALL_PROMPTS[4]


def test_threshold_boundary_exact_match_is_accepted():
    """Positive mass exactly equal to the threshold counts as accepted
    (>=, not >) — matches the convention used elsewhere in this codebase
    (see EnsemblePredictor.aggregate_scores)."""
    probs = [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    result = classify_probs(probs, threshold=0.5)
    assert result['is_chest_xray'] is True


def test_custom_threshold_is_respected():
    probs = [0.3, 0.1, 0.05, 0.2, 0.2, 0.1, 0.03, 0.02]
    # positive mass = 0.45
    assert classify_probs(probs, threshold=0.5)['is_chest_xray'] is False
    assert classify_probs(probs, threshold=0.4)['is_chest_xray'] is True


def test_top_label_picks_single_highest_prompt():
    """top_label should reflect the single best-matching prompt, not just
    which side (positive/negative) won the mass comparison."""
    probs = [0.10, 0.10, 0.10, 0.55, 0.05, 0.04, 0.03, 0.03]
    result = classify_probs(probs)
    assert result['top_label'] == ALL_PROMPTS[3]
    assert result['is_chest_xray'] is False


def test_prompt_lists_partition_correctly():
    """Sanity check that N_POSITIVE_PROMPTS actually matches the split
    used to build ALL_PROMPTS, so classify_probs() sums the right slice."""
    from src.models.ood_check import CHEST_XRAY_PROMPTS, NEGATIVE_PROMPTS
    assert ALL_PROMPTS == CHEST_XRAY_PROMPTS + NEGATIVE_PROMPTS
    assert N_POSITIVE_PROMPTS == len(CHEST_XRAY_PROMPTS)
    assert len(ALL_PROMPTS) == len(CHEST_XRAY_PROMPTS) + len(NEGATIVE_PROMPTS)


def test_get_ood_checker_returns_none_on_load_failure(monkeypatch):
    """If the underlying model fails to load, get_ood_checker() should
    return None (fail open) instead of raising — predict.py relies on this
    to fall back to physics-only validation without crashing the app."""
    import src.models.ood_check as ood_check_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(ood_check_module, "CLIPOODChecker", _boom)
    monkeypatch.setattr(ood_check_module, "_checker_instance", None)
    monkeypatch.setattr(ood_check_module, "_load_attempted", False)

    result = ood_check_module.get_ood_checker()
    assert result is None
