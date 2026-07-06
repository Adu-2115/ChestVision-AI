"""
Tests for EnsemblePredictor.aggregate_scores — the averaging/disagreement
math, tested in isolation without loading any actual models.

Save this as: backend/tests/test_ensemble_aggregation.py
(adjust the import path below to match where ensemble.py actually lives
relative to your test runner's working directory / PYTHONPATH)
"""
import pytest
from src.models.ensemble import EnsemblePredictor


def test_all_three_models_agree():
    scores = {'efficientnet_b0': 0.80, 'mobilenet_v2': 0.82, 'torchxrayvision': 0.81}
    result = EnsemblePredictor.aggregate_scores('Edema', scores, threshold=0.5)

    assert result['disease'] == 'Edema'
    assert result['probability'] == pytest.approx(0.81, abs=0.01)
    assert result['positive'] is True
    assert result['n_models_used'] == 3
    assert result['disagreement'] == pytest.approx(0.02, abs=0.001)


def test_models_disagree_significantly():
    scores = {'efficientnet_b0': 0.90, 'mobilenet_v2': 0.30, 'torchxrayvision': 0.60}
    result = EnsemblePredictor.aggregate_scores('Atelectasis', scores, threshold=0.5)

    assert result['probability'] == pytest.approx(0.60, abs=0.01)
    assert result['disagreement'] == pytest.approx(0.60, abs=0.01)
    assert result['n_models_used'] == 3


def test_torchxrayvision_does_not_cover_disease():
    """When XRV's checkpoint doesn't cover a disease, its score is None
    and should be excluded from the average rather than treated as 0."""
    scores = {'efficientnet_b0': 0.70, 'mobilenet_v2': 0.74, 'torchxrayvision': None}
    result = EnsemblePredictor.aggregate_scores('Consolidation', scores, threshold=0.5)

    assert result['n_models_used'] == 2
    assert result['probability'] == pytest.approx(0.72, abs=0.01)
    # disagreement should be computed only over the 2 valid scores
    assert result['disagreement'] == pytest.approx(0.04, abs=0.001)


def test_threshold_boundary_exact_match_is_positive():
    """probability exactly equal to threshold should count as positive
    (>=, not >) — matches the convention used throughout the codebase."""
    scores = {'efficientnet_b0': 0.5, 'mobilenet_v2': 0.5, 'torchxrayvision': 0.5}
    result = EnsemblePredictor.aggregate_scores('Cardiomegaly', scores, threshold=0.5)
    assert result['positive'] is True


def test_no_valid_scores_raises():
    """If every model returns None for a disease, that's a real bug
    upstream (shouldn't happen given DISEASE_COLS is fixed), and should
    fail loudly rather than silently returning a fake probability."""
    scores = {'efficientnet_b0': None, 'mobilenet_v2': None, 'torchxrayvision': None}
    with pytest.raises(ValueError):
        EnsemblePredictor.aggregate_scores('Edema', scores)


def test_disagreement_rounds_to_four_decimals():
    scores = {'efficientnet_b0': 0.123456, 'mobilenet_v2': 0.654321, 'torchxrayvision': 0.5}
    result = EnsemblePredictor.aggregate_scores('Edema', scores)
    # max - min = 0.654321 - 0.123456 = 0.530865, rounded to 4 decimals
    assert result['disagreement'] == round(0.654321 - 0.123456, 4)
