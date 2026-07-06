"""
Tests for the rule-based fallback report builders in src/report.py —
these run whenever the Groq LLM call fails or GROQ_API_KEY isn't set,
so they need to be correct on their own.

Save this as: backend/tests/test_report_rules.py
"""
from src.report import (
    _build_rule_findings,
    _build_rule_impression,
    _build_rule_recommendations,
)


def make_prediction(disease, probability):
    return {'disease': disease, 'probability': probability, 'positive': probability >= 0.5}


def test_no_findings_when_all_negative():
    findings = _build_rule_findings([])
    assert 'no significant' in findings.lower()


def test_single_positive_finding_impression():
    positives = [make_prediction('Edema', 0.82)]
    impression = _build_rule_impression(positives)
    assert 'Edema' in impression
    assert '82.0%' in impression


def test_multiple_positive_findings_impression_mentions_all():
    positives = [
        make_prediction('Pleural Effusion', 0.90),
        make_prediction('Atelectasis', 0.65),
        make_prediction('Edema', 0.55),
    ]
    impression = _build_rule_impression(positives)
    assert 'Pleural Effusion' in impression
    assert 'Atelectasis' in impression
    assert 'Edema' in impression
    assert 'urgent' in impression.lower()


def test_recommendations_empty_when_no_findings():
    recs = _build_rule_recommendations([])
    assert len(recs) > 0
    assert any('routine' in r.lower() for r in recs)


def test_recommendations_include_correct_specialist():
    positives = [make_prediction('Cardiomegaly', 0.88)]
    recs = _build_rule_recommendations(positives)
    assert any('Cardiologist' in r for r in recs)
