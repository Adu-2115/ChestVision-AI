import os
import sys
from datetime import datetime
from src.dataset import DISEASE_COLS

# ── Disease knowledge base ─────────────────────────────────
DISEASE_INFO = {
    'Atelectasis': {
        'description': 'Partial or complete collapse of lung tissue, reducing gas exchange.',
        'findings':    'Increased opacity with volume loss in the affected lung region.',
        'region':      'Lung bases or lower lobes',
        'symptoms':    ['shortness of breath', 'decreased breath sounds', 'chest pain'],
        'causes':      ['airway obstruction', 'post-surgical complication', 'prolonged bed rest'],
        'specialist':  'Pulmonologist',
    },
    'Cardiomegaly': {
        'description': 'Enlargement of the cardiac silhouette beyond normal limits.',
        'findings':    'Cardiothoracic ratio greater than 0.5 on PA view.',
        'region':      'Cardiac silhouette',
        'symptoms':    ['breathlessness', 'fatigue', 'peripheral edema'],
        'causes':      ['heart failure', 'hypertension', 'cardiomyopathy'],
        'specialist':  'Cardiologist',
    },
    'Consolidation': {
        'description': 'Replacement of airspaces with fluid, cells, or other material.',
        'findings':    'Homogeneous opacification with air bronchograms.',
        'region':      'Lung parenchyma',
        'symptoms':    ['productive cough', 'fever', 'pleuritic chest pain'],
        'causes':      ['bacterial pneumonia', 'aspiration', 'pulmonary infarction'],
        'specialist':  'Pulmonologist / Infectious Disease',
    },
    'Edema': {
        'description': 'Accumulation of fluid in the lung interstitium and alveoli.',
        'findings':    'Bilateral perihilar haziness with Kerley B lines.',
        'region':      'Bilateral lung fields, perihilar distribution',
        'symptoms':    ['acute breathlessness', 'orthopnea', 'pink frothy sputum'],
        'causes':      ['congestive heart failure', 'fluid overload', 'ARDS'],
        'specialist':  'Cardiologist / Intensivist',
    },
    'Pleural Effusion': {
        'description': 'Abnormal accumulation of fluid in the pleural space.',
        'findings':    'Blunting of costophrenic angles with meniscus sign.',
        'region':      'Pleural space, lower lung zones',
        'symptoms':    ['pleuritic chest pain', 'dyspnea', 'dry cough'],
        'causes':      ['heart failure', 'malignancy', 'infection', 'liver cirrhosis'],
        'specialist':  'Pulmonologist',
    },
}


def build_report(predictions, image_filename='uploaded_image.jpg',
                 patient_id='N/A', threshold=0.5):
    """
    Build a structured radiology-style report from model predictions.

    Args:
        predictions : list of dicts from gradcam.predict()
        image_filename : original filename of the X-ray
        patient_id : optional patient identifier
        threshold  : confidence threshold for positive finding

    Returns:
        dict with all report sections
    """
    now       = datetime.now()
    positives = [p for p in predictions if p['probability'] >= threshold]
    all_probs = {p['disease']: p['probability'] for p in predictions}

    # ── Findings section ──────────────────────────────────
    if positives:
        finding_lines = []
        for p in positives:
            info = DISEASE_INFO[p['disease']]
            finding_lines.append(
                f"{p['disease']} ({p['probability']*100:.1f}%): "
                f"{info['findings']} Region: {info['region']}."
            )
        findings_text = ' '.join(finding_lines)
    else:
        findings_text = (
            'No significant acute cardiopulmonary abnormality detected. '
            'Lung fields appear clear. Cardiac silhouette within normal limits.'
        )

    # ── Impression section ────────────────────────────────
    if len(positives) == 0:
        impression = 'No significant findings detected. Normal chest radiograph.'

    elif len(positives) == 1:
        d    = positives[0]['disease']
        prob = positives[0]['probability'] * 100
        impression = (
            f"Findings are suggestive of {d} ({prob:.1f}% confidence). "
            f"{DISEASE_INFO[d]['description']} "
            f"Clinical correlation is recommended."
        )

    else:
        disease_list = ', '.join(
            [f"{p['disease']} ({p['probability']*100:.1f}%)" for p in positives]
        )
        primary   = positives[0]
        secondary = positives[1:]
        sec_names = ' and '.join([p['disease'] for p in secondary])
        impression = (
            f"Findings are suggestive of {primary['disease']} "
            f"({primary['probability']*100:.1f}% confidence) "
            f"with associated {sec_names}. "
            f"Multiple concurrent abnormalities identified: {disease_list}. "
            f"Urgent clinical correlation and specialist review advised."
        )

    # ── Recommendations ───────────────────────────────────
    if not positives:
        recommendations = [
            'Routine clinical follow-up as indicated.',
            'Repeat imaging if symptoms persist or worsen.'
        ]
    else:
        specialists = list({DISEASE_INFO[p['disease']]['specialist']
                            for p in positives})
        recommendations = [
            f"Specialist consultation recommended: {', '.join(specialists)}.",
            'Correlate with clinical presentation and laboratory findings.',
            'Consider additional imaging or investigations as clinically indicated.',
            'This report is AI-generated and should be reviewed by a qualified radiologist.'
        ]

    # ── Disease details for knowledge panel ───────────────
    disease_details = []
    for p in positives:
        info = DISEASE_INFO[p['disease']]
        disease_details.append({
            'disease':     p['disease'],
            'probability': p['probability'],
            'description': info['description'],
            'symptoms':    info['symptoms'],
            'causes':      info['causes'],
            'specialist':  info['specialist'],
            'region':      info['region'],
        })

    report = {
        'report_id':       f"CV-{now.strftime('%Y%m%d%H%M%S')}",
        'generated_at':    now.strftime('%Y-%m-%d %H:%M:%S'),
        'patient_id':      patient_id,
        'image_filename':  image_filename,
        'all_predictions': [
            {'disease': p['disease'], 'probability': round(p['probability'], 4)}
            for p in predictions
        ],
        'positive_findings': positives,
        'findings':        findings_text,
        'impression':      impression,
        'recommendations': recommendations,
        'disease_details': disease_details,
        'disclaimer': (
            'IMPORTANT: This report is generated by an AI system (ChestVision AI) '
            'for decision-support purposes only. It is NOT a substitute for '
            'professional medical diagnosis. All findings must be verified by '
            'a qualified radiologist or physician.'
        )
    }

    return report


def format_report_text(report):
    """Format report as plain text for display or printing."""
    sep  = '=' * 60
    sep2 = '-' * 60

    lines = [
        sep,
        'CHESTVISION AI — CHEST X-RAY ANALYSIS REPORT',
        sep,
        f"Report ID    : {report['report_id']}",
        f"Generated    : {report['generated_at']}",
        f"Patient ID   : {report['patient_id']}",
        f"Image        : {report['image_filename']}",
        sep2,
        '',
        'PREDICTIONS',
        sep2,
    ]

    for p in report['all_predictions']:
        bar    = '█' * int(p['probability'] * 20)
        status = '✓ POSITIVE' if p['probability'] >= 0.5 else '  negative'
        lines.append(
            f"  {status}  {p['disease']:20s} "
            f"{p['probability']*100:5.1f}%  {bar}"
        )

    lines += [
        '',
        'FINDINGS',
        sep2,
        report['findings'],
        '',
        'IMPRESSION',
        sep2,
        report['impression'],
        '',
        'RECOMMENDATIONS',
        sep2,
    ]

    for rec in report['recommendations']:
        lines.append(f"  • {rec}")

    lines += [
        '',
        sep,
        report['disclaimer'],
        sep,
    ]

    return '\n'.join(lines)


def save_report_text(report, save_path):
    """Save formatted report to a .txt file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    text = format_report_text(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Report saved to {save_path}")
    return text


if __name__ == '__main__':
    # Test with mock predictions
    mock_predictions = [
        {'disease': 'Pleural Effusion', 'probability': 0.864, 'positive': True},
        {'disease': 'Atelectasis',      'probability': 0.630, 'positive': True},
        {'disease': 'Consolidation',    'probability': 0.531, 'positive': True},
        {'disease': 'Edema',            'probability': 0.513, 'positive': True},
        {'disease': 'Cardiomegaly',     'probability': 0.120, 'positive': False},
    ]

    report = build_report(
        predictions=mock_predictions,
        image_filename='test_xray.jpg',
        patient_id='TEST-001'
    )

    text = format_report_text(report)
    print(text)