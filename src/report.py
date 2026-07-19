import os
import sys
import json
from datetime import datetime
from groq import Groq

if os.name == 'nt':
    sys.path.append(r'D:\Projects\ChestVision-AI')

from src.dataset import DISEASE_COLS

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

# Confidence bands -> permitted certainty language in the report.
# Tied to the temperature-scaled (calibrated) probabilities.
TIER_CONSISTENT = 'consistent with'
TIER_SUGGESTIVE = 'suggestive of'
TIER_BORDERLINE = 'possible / borderline'
TIER_NEGATIVE   = 'no significant evidence of'

DISAGREEMENT_THRESHOLD = 0.30


def _confidence_tier(prob: float) -> str:
    """Map a calibrated probability to the certainty language it permits."""
    if prob >= 0.80:
        return TIER_CONSISTENT
    if prob >= 0.65:
        return TIER_SUGGESTIVE
    if prob >= 0.50:
        return TIER_BORDERLINE
    return TIER_NEGATIVE


def _build_prediction_block(predictions: list, spatial_data: dict = None) -> str:
    """
    Render every disease as one structured line: score, confidence tier,
    inter-model disagreement, and — for positives with spatial data — the
    Grad-CAM zones and laterality. Negatives are included so the model can
    state pertinent negatives.
    """
    lines = []
    for p in predictions:
        prob = p['probability']
        line = f"{p['disease']}: {prob * 100:.1f}% [{_confidence_tier(prob)}]"

        disagreement = p.get('disagreement')
        if disagreement is not None:
            flag = ('  <-- MODELS DISAGREE, express uncertainty'
                    if disagreement > DISAGREEMENT_THRESHOLD else '')
            line += f" | model_disagreement={disagreement:.2f}{flag}"

        if prob >= 0.50 and spatial_data and p['disease'] in spatial_data:
            sp = spatial_data[p['disease']]
            if isinstance(sp, dict):
                zones = ', '.join(sp.get('activated_regions', [])[:3]) or 'diffuse / no focal zone'
                lat   = sp.get('laterality', 'not specified')
                line += f" | gradcam_zones=[{zones}] | laterality={lat}"

        lines.append(line)
    return '\n'.join(lines)


def generate_llm_report(predictions: list, findings: str,
                        patient_age: float = 60.0,
                        patient_sex: str = 'Unknown',
                        spatial_data: dict = None) -> dict:
    """
    Generate a structured radiology report via Groq LLaMA-3.3-70B in JSON mode.

    Grad-CAM spatial data is treated as the only authoritative source of
    spatial/laterality claims; the DISEASE_INFO textbook descriptions are
    passed as reference vocabulary only. Certainty language is constrained by
    calibrated confidence tiers, and inter-model disagreement forces an
    explicit uncertainty statement.

    Returns the same dict shape as before (findings / differential /
    impression / recommendations / llm_generated), or None on any failure so
    build_report() falls back to the rule-based report.

    spatial_data: dict of {disease: spatial_description_dict}
                  from gradcam.get_spatial_description()
    """
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("GROQ_API_KEY not set, using rule-based report.")
        return None

    positives = [p for p in predictions if p['probability'] >= 0.5]
    if not positives:
        return None

    sex_display     = patient_sex if patient_sex != 'Unknown' else 'unspecified sex'
    patient_context = f"{sex_display}, {int(patient_age)} years old"

    prediction_block = _build_prediction_block(predictions, spatial_data)

    reference_lines = []
    for p in positives:
        info = DISEASE_INFO.get(p['disease'], {})
        if info:
            reference_lines.append(
                f"- {p['disease']}: typically presents as \"{info.get('findings', '')}\" "
                f"(reference only - do NOT state as observed unless Grad-CAM supports it)"
            )
    reference_block = '\n'.join(reference_lines) if reference_lines else '(none)'

    specialists = sorted({DISEASE_INFO[p['disease']]['specialist'] for p in positives})

    system_msg = (
        "You are a senior radiologist producing a preliminary, AI-assisted chest "
        "X-ray report that a qualified radiologist will verify before any clinical "
        "use. You are rigorous and never overstate certainty. You respond ONLY with "
        "a single valid JSON object and no other text."
    )

    prompt = f"""Produce a preliminary chest X-ray report from the AI analysis below.

PATIENT: {patient_context}

MODEL PREDICTIONS (all diseases; score, confidence tier, and - for positives - the exact Grad-CAM zones the model attended to):
{prediction_block}

DISEASE REFERENCE VOCABULARY (textbook descriptions - use as vocabulary ONLY, never report these as findings unless the Grad-CAM zones above support them):
{reference_block}

HARD RULES (a violation makes the report clinically unsafe):
1. Describe ONLY diseases scored >=50% as present. Do not introduce any disease not in the list above.
2. Grad-CAM zones are the ONLY source of spatial and laterality claims. If a positive disease has no zones listed, describe it without inventing a location. Never contradict the zones.
3. Match your certainty language to each disease's confidence tier exactly:
   - "{TIER_CONSISTENT}" -> definitive language permitted
   - "{TIER_SUGGESTIVE}" -> hedged language
   - "{TIER_BORDERLINE}" -> explicitly tentative; recommend correlation or repeat imaging
4. If a disease is flagged MODELS DISAGREE, state that the models were not in agreement and that the finding needs confirmation, regardless of its score.
5. State pertinent NEGATIVES: for clinically important diseases scored <50%, briefly note their absence.
6. Ground the differential and impression in the actual spatial distribution and the patient's age and sex.

Return EXACTLY this JSON shape and nothing else (no markdown, no code fences):
{{
  "findings": "2-4 sentences per positive finding, referencing the specific Grad-CAM zones and laterality, with certainty language matched to the tier. Include pertinent negatives in a final sentence.",
  "differential_diagnosis": ["condition 1 with reasoning tied to distribution and demographics", "condition 2 ...", "condition 3 ..."],
  "impression": "3-4 sentence synthesis: most likely picture given distribution and demographics, clinical significance, and the single most urgent next step.",
  "urgency": "one of: Routine | Soon | Urgent | Emergency",
  "recommendations": ["specialist referral among [{', '.join(specialists)}] with reason", "appropriate follow-up imaging or labs for this age and sex", "clinical correlation point"]
}}"""

    try:
        # Explicit timeout - without this, a hung Groq API call would hold this
        # worker indefinitely, which is especially costly now that requests
        # already take longer due to the 3-model ensemble.
        client   = Groq(api_key=groq_api_key, timeout=30.0)
        response = client.chat.completions.create(
            model           = "llama-3.3-70b-versatile",
            messages        = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt},
            ],
            max_tokens      = 1100,
            temperature     = 0.2,
            response_format = {"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        data    = _safe_json_parse(content)
        if data is None:
            print("LLM returned unparseable JSON, falling back to rule-based")
            return None

        findings_text   = (data.get('findings') or '').strip()
        impression_text = (data.get('impression') or '').strip()
        if not findings_text or not impression_text:
            print("LLM JSON missing required fields, falling back to rule-based")
            return None

        differential_list = data.get('differential_diagnosis') or []
        if isinstance(differential_list, list):
            differential_text = '\n'.join(
                f"{i + 1}. {item}"
                for i, item in enumerate(differential_list)
                if str(item).strip()
            )
        else:
            differential_text = str(differential_list).strip()

        recommendations = data.get('recommendations') or []
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        recommendations = [str(r).strip() for r in recommendations if str(r).strip()]

        urgency = (data.get('urgency') or '').strip()
        if urgency:
            recommendations.insert(0, f"Urgency: {urgency}")

        print("LLM report generation successful (JSON mode, calibrated language)")
        return {
            'findings':        findings_text,
            'differential':    differential_text,
            'impression':      impression_text,
            'recommendations': recommendations,
            'llm_generated':   True,
        }

    except Exception as e:
        print(f"LLM generation failed: {e}, falling back to rule-based")

    return None


def _safe_json_parse(content: str):
    """
    Parse JSON that may arrive wrapped in stray prose or code fences.
    Returns a dict, or None if nothing parseable is found.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    cleaned = content.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find('{')
    end   = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def build_report(predictions, image_filename='uploaded_image.jpg',
                 patient_id='N/A', threshold=0.5,
                 patient_age=60.0, patient_sex='Unknown',
                 spatial_data: dict = None):
    """
    Build structured radiology report.
    spatial_data: Grad-CAM spatial descriptions from gradcam.get_spatial_description()
    """
    now       = datetime.now()
    positives = [p for p in predictions if p['probability'] >= threshold]

    findings_text = _build_rule_findings(positives)

    llm_result = generate_llm_report(
        predictions, findings_text,
        patient_age=patient_age,
        patient_sex=patient_sex,
        spatial_data=spatial_data
    )

    if llm_result:
        findings_text   = llm_result['findings']
        differential    = llm_result.get('differential', '')
        impression      = llm_result['impression']
        recommendations = llm_result['recommendations']
        llm_generated   = True
    else:
        findings_text   = _build_rule_findings(positives)
        differential    = ''
        impression      = _build_rule_impression(positives)
        recommendations = _build_rule_recommendations(positives)
        llm_generated   = False

    disclaimer_rec = 'This report is AI-generated and must be reviewed by a qualified radiologist.'
    if disclaimer_rec not in recommendations:
        recommendations.append(disclaimer_rec)

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

    return {
        'report_id':         f"CV-{now.strftime('%Y%m%d%H%M%S')}",
        'generated_at':      now.strftime('%Y-%m-%d %H:%M:%S'),
        'patient_id':        patient_id,
        'patient_age':       patient_age,
        'patient_sex':       patient_sex,
        'image_filename':    image_filename,
        'llm_generated':     llm_generated,
        'all_predictions':   [
            {'disease': p['disease'], 'probability': round(p['probability'], 4)}
            for p in predictions
        ],
        'positive_findings': positives,
        'findings':          findings_text,
        'differential':      differential,
        'impression':        impression,
        'recommendations':   recommendations,
        'disease_details':   disease_details,
        'disclaimer': (
            'IMPORTANT: This report is generated by an AI system (ChestVision AI) '
            'for decision-support purposes only. It is NOT a substitute for '
            'professional medical diagnosis. All findings must be verified by '
            'a qualified radiologist or physician.'
        )
    }


def _build_rule_findings(positives: list) -> str:
    if not positives:
        return (
            'No significant acute cardiopulmonary abnormality detected. '
            'Lung fields appear clear. Cardiac silhouette within normal limits.'
        )
    lines = []
    for p in positives:
        info = DISEASE_INFO[p['disease']]
        tier = _confidence_tier(p['probability'])
        lines.append(
            f"{p['disease']} ({p['probability']*100:.1f}%, {tier}): "
            f"{info['findings']} Region: {info['region']}."
        )
    return ' '.join(lines)


def _build_rule_impression(positives: list) -> str:
    if not positives:
        return 'No significant findings detected. Normal chest radiograph.'
    if len(positives) == 1:
        d    = positives[0]['disease']
        prob = positives[0]['probability']
        tier = _confidence_tier(prob)
        return (
            f"Findings are {tier} {d} ({prob*100:.1f}% confidence). "
            f"{DISEASE_INFO[d]['description']} "
            f"Clinical correlation is recommended."
        )
    disease_list = ', '.join(
        [f"{p['disease']} ({p['probability']*100:.1f}%)" for p in positives]
    )
    primary   = positives[0]
    secondary = positives[1:]
    sec_names = ' and '.join([p['disease'] for p in secondary])
    tier      = _confidence_tier(primary['probability'])
    return (
        f"Findings are {tier} {primary['disease']} "
        f"({primary['probability']*100:.1f}% confidence) "
        f"with associated {sec_names}. "
        f"Multiple concurrent abnormalities: {disease_list}. "
        f"Urgent clinical correlation and specialist review advised."
    )


def _build_rule_recommendations(positives: list) -> list:
    if not positives:
        return [
            'Routine clinical follow-up as indicated.',
            'Repeat imaging if symptoms persist or worsen.'
        ]
    specialists = list({DISEASE_INFO[p['disease']]['specialist'] for p in positives})
    return [
        f"Specialist consultation recommended: {', '.join(specialists)}.",
        'Correlate with clinical presentation and laboratory findings.',
        'Consider additional imaging or investigations as clinically indicated.',
    ]


def format_report_text(report: dict) -> str:
    sep  = '=' * 60
    sep2 = '-' * 60
    llm_badge = 'AI-Generated (LLaMA3-70B + Grad-CAM)' if report.get('llm_generated') else 'Template-Based'

    lines = [
        sep,
        'CHESTVISION AI CHEST X-RAY ANALYSIS REPORT',
        sep,
        f"Report ID    : {report['report_id']}",
        f"Generated    : {report['generated_at']}",
        f"Patient ID   : {report['patient_id']}",
        f"Patient      : {report.get('patient_sex', 'Unknown')}, Age {int(report.get('patient_age', 60))}",
        f"Image        : {report['image_filename']}",
        f"Report Type  : {llm_badge}",
        sep2,
        '',
        'PREDICTIONS',
        sep2,
    ]

    for p in report['all_predictions']:
        bar    = 'X' * int(p['probability'] * 20)
        status = 'POSITIVE' if p['probability'] >= 0.5 else 'negative'
        lines.append(
            f"  {status:10s} {p['disease']:20s} "
            f"{p['probability']*100:5.1f}%  {bar}"
        )

    lines += ['', 'FINDINGS', sep2, report['findings']]

    if report.get('differential'):
        lines += ['', 'DIFFERENTIAL DIAGNOSIS', sep2, report['differential']]

    lines += ['', 'IMPRESSION', sep2, report['impression'],
              '', 'RECOMMENDATIONS', sep2]

    for rec in report['recommendations']:
        lines.append(f"  - {rec}")

    lines += ['', sep, report['disclaimer'], sep]
    return '\n'.join(lines)


def save_report_text(report: dict, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    text = format_report_text(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Report saved to {save_path}")
    return text