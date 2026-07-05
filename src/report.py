import os
import sys
from datetime import datetime
from groq import Groq

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


def generate_llm_report(predictions: list, findings: str,
                         patient_age: float = 60.0,
                         patient_sex: str = 'Unknown',
                         spatial_data: dict = None) -> dict:
    """
    Generate report using Groq LLaMA3-70B.
    Now includes Grad-CAM spatial activation data for anatomically
    precise report generation.

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

    disease_summary = "\n".join([
        f"- {p['disease']}: {p['probability']*100:.1f}% confidence"
        for p in predictions
    ])

    positive_summary = "\n".join([
        f"- {p['disease']} ({p['probability']*100:.1f}%): "
        f"{DISEASE_INFO[p['disease']]['findings']} "
        f"Region: {DISEASE_INFO[p['disease']]['region']}"
        for p in positives
    ])

    specialists = list({DISEASE_INFO[p['disease']]['specialist'] for p in positives})

    # Patient context
    sex_display     = patient_sex if patient_sex != 'Unknown' else 'Unknown sex'
    patient_context = f"{sex_display}, {int(patient_age)} years old"

    # Build Grad-CAM spatial context
    spatial_context = ""
    if spatial_data:
        spatial_lines = []
        for disease, spatial in spatial_data.items():
            if isinstance(spatial, dict) and 'description' in spatial:
                spatial_lines.append(f"- {spatial['description']}")

                # Add laterality info if available
                if spatial.get('laterality'):
                    spatial_lines.append(
                        f"  Laterality: {spatial['laterality']}"
                    )

                # Add top activated regions
                if spatial.get('activated_regions'):
                    top_regions = spatial['activated_regions'][:3]
                    spatial_lines.append(
                        f"  Primary zones: {', '.join(top_regions)}"
                    )

        if spatial_lines:
            spatial_context = (
                "GRAD-CAM SPATIAL ACTIVATION DATA:\n"
                "(These show which exact anatomical regions the AI focused on)\n"
                + "\n".join(spatial_lines)
            )

    prompt = f"""You are a senior radiologist with 20 years of experience reviewing a chest X-ray AI analysis.
Your role is to provide a detailed, insightful preliminary report that will help clinicians understand
the significance of the findings. This report will be reviewed by a qualified radiologist before clinical use.

PATIENT INFORMATION:
{patient_context}

AI MODEL PREDICTIONS:
{disease_summary}

POSITIVE FINDINGS (confidence >50%):
{positive_summary}

{spatial_context}

IMPORTANT: Use the Grad-CAM spatial activation data above to describe EXACTLY which lung zones
are affected. For example, if Pleural Effusion shows activation in lower lung zones, mention
"blunting of the costophrenic angles in the lower zones" rather than a generic description.
This makes the report anatomically precise and clinically useful.

Generate a detailed radiology report with these exact sections:

FINDINGS:
Describe each positive finding using the spatial activation data above.
For each finding:
- Reference the specific anatomical zones highlighted by Grad-CAM
- Describe the laterality (bilateral, left-sided, right-sided)
- Explain the clinical significance of that specific location
- Relate findings to each other and to patient demographics
Write 3-4 sentences per finding.

DIFFERENTIAL DIAGNOSIS:
List 3-4 possible underlying conditions explaining the combination of findings.
For each condition explain why the specific anatomical distribution supports it.
Consider the patient age and sex in your differential.
Format as numbered list.

IMPRESSION:
3-4 sentence clinical summary:
- State the most likely diagnosis considering imaging distribution and demographics
- Explain clinical significance and urgency
- Note findings requiring immediate attention
- Recommend the most important next step

RECOMMENDATIONS:
Specific actionable recommendations:
- Urgency level (Routine / Soon / Urgent / Emergency)
- Specialist referrals with specific reason (refer to {', '.join(specialists)})
- Follow-up investigations appropriate for this patient age and sex
- Clinical correlation points

Important formatting rules:
- Do not use markdown formatting like ** or ## anywhere
- Use plain text only
- Each section heading on its own line in capitals

End with:
DISCLAIMER: This report is AI-generated and must be verified by a qualified radiologist."""

    try:
        client   = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 900,
            temperature = 0.2,
        )

        content = response.choices[0].message.content.strip()

        findings_text       = _extract_section(content, 'FINDINGS:', 'DIFFERENTIAL DIAGNOSIS:')
        differential_text   = _extract_section(content, 'DIFFERENTIAL DIAGNOSIS:', 'IMPRESSION:')
        impression_text     = _extract_section(content, 'IMPRESSION:', 'RECOMMENDATIONS:')
        recommendations_raw = _extract_section(content, 'RECOMMENDATIONS:', 'DISCLAIMER:')

        if not findings_text:
            findings_text = _extract_section(content, 'FINDINGS:', 'IMPRESSION:')

        recommendations = [
            line.strip().lstrip('•-*0123456789.').strip()
            for line in recommendations_raw.split('\n')
            if line.strip() and len(line.strip()) > 10
        ]

        if findings_text and impression_text:
            print("LLM report generation successful (with spatial data)")
            return {
                'findings':        findings_text.strip(),
                'differential':    differential_text.strip() if differential_text else '',
                'impression':      impression_text.strip(),
                'recommendations': recommendations,
                'llm_generated':   True
            }

    except Exception as e:
        print(f"LLM generation failed: {e}, falling back to rule-based")

    return None


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    try:
        start = text.index(start_marker) + len(start_marker)
        end   = text.index(end_marker)
        return text[start:end].strip()
    except ValueError:
        return ''


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
        lines.append(
            f"{p['disease']} ({p['probability']*100:.1f}%): "
            f"{info['findings']} Region: {info['region']}."
        )
    return ' '.join(lines)


def _build_rule_impression(positives: list) -> str:
    if not positives:
        return 'No significant findings detected. Normal chest radiograph.'
    if len(positives) == 1:
        d    = positives[0]['disease']
        prob = positives[0]['probability'] * 100
        return (
            f"Findings are suggestive of {d} ({prob:.1f}% confidence). "
            f"{DISEASE_INFO[d]['description']} "
            f"Clinical correlation is recommended."
        )
    disease_list = ', '.join(
        [f"{p['disease']} ({p['probability']*100:.1f}%)" for p in positives]
    )
    primary   = positives[0]
    secondary = positives[1:]
    sec_names = ' and '.join([p['disease'] for p in secondary])
    return (
        f"Findings are suggestive of {primary['disease']} "
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