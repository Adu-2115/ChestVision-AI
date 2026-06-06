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


def generate_llm_report(predictions: list, findings: str) -> dict:
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

    prompt = f"""You are a senior radiologist with 20 years of experience reviewing a chest X-ray AI analysis. 
        Your role is to provide a detailed, insightful preliminary report that will help clinicians understand 
        the significance of the findings. This report will be reviewed by a qualified radiologist before clinical use.

        AI MODEL PREDICTIONS:
        {disease_summary}

        POSITIVE FINDINGS (confidence >50%):
        {positive_summary}

        Generate a detailed radiology report with these exact sections:

        FINDINGS:
        Describe each positive finding in detail. For each finding explain:
        - The specific radiological features observed
        - The anatomical location and extent
        - How confident the AI model is and what this means clinically
        - How the findings relate to each other (e.g., pleural effusion often accompanies heart failure)
        Write 3-4 sentences per finding.

        DIFFERENTIAL DIAGNOSIS:
        List 3-4 possible underlying conditions that could explain the combination of findings.
        For each condition explain why these findings support or suggest it.
        Format as numbered list.

        IMPRESSION:
        Provide a 3-4 sentence clinical summary that:
        - States the most likely overall diagnosis or clinical picture
        - Explains the clinical significance and potential urgency
        - Notes any findings that require immediate attention
        - Recommends the most important next step

        RECOMMENDATIONS:
        Provide specific actionable recommendations:
        - Urgency level (Routine / Soon / Urgent / Emergency)
        - Specific specialist referrals with reason why (refer to {', '.join(specialists)})
        - Specific follow-up investigations (blood tests, echo, CT etc.)
        - Clinical correlation points the physician should check

        Important formatting rules:
        - Do not use markdown formatting like ** or ## anywhere
        - Use plain text only
        - Each section heading should be on its own line in capitals

        End with:
        DISCLAIMER: This report is AI-generated and must be verified by a qualified radiologist before any clinical decisions."""

    try:
        client   = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 800,
            temperature = 0.2,
        )

        content = response.choices[0].message.content.strip()

        findings_text       = _extract_section(content, 'FINDINGS:', 'IMPRESSION:')
        impression_text     = _extract_section(content, 'IMPRESSION:', 'RECOMMENDATIONS:')
        recommendations_raw = _extract_section(content, 'RECOMMENDATIONS:', 'DISCLAIMER:')

        recommendations = [
            line.strip().lstrip('•').strip()
            for line in recommendations_raw.split('\n')
            if line.strip() and line.strip().startswith('•')
        ]

        if not recommendations:
            recommendations = [
                line.strip()
                for line in recommendations_raw.split('\n')
                if line.strip() and len(line.strip()) > 10
            ]

        if findings_text and impression_text:
            print("LLM report generation successful")
            return {
                'findings':        findings_text.strip(),
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
                 patient_id='N/A', threshold=0.5):
    now       = datetime.now()
    positives = [p for p in predictions if p['probability'] >= threshold]

    findings_text = _build_rule_findings(positives)
    llm_result    = generate_llm_report(predictions, findings_text)

    if llm_result:
        findings_text   = llm_result['findings']
        impression      = llm_result['impression']
        recommendations = llm_result['recommendations']
        llm_generated   = True
    else:
        findings_text   = _build_rule_findings(positives)
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
        'image_filename':    image_filename,
        'llm_generated':     llm_generated,
        'all_predictions':   [
            {'disease': p['disease'], 'probability': round(p['probability'], 4)}
            for p in predictions
        ],
        'positive_findings': positives,
        'findings':          findings_text,
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
    llm_badge = 'AI-Generated (LLaMA3-70B)' if report.get('llm_generated') else 'Template-Based'

    lines = [
        sep,
        'CHESTVISION AI CHEST X-RAY ANALYSIS REPORT',
        sep,
        f"Report ID    : {report['report_id']}",
        f"Generated    : {report['generated_at']}",
        f"Patient ID   : {report['patient_id']}",
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
        lines.append(f"  {status:10s} {p['disease']:20s} {p['probability']*100:5.1f}%  {bar}")

    lines += ['', 'FINDINGS', sep2, report['findings'],
              '', 'IMPRESSION', sep2, report['impression'],
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