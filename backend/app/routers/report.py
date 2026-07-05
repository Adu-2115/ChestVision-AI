import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

from app.config import REPORTS_DIR

router = APIRouter()

os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_pdf(report: dict, save_path: str):
    """Generate a styled PDF report using ReportLab."""
    doc  = SimpleDocTemplate(save_path, pagesize=A4,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor('#1a5276'),
        spaceAfter=6
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'],
        fontSize=12, textColor=colors.HexColor('#2874a6'),
        spaceBefore=12, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=6
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=colors.red,
        leading=12, spaceAfter=6
    )

    story = []

    # ── Header ────────────────────────────────────────────
    story.append(Paragraph('ChestVision AI', title_style))
    story.append(Paragraph('Chest X-Ray Analysis Report', styles['Heading2']))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#2874a6')))
    story.append(Spacer(1, 0.1*inch))

    # ── Report metadata ───────────────────────────────────
    meta_data = [
        ['Report ID',   report['report_id']],
        ['Generated',   report['generated_at']],
        ['Patient ID',  report['patient_id']],
        ['Image',       report['image_filename']],
    ]
    meta_table = Table(meta_data, colWidths=[1.5*inch, 4.5*inch])
    meta_table.setStyle(TableStyle([
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 0), (0, -1), colors.HexColor('#2874a6')),
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.15*inch))

    # ── Predictions table ─────────────────────────────────
    story.append(Paragraph('Predictions', heading_style))
    pred_rows = [['Disease', 'Confidence', 'Status']]
    for p in report['all_predictions']:
        pct    = f"{p['probability']*100:.1f}%"
        status = 'POSITIVE' if p['probability'] >= 0.5 else 'negative'
        color  = colors.red if p['probability'] >= 0.5 else colors.grey
        pred_rows.append([p['disease'], pct, status])

    pred_table = Table(pred_rows, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#2874a6')),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#eaf4fb'), colors.white]),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ALIGN',       (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',  (0, 0), (-1, -1), 6),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.1*inch))

    # ── Findings ──────────────────────────────────────────
    story.append(Paragraph('Findings', heading_style))
    story.append(Paragraph(report['findings'], body_style))

    # ── Impression ────────────────────────────────────────
    story.append(Paragraph('Impression', heading_style))
    story.append(Paragraph(report['impression'], body_style))

    # ── Recommendations ───────────────────────────────────
    story.append(Paragraph('Recommendations', heading_style))
    for rec in report['recommendations']:
        story.append(Paragraph(f"• {rec}", body_style))

    # ── Disease details ───────────────────────────────────
    if report['disease_details']:
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph('Disease Information', heading_style))
        for detail in report['disease_details']:
            story.append(Paragraph(
                f"<b>{detail['disease']}</b> — {detail['description']}",
                body_style
            ))
            story.append(Paragraph(
                f"<b>Recommended Specialist:</b> {detail['specialist']}",
                body_style
            ))

    # ── Disclaimer ────────────────────────────────────────
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.red))
    story.append(Paragraph(report['disclaimer'], disclaimer_style))

    doc.build(story)


@router.get('/report/{scan_id}')
async def download_report(scan_id: str):
    """Generate and download PDF report for a given scan_id."""
    # In production this would fetch report data from DB by scan_id
    # For now raise a clear error
    raise HTTPException(
        status_code=404,
        detail='Report not found. Use /api/predict first to generate a report.'
    )


@router.post('/report/generate')
async def generate_report(report: dict):
    """Generate PDF from report dict and return as download."""
    scan_id   = report.get('report_id', 'unknown')
    save_path = os.path.join(REPORTS_DIR, f"{scan_id}.pdf")

    try:
        generate_pdf(report, save_path)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"PDF generation error: {str(e)}")

    return FileResponse(
        path=save_path,
        media_type='application/pdf',
        filename=f"ChestVision_Report_{scan_id}.pdf"
    )