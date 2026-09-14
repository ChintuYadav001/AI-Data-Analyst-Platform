"""
core.ml.report_generator
========================
Institutional Machine Learning & Predictive Analytics Report Generator.
Generates comprehensive executive reports in PDF (via ReportLab), standalone HTML,
and tabular CSV/Excel formats covering dataset readiness, model leaderboards,
explainability, forecasting projections, and governance disclosures.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


def generate_ml_pdf_report(report_data: Dict[str, Any]) -> bytes:
    """Generates an institutional PDF report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=14,
        )
        h2_style = ParagraphStyle(
            "ReportH2",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
        body_bold = ParagraphStyle(
            "ReportBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold",
        )

        story = []

        # Header Title
        story.append(Paragraph("QUANTUM ANALYST — ML & PREDICTIVE ANALYTICS AUDIT", title_style))
        proj_name = report_data.get("dataset_name", "Active Dataset")
        gen_time = datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
        story.append(Paragraph(f"Institutional Audit & Modeling Report | Dataset: <b>{proj_name}</b> | Generated: {gen_time}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=14))

        # 1. Executive Summary Table
        story.append(Paragraph("1. Executive Summary & Telemetry", h2_style))
        summary_rows = [
            [Paragraph("<b>Target Variable</b>", body_style), Paragraph(str(report_data.get("target_col", "N/A")), body_style),
             Paragraph("<b>Task Classification</b>", body_style), Paragraph(str(report_data.get("task_type", "N/A")).replace("_", " ").title(), body_style)],
            [Paragraph("<b>Champion Model</b>", body_style), Paragraph(str(report_data.get("champion_name", "N/A")), body_bold),
             Paragraph("<b>Readiness Score</b>", body_style), Paragraph(f"{report_data.get('readiness_score', 'N/A')}/100", body_bold)],
            [Paragraph("<b>Total Samples</b>", body_style), Paragraph(f"{report_data.get('total_rows', 'N/A'):,}" if isinstance(report_data.get('total_rows'), (int, float)) else str(report_data.get('total_rows')), body_style),
             Paragraph("<b>Feature Count</b>", body_style), Paragraph(str(report_data.get("feature_count", "N/A")), body_style)],
        ]
        t_sum = Table(summary_rows, colWidths=[120, 150, 120, 150])
        t_sum.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_sum)
        story.append(Spacer(1, 10))

        # 2. Benchmark Leaderboard Table
        story.append(Paragraph("2. Model Benchmark Leaderboard", h2_style))
        leaderboard = report_data.get("leaderboard", [])
        if leaderboard:
            t_data = [[Paragraph("<b>Model Name</b>", body_style), Paragraph("<b>Primary Metric</b>", body_style), Paragraph("<b>Status</b>", body_style)]]
            for row in leaderboard[:6]:
                m_name = row.get("name", row.get("model_id", "Estimator"))
                p_metric = row.get("primary_metric_val", "N/A")
                is_champ = row.get("is_champion", False)
                status_text = "<font color='#0284C7'><b>Champion</b></font>" if is_champ else "Evaluated"
                t_data.append([
                    Paragraph(str(m_name), body_style),
                    Paragraph(f"{p_metric:.4f}" if isinstance(p_metric, (int, float)) else str(p_metric), body_style),
                    Paragraph(status_text, body_style),
                ])
            t_lb = Table(t_data, colWidths=[240, 150, 150])
            t_lb.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_lb)
        else:
            story.append(Paragraph("No multi-model benchmark results registered.", body_style))
        story.append(Spacer(1, 10))

        # 3. Top Predictive Drivers (Explainable AI)
        story.append(Paragraph("3. Key Predictive Drivers & Attribution", h2_style))
        feat_imp = report_data.get("feature_importance", [])
        if feat_imp:
            fi_data = [[Paragraph("<b>Rank</b>", body_style), Paragraph("<b>Feature Name</b>", body_style), Paragraph("<b>Importance Weight</b>", body_style)]]
            for f in feat_imp[:5]:
                fi_data.append([
                    Paragraph(str(f.get("rank", 1)), body_style),
                    Paragraph(str(f.get("feature", "")), body_style),
                    Paragraph(f"{f.get('importance', 0.0):.4f}", body_style),
                ])
            t_fi = Table(fi_data, colWidths=[60, 330, 150])
            t_fi.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_fi)
        else:
            story.append(Paragraph("Feature attribution will populate following model training.", body_style))
        story.append(Spacer(1, 10))

        # 4. Future Forecasting Summary (if available)
        forecast_res = report_data.get("forecast_results")
        if forecast_res:
            story.append(Paragraph("4. Time-Series Forecasting Trajectory", h2_style))
            f_text = (
                f"Projected {forecast_res.get('horizon', 14)} future periods using "
                f"<b>{forecast_res.get('best_model', 'Forecasting Model')}</b>. "
                f"Projected Growth: <b>{forecast_res.get('metrics', {}).get('projected_growth_pct', 0.0):+.1f}%</b>. "
                f"Backtest MAE: <b>{forecast_res.get('metrics', {}).get('mae', 'N/A')}</b>."
            )
            story.append(Paragraph(f_text, body_style))
            story.append(Spacer(1, 10))

        # 5. Regulatory Disclosures & Compliance
        story.append(Paragraph("5. Model Governance, Limitations & Disclaimers", h2_style))
        disclaimer_text = (
            "<b>STATISTICAL ASSOCIATION NOTICE</b>: Features identified as top predictors represent statistical correlation "
            "and predictive association within the observed historical sample. They do not prove causal relationships. "
            "Model predictions and future forecasts are estimates based on historical probability distributions and must "
            "be reviewed by domain decision-makers prior to execution."
        )
        story.append(Paragraph(disclaimer_text, body_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        # Fallback to plain byte text if reportlab encounters rendering issues
        fallback_str = f"Quantum Analyst ML Report\nGenerated: {datetime.now()}\nError generating PDF: {str(e)}"
        return fallback_str.encode("utf-8")


def generate_ml_html_report(report_data: Dict[str, Any]) -> str:
    """Generates a standalone dark-themed HTML report."""
    proj_name = report_data.get("dataset_name", "Active Dataset")
    gen_time = datetime.now().strftime("%B %d, %Y %H:%M:%S UTC")
    champ_name = report_data.get("champion_name", "Trained Estimator")
    target_col = report_data.get("target_col", "N/A")
    task_type = str(report_data.get("task_type", "N/A")).replace("_", " ").title()
    readiness_score = report_data.get("readiness_score", "N/A")

    leaderboard_html = ""
    for r in report_data.get("leaderboard", [])[:8]:
        m_name = r.get("name", r.get("model_id", "Model"))
        p_val = r.get("primary_metric_val", "N/A")
        val_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
        is_c = r.get("is_champion", False)
        badge = '<span style="color: #00F0FF; font-weight: bold;">[CHAMPION]</span>' if is_c else ""
        leaderboard_html += f"<tr><td style='padding: 8px; border-bottom: 1px solid #334155;'>{m_name} {badge}</td><td style='padding: 8px; border-bottom: 1px solid #334155;'>{val_str}</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Quantum Analyst ML Audit Report - {proj_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0B0F19; color: #E2E8F0; padding: 40px; margin: 0; }}
        .card {{ background: #131B2E; border: 1px solid #1E293B; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        h1 {{ color: #00F0FF; margin-top: 0; font-size: 24px; }}
        h2 {{ color: #38BDF8; font-size: 18px; border-bottom: 1px solid #1E293B; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #1E293B; color: #94A3B8; padding: 8px; }}
        .disclaimer {{ background: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; padding: 12px; border-radius: 4px; font-size: 13px; color: #FCD34D; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>QUANTUM ANALYST — ML & PREDICTIVE ANALYTICS AUDIT REPORT</h1>
        <p style="color: #94A3B8; margin-bottom: 0;">Dataset: <b>{proj_name}</b> | Generated: {gen_time}</p>
    </div>

    <div class="card">
        <h2>1. Executive Telemetry</h2>
        <p><b>Target Variable:</b> <code>{target_col}</code> | <b>Task Formulation:</b> {task_type}</p>
        <p><b>Champion Model:</b> <span style="color: #00F0FF; font-weight: bold;">{champ_name}</span> | <b>Readiness Score:</b> {readiness_score}/100</p>
    </div>

    <div class="card">
        <h2>2. Model Benchmark Leaderboard</h2>
        <table>
            <thead><tr><th>Model</th><th>Primary Validation Metric</th></tr></thead>
            <tbody>
                {leaderboard_html if leaderboard_html else "<tr><td colspan='2'>No benchmark records available.</td></tr>"}
            </tbody>
        </table>
    </div>

    <div class="card disclaimer">
        <b>STATISTICAL ASSOCIATION NOTICE:</b> All features, models, and predictions represent algorithmic associations grounded in historical data. Correlation does not imply causation. Decisions must be verified by domain authorities.
    </div>
</body>
</html>
"""
    return html


def generate_ml_csv_summary(report_data: Dict[str, Any]) -> str:
    """Generates a tabular CSV summary of benchmark models and metrics."""
    lb = report_data.get("leaderboard", [])
    if not lb:
        return "model_id,name,primary_metric_val,is_champion\n"

    df_lb = pd.DataFrame(lb)
    cols_to_export = [c for c in ["model_id", "name", "primary_metric_val", "is_champion", "speed"] if c in df_lb.columns]
    return df_lb[cols_to_export].to_csv(index=False)
