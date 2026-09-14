"""
core.bi
=======
AI-Powered Business Intelligence & Decision Intelligence Suite for Quantum Analyst.
Core Principle: "Data engines calculate. ML models predict. AI explains. Business intelligence recommends. Users decide."
"""

from core.bi.domain_detector import detect_business_domain, map_semantic_columns, get_domain_context
from core.bi.recommendations import generate_business_recommendations, detect_business_opportunities, calculate_priority_score
from core.bi.scenario_simulator import simulate_business_scenario, compare_business_scenarios, compare_multiple_scenarios, get_dataset_baseline_metrics
from core.bi.alert_engine import generate_business_alerts
from core.bi.department_analysts import get_department_analysis, DEPARTMENT_CONFIGS
from core.bi.command_center import calculate_business_health_score, generate_executive_command_center
from core.bi.business_report import (
    generate_executive_business_pdf_report,
    generate_executive_business_html_report,
    generate_executive_business_csv_summary,
    generate_business_report_dossier,
)
import core.bi.bi_tools as bi_tools

__all__ = [
    "detect_business_domain",
    "map_semantic_columns",
    "get_domain_context",
    "generate_business_recommendations",
    "detect_business_opportunities",
    "calculate_priority_score",
    "simulate_business_scenario",
    "compare_business_scenarios",
    "compare_multiple_scenarios",
    "get_dataset_baseline_metrics",
    "generate_business_alerts",
    "get_department_analysis",
    "DEPARTMENT_CONFIGS",
    "calculate_business_health_score",
    "generate_executive_command_center",
    "generate_executive_business_pdf_report",
    "generate_executive_business_html_report",
    "generate_executive_business_csv_summary",
    "generate_business_report_dossier",
    "bi_tools",
]
