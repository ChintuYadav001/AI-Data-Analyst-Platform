"""
core.bi.bi_tools
================
Controlled Business Intelligence Calculation Tools for Quantum AI Assistant.
Exposes certified, deterministic analytical routines:
- analyze_business_health()
- detect_business_risks()
- detect_business_opportunities()
- generate_recommendations()
- calculate_priority()
- run_scenario()
- compare_scenarios()
- generate_alerts()
- get_department_analysis()
- generate_executive_summary()
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from core.bi.command_center import calculate_business_health_score, generate_executive_command_center
from core.bi.recommendations import generate_business_recommendations, detect_business_opportunities, calculate_priority_score
from core.bi.scenario_simulator import simulate_business_scenario, compare_business_scenarios
from core.bi.alert_engine import generate_business_alerts
from core.bi.department_analysts import get_department_analysis, DEPARTMENT_CONFIGS


def analyze_business_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes 0-100 Business Health Score and dimensional breakdown."""
    res = calculate_business_health_score(df)
    return {
        "overall_score": res["overall_score"],
        "grade": res["grade"],
        "dimensions": res["dimensions"],
        "summary": f"Overall Business Health Score is {res['overall_score']:.0f}/100 ({res['grade']}).",
    }


def detect_business_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identifies active business risk alerts and vulnerabilities."""
    alerts = generate_business_alerts(df)
    risks = [a for a in alerts if a.get("severity") in ("CRITICAL", "HIGH")]
    return risks or alerts[:3]


def generate_recommendations(df: pd.DataFrame, dataset_name: str = "Active Dataset") -> List[Dict[str, Any]]:
    """Generates prioritized, actionable business recommendations."""
    return generate_business_recommendations(df, dataset_name=dataset_name)


def run_scenario(df: pd.DataFrame, modifications: Dict[str, float]) -> Dict[str, Any]:
    """Runs a What-If business decision simulation."""
    return simulate_business_scenario(df, modifications)


def compare_scenarios(df: pd.DataFrame) -> Dict[str, Any]:
    """Compares Baseline vs Conservative vs Aggressive vs Defensive scenarios."""
    return compare_business_scenarios(df)


def generate_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generates multi-type business alerts."""
    return generate_business_alerts(df)


def generate_executive_summary(df: pd.DataFrame, dataset_name: str = "Active Dataset") -> Dict[str, Any]:
    """Generates board-ready executive summary, health scores, and positive/risk signals."""
    cc = generate_executive_command_center(df, dataset_name=dataset_name)
    return cc.get("summary", {})
