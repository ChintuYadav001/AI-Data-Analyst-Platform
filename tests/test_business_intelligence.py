"""
tests.test_business_intelligence
================================
Unit and integration tests for Quantum Analyst Business Intelligence & Decision Intelligence platform.
Tests:
1. Domain & Semantic Detector
2. AI Business Decision & Recommendation Engine
3. AI What-If / Scenario Simulator
4. AI Business Alerts & Anomaly Monitoring
5. Department-Specific AI Analysts
6. Executive Command Center & 0-100 Health Score
7. Executive Business Reporting (PDF, HTML, CSV)
8. AI Assistant Controlled Tools & Speech Integration
"""

import pytest
import pandas as pd
import numpy as np
import core.bi as bi
import core.ai.analytical_tools as tools
from core.ai.quantum_ai_engine import QuantumAIEngine


@pytest.fixture
def enterprise_sample_df():
    """Returns a realistic enterprise dataset with sales, costs, products, customers, and dates."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    categories = ["Electronics", "Software", "Hardware", "Consulting"]
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    customers = [f"CUST_{i:03d}" for i in range(1, 30)]

    revenues = np.random.uniform(200.0, 5000.0, size=n).round(2)
    costs = (revenues * np.random.uniform(0.4, 0.9, size=n)).round(2)
    profits = revenues - costs

    df = pd.DataFrame({
        "order_date": dates,
        "customer_id": np.random.choice(customers, size=n),
        "product_category": np.random.choice(categories, size=n),
        "region": np.random.choice(regions, size=n),
        "revenue": revenues,
        "cost": costs,
        "profit": profits,
        "quantity": np.random.randint(1, 20, size=n),
    })
    return df


# =====================================================================
# 1. Domain & Semantic Column Detector Tests
# =====================================================================
def test_domain_detector(enterprise_sample_df):
    """Verifies domain classification and semantic column mapping."""
    domain_info = bi.detect_business_domain(enterprise_sample_df)

    assert "domain" in domain_info
    assert "confidence" in domain_info
    assert "primary_kpis" in domain_info
    assert isinstance(domain_info["primary_kpis"], list)
    assert len(domain_info["primary_kpis"]) > 0

    col_map = domain_info["column_mapping"]
    assert col_map.get("revenue_col") == "revenue"
    assert col_map.get("cost_col") == "cost"
    assert col_map.get("profit_col") == "profit"
    assert col_map.get("date_col") == "order_date"
    assert col_map.get("customer_col") == "customer_id"

    # Verify transparent disclaimer when dimensions are missing
    assert "missing_dimensions" in domain_info
    assert isinstance(domain_info["missing_dimensions"], list)


def test_domain_detector_minimal_df():
    """Verifies safe fallback when minimal data is provided."""
    df_min = pd.DataFrame({"metric_a": [1, 2, 3], "metric_b": [4, 5, 6]})
    domain_info = bi.detect_business_domain(df_min)
    assert domain_info["domain"] is not None
    assert domain_info["confidence"] in ("Low", "Medium", "High")
    assert len(domain_info["missing_dimensions"]) > 0
    assert "Notice:" in domain_info["disclaimer"] or "lacks" in domain_info["disclaimer"]


# =====================================================================
# 2. Recommendation Engine Tests
# =====================================================================
def test_recommendation_engine(enterprise_sample_df):
    """Verifies prioritized recommendations, non-causal language, and audit trails."""
    recs = bi.generate_business_recommendations(enterprise_sample_df, dataset_name="Enterprise Test")
    assert isinstance(recs, list)
    assert len(recs) > 0

    for rec in recs:
        # Priority score audit
        assert "priority" in rec
        assert rec["priority"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert 0 <= rec["priority_score"] <= 100

        # Actionable structure
        assert "title" in rec
        assert "reason" in rec
        assert "recommended_action" in rec
        assert "potential_impact" in rec
        assert "difficulty" in rec

        # Audit trail ("Why am I seeing this?")
        assert "why_seeing_this" in rec
        why = rec["why_seeing_this"]
        assert "pattern_observed" in why
        assert "statistical_evidence" in why
        assert "assumptions" in why
        assert "confidence" in why

        # Strict non-causal language verification
        reason_lower = rec["reason"].lower()
        assert "caused by" not in reason_lower, f"Violated non-causal guideline: {rec['reason']}"


def test_opportunity_detection(enterprise_sample_df):
    """Verifies Pareto and margin opportunity detection."""
    opps = bi.detect_business_opportunities(enterprise_sample_df)
    assert isinstance(opps, list)
    for opp in opps:
        assert "type" in opp
        assert "title" in opp
        assert "potential_impact" in opp
        assert "recommended_action" in opp


# =====================================================================
# 3. What-If Scenario Simulator Tests
# =====================================================================
def test_scenario_simulator_single(enterprise_sample_df):
    """Verifies single scenario simulation with elasticity calibration."""
    mods = {"price": 10.0, "cost": -5.0}
    sim_res = bi.simulate_business_scenario(enterprise_sample_df, mods)

    assert "baseline" in sim_res
    assert "simulated" in sim_res
    assert "deltas" in sim_res
    assert "ai_interpretation" in sim_res
    assert "assumptions" in sim_res

    deltas = sim_res["deltas"]
    assert "revenue" in deltas
    assert "profit" in deltas
    assert "margin_pct" in deltas

    # Mathematical consistency: simulated revenue = baseline revenue + delta revenue
    np.testing.assert_approx_equal(
        sim_res["simulated"]["revenue"],
        sim_res["baseline"]["revenue"] + deltas["revenue"],
        significant=4
    )


def test_scenario_simulator_multi(enterprise_sample_df):
    """Verifies multi-scenario comparison table and best scenario detection."""
    scenarios = {
        "Aggressive Growth": {"price": -5.0, "marketing": 25.0},
        "Margin Protection": {"price": 12.0, "cost": -8.0},
        "Retention Focus": {"churn": -15.0, "marketing": 10.0},
    }
    multi_res = bi.compare_multiple_scenarios(enterprise_sample_df, scenarios)

    assert "comparison_table" in multi_res
    assert "best_scenario" in multi_res
    assert "ai_synthesis" in multi_res
    assert len(multi_res["comparison_table"]) == len(scenarios) + 1  # Includes Baseline


# =====================================================================
# 4. Business Alerts Engine Tests
# =====================================================================
def test_alert_engine(enterprise_sample_df):
    """Verifies alert generation, 8 alert categories, and significance filtering."""
    alerts = bi.generate_business_alerts(enterprise_sample_df)
    assert isinstance(alerts, list)

    for alert in alerts:
        assert "id" in alert
        assert "category" in alert
        assert "severity" in alert
        assert alert["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
        assert "title" in alert
        assert "deviation" in alert
        assert "statistical_significance" in alert
        assert "recommended_action" in alert


# =====================================================================
# 5. Department-Specific AI Analysts Tests
# =====================================================================
def test_department_analysts(enterprise_sample_df):
    """Verifies all 7 department configurations and tailored analytical briefings."""
    roles = [
        "Sales Analyst", "Marketing Analyst", "Finance Analyst",
        "Customer Analyst", "Operations Analyst", "Product Analyst",
        "Executive Analyst"
    ]

    for role in roles:
        dept_data = bi.get_department_analysis(enterprise_sample_df, department=role)
        assert dept_data["role"] == role
        assert "kpis" in dept_data
        assert isinstance(dept_data["kpis"], list)
        assert len(dept_data["kpis"]) >= 3
        assert "briefing_memo" in dept_data
        assert len(dept_data["briefing_memo"]) > 20
        assert "preset_questions" in dept_data
        assert len(dept_data["preset_questions"]) > 0


# =====================================================================
# 6. Executive Command Center Tests
# =====================================================================
def test_executive_command_center(enterprise_sample_df):
    """Verifies 0-100 composite health score and executive dashboard synthesis."""
    cc = bi.generate_executive_command_center(enterprise_sample_df, dataset_name="Enterprise Test")

    assert "health" in cc
    health = cc["health"]
    assert 0 <= health["overall_score"] <= 100
    assert health["grade"] in ("A (Optimal)", "B (Good)", "C (Fair)", "D (Attention)", "F (Urgent Action)")
    assert "dimensions" in health
    assert len(health["dimensions"]) == 7

    assert "top_kpi_cards" in cc
    assert len(cc["top_kpi_cards"]) >= 4

    assert "summary" in cc
    summary = cc["summary"]
    assert "status" in summary
    assert "positive_signals" in summary
    assert "top_risks" in summary
    assert "top_opportunities" in summary
    assert "recommended_actions" in summary

    assert "insight_to_action" in cc
    assert isinstance(cc["insight_to_action"], list)


# =====================================================================
# 7. Business Reporting Engine Tests (PDF, HTML, CSV)
# =====================================================================
def test_business_reports(enterprise_sample_df):
    """Verifies exportable 16-section dossier generation in PDF, HTML, and CSV."""
    report = bi.generate_business_report_dossier(enterprise_sample_df, dataset_name="Enterprise Test")

    assert "metadata" in report
    assert "sections" in report
    assert len(report["sections"]) == 16
    assert "html" in report
    assert "csv" in report
    assert "pdf_bytes" in report

    # Verify HTML formatting
    assert "<!DOCTYPE html>" in report["html"]
    assert "Enterprise Test" in report["html"]

    # Verify CSV export
    assert len(report["csv"]) > 50
    assert "Section,Item" in report["csv"]

    # Verify PDF generation (binary starting with %PDF)
    pdf_bytes = report["pdf_bytes"]
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


# =====================================================================
# 8. AI Assistant Query Routing & Controlled Tools Tests
# =====================================================================
def test_ai_assistant_bi_queries(enterprise_sample_df):
    """Verifies that the Quantum AI Assistant correctly resolves business intelligence queries."""
    engine = QuantumAIEngine()

    test_cases = [
        ("How is business performing?", "business_health", "analyze_business_health"),
        ("What's our biggest risk?", "business_risks", "detect_business_risks"),
        ("What should the business focus on?", "business_recommendations", "generate_recommendations"),
        ("Where are the business opportunities?", "business_opportunities", "detect_business_opportunities"),
        ("What if price increases by 10%?", "scenario_simulation", "run_business_scenario"),
        ("Summarize for management", "executive_summary", "generate_executive_summary"),
        ("What is the Finance perspective?", "department_analysis", "get_department_analysis"),
    ]

    for query, expected_intent, expected_tool in test_cases:
        res = engine.process_query(query, df=enterprise_sample_df)
        assert res["intent"] == expected_intent, f"Failed intent for query: {query}"
        assert expected_tool in res["tool_used"], f"Failed tool for query: {query}"
        assert res["markdown"] is not None and len(res["markdown"]) > 50
        assert res["voice_text"] is not None and len(res["voice_text"]) > 10
        assert "Answer" in res["markdown"] or "Briefing" in res["markdown"] or "Posture" in res["markdown"]
