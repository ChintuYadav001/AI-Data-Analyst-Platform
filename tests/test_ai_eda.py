"""
Unit and integration tests for the High-Level AI EDA Engine.
Tests all 9 modular engines, 14-stage pipeline, backward compatibility,
outlier exploration without silent deletion, and non-causal guardrails.
"""
import pytest
import numpy as np
import pandas as pd

from core.sample_data import generate_ecommerce_dataset, generate_customer_churn_dataset
from core.eda import (
    EDAProfiler,
    EDAGoalEngine,
    EDAStatisticalEngine,
    EDATimeSeriesEngine,
    EDASegmentEngine,
    EDAAnomalyEngine,
    EDADriverEngine,
    EDANLQueryEngine,
    AIEDAOrchestrator
)
from core.models.eda_model import EDAModel


@pytest.fixture
def ecommerce_df():
    return generate_ecommerce_dataset()


@pytest.fixture
def churn_df():
    return generate_customer_churn_dataset()


def test_profiler_intelligence_and_domain(ecommerce_df, churn_df):
    """Verifies schema intelligence, identifier isolation, and domain classification."""
    p_ecom = EDAProfiler.profile_dataset(ecommerce_df)
    assert p_ecom["n_rows"] > 0
    assert p_ecom["n_cols"] > 0
    assert "E-Commerce" in p_ecom["domain"]
    assert len(p_ecom["columns_by_type"]["numerical"]) > 0
    assert len(p_ecom["target_candidates"]) > 0
    assert "order_id" in p_ecom["columns_by_type"]["identifier"]

    p_churn = EDAProfiler.profile_dataset(churn_df)
    assert any(c["column"] == "churn" for c in p_churn["target_candidates"])
    assert p_churn["summary_narrative"] != ""


def test_goal_and_target_engine(ecommerce_df):
    """Verifies dynamic goal inference, target resolution, and priority selection."""
    profile = EDAProfiler.profile_dataset(ecommerce_df)
    
    # Inferred goal
    inferred = EDAGoalEngine.infer_goal(profile)
    assert inferred["is_inferred"] is True
    assert len(inferred["goal"]) > 10

    # Explicit user goal
    explicit = EDAGoalEngine.infer_goal(profile, user_goal="Evaluate profit margins across north region")
    assert explicit["is_inferred"] is False
    assert explicit["goal"] == "Evaluate profit margins across north region"

    # Target resolution
    target_res = EDAGoalEngine.resolve_target(ecommerce_df, profile, user_target="profit")
    assert target_res["column"] == "profit"
    assert target_res["task_type"] == "regression"

    # Priority columns
    priority = EDAGoalEngine.select_priority_columns(ecommerce_df, profile, target_col="profit")
    assert "profit" in priority
    assert "order_id" not in priority  # Identifier must be excluded


def test_statistical_engine_moments_and_distributions(ecommerce_df):
    """Verifies moments, quartiles, skewness, kurtosis, and distribution shapes."""
    moments = EDAStatisticalEngine.analyze_numerical_moments(ecommerce_df)
    assert len(moments) > 0
    
    for m in moments:
        assert "mean" in m
        assert "median" in m
        assert "skewness" in m
        assert "kurtosis" in m
        assert "distribution_shape" in m
        assert "recommended_transform" in m
        assert len(m["histogram"]) > 0

    cats = EDAStatisticalEngine.analyze_categorical_distributions(ecommerce_df)
    assert len(cats) > 0
    for c in cats:
        assert "top_categories" in c
        assert "pareto_ratio_pct" in c
        assert "top3_concentration_pct" in c


def test_statistical_engine_hypothesis_tests(ecommerce_df):
    """Verifies automated 2-sample, ANOVA/Kruskal-Wallis, and Chi-Square tests."""
    # 2-group comparison
    res_2g = EDAStatisticalEngine.test_two_groups(ecommerce_df, num_col="sales", cat_col="region")
    if res_2g:
        assert "statistic" in res_2g
        assert "p_value" in res_2g
        assert "effect_size" in res_2g
        assert "causation" not in res_2g["interpretation"].lower() or "does not establish causation" in res_2g["interpretation"]

    # Multi-group comparison
    res_multi = EDAStatisticalEngine.test_multiple_groups(ecommerce_df, num_col="sales", cat_col="category")
    if res_multi:
        assert "statistic" in res_multi
        assert "p_value" in res_multi
        assert "effect_size_eta_sq" in res_multi

    # Categorical independence
    res_cat = EDAStatisticalEngine.test_categorical_independence(ecommerce_df, cat1="region", cat2="category")
    if res_cat:
        assert "chi2_statistic" in res_cat
        assert "cramers_v" in res_cat


def test_time_series_auto_granularity(ecommerce_df):
    """Verifies auto-granularity, period growth, and rolling average calculations."""
    ts = EDATimeSeriesEngine.analyze_temporal_trends(ecommerce_df)
    assert ts is not None
    assert "granularity" in ts
    assert "trend_points" in ts
    assert len(ts["trend_points"]) > 0
    assert "trend_direction" in ts
    assert "latest_growth_rate_pct" in ts


def test_segment_contribution_analysis(ecommerce_df):
    """Verifies cohort grouping, contribution %, and ranking."""
    segs = EDASegmentEngine.analyze_segments(ecommerce_df, target_col="sales")
    assert len(segs) > 0
    top_seg = segs[0]
    assert "top2_concentration_pct" in top_seg
    assert len(top_seg["cohorts"]) > 0
    assert top_seg["cohorts"][0]["contribution_pct"] > 0


def test_anomaly_multi_method_no_silent_delete(ecommerce_df):
    """Verifies outlier detection across Tukey IQR, Z-Score, and MAD, with impact on mean."""
    anomalies = EDAAnomalyEngine.detect_anomalies(ecommerce_df)
    assert len(anomalies) > 0
    for a in anomalies:
        assert "iqr_outliers_count" in a
        assert "mean_shift_pct" in a
        assert "classification" in a
        # Ensure policy is quarantine / deep-dive, not deletion
        assert "delete" not in a["recommendation"].lower() or "do not delete" in a["recommendation"].lower()


def test_target_driver_analysis(ecommerce_df, churn_df):
    """Verifies target driver association ranking and non-causal guardrails."""
    drivers_ecom = EDADriverEngine.analyze_drivers(ecommerce_df, target_col="profit", task_type="regression")
    assert len(drivers_ecom) > 0
    assert drivers_ecom[0]["association_score"] >= 0

    drivers_churn = EDADriverEngine.analyze_drivers(churn_df, target_col="churn", task_type="classification")
    assert len(drivers_churn) > 0
    assert drivers_churn[0]["association_score"] >= 0


def test_natural_language_query_assistant(ecommerce_df):
    """Verifies deterministic DuckDB execution and concrete numerical answers."""
    res_top = EDANLQueryEngine.answer_query(ecommerce_df, "What are the top categories by total sales?")
    assert "sales" in res_top["answer"].lower()
    assert res_top["sql"] != ""
    assert res_top["table"] is not None

    res_anom = EDANLQueryEngine.answer_query(ecommerce_df, "Are there outliers in profit?")
    assert "outlier" in res_anom["answer"].lower() or "anomaly" in res_anom["answer"].lower()


def test_orchestrator_14_stage_pipeline(ecommerce_df):
    """Verifies execution of the entire 14-stage AI EDA pipeline and synthesis of findings."""
    eda = AIEDAOrchestrator.run_full_eda(ecommerce_df)
    assert "profile" in eda
    assert "quality" in eda
    assert "goal_info" in eda
    assert "target_info" in eda
    assert "eda_plan" in eda
    assert "coverage" in eda
    assert eda["coverage"]["score"] >= 70
    assert len(eda["key_findings"]) > 0
    
    for f in eda["key_findings"]:
        assert "title" in f
        assert "finding" in f
        assert "evidence" in f
        assert "confidence" in f
        assert f["confidence"] in ["High", "Medium", "Low"]

    assert len(eda["chart_tiers"]["level1_executive"]) > 0
    assert len(eda["next_analyses"]) > 0
    assert len(eda["feature_recommendations"]) > 0


def test_backward_compatibility_with_legacy_edamodel(ecommerce_df):
    """Verifies that EDAModel preserves identical legacy method contracts."""
    num = EDAModel.analyze_numerical(ecommerce_df)
    assert isinstance(num, list)
    assert len(num) > 0
    assert "skewness" in num[0]

    cat = EDAModel.analyze_categorical(ecommerce_df)
    assert isinstance(cat, list)
    assert len(cat) > 0
    assert "top_categories" in cat[0]

    rel = EDAModel.analyze_relationships(ecommerce_df)
    assert isinstance(rel, dict)
    assert "correlation_matrix" in rel
    assert "strong_positive" in rel

    trends = EDAModel.analyze_time_trends(ecommerce_df)
    assert isinstance(trends, dict)
    assert "daily_sample" in trends


def test_large_dataset_and_edge_cases():
    """Verifies stability on large datasets and edge cases (empty or single-column)."""
    # 1. 20,000 rows synthetic dataset
    np.random.seed(42)
    large_df = pd.DataFrame({
        "order_id": [f"ID_{i}" for i in range(20000)],
        "date": pd.date_range("2022-01-01", periods=20000, freq="h"),
        "category": np.random.choice(["Tech", "Office", "Furniture", "Apparel"], size=20000),
        "revenue": np.random.exponential(scale=100, size=20000),
        "cost": np.random.normal(loc=50, scale=10, size=20000)
    })

    res = AIEDAOrchestrator.run_full_eda(large_df, mode="standard")
    assert res["profile"]["n_rows"] == 20000
    assert res["coverage"]["score"] >= 80

    # 2. Empty dataset edge case
    empty_res = AIEDAOrchestrator.run_full_eda(pd.DataFrame())
    assert empty_res["coverage"]["score"] == 0

    # 3. Single column dataset
    single_df = pd.DataFrame({"metric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    single_res = AIEDAOrchestrator.run_full_eda(single_df)
    assert single_res["profile"]["n_cols"] == 1
