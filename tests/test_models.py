import pytest
import pandas as pd
import numpy as np
from core.sample_data import generate_ecommerce_dataset, generate_customer_churn_dataset
from core.models import (
    DataProfilerModel,
    DataQualityModel,
    DataCleanerModel,
    EDAModel,
    ChartModel,
    FeatureEngineeringModel,
    SQLAnalyticsModel,
    AIInsightsModel,
    AIChatModel,
    DashboardModel,
    ReportModel,
    ProjectModel
)

@pytest.fixture
def ecommerce_df():
    return generate_ecommerce_dataset(100)

def test_data_profiler(ecommerce_df):
    profile = DataProfilerModel.profile_dataset(ecommerce_df)
    assert "dataset_info" in profile
    assert profile["dataset_info"]["rows"] > 0
    assert profile["dataset_info"]["columns"] > 0
    assert len(profile["columns"]) == len(ecommerce_df.columns)
    
    # Check column types classification
    types = [c["semantic_type"] for c in profile["columns"]]
    assert "Identifier" in types or "Numerical" in types

def test_data_quality(ecommerce_df):
    quality = DataQualityModel.calculate_quality_score(ecommerce_df)
    assert 0 <= quality["score"] <= 100
    assert len(quality["breakdown"]) > 0

def test_data_cleaner(ecommerce_df):
    recs = DataCleanerModel.recommend_cleaning_plan(ecommerce_df)
    assert len(recs) > 0
    
    actions = [{"action": "remove_duplicates"}, {"action": "impute_median", "column": "profit"}]
    cleaned_df, log = DataCleanerModel.apply_cleaning_actions(ecommerce_df, actions)
    assert len(cleaned_df) <= len(ecommerce_df)
    assert len(log) > 0
    assert cleaned_df["profit"].isna().sum() == 0

def test_eda_model(ecommerce_df):
    num_analysis = EDAModel.analyze_numerical(ecommerce_df)
    assert len(num_analysis) > 0
    assert "skewness" in num_analysis[0]
    
    cat_analysis = EDAModel.analyze_categorical(ecommerce_df)
    assert len(cat_analysis) > 0
    
    relationships = EDAModel.analyze_relationships(ecommerce_df)
    assert "correlation_matrix" in relationships

def test_chart_model(ecommerce_df):
    charts = ChartModel.recommend_charts(ecommerce_df)
    assert len(charts) > 0
    fig = ChartModel.create_plotly_figure(ecommerce_df, charts[0])
    assert fig is not None

def test_feature_model(ecommerce_df):
    feats = FeatureEngineeringModel.recommend_features(ecommerce_df)
    assert len(feats) > 0
    
    date_feat = next((f for f in feats if f["type"] == "date_expansion"), None)
    if date_feat:
        df_new, msg = FeatureEngineeringModel.apply_feature(ecommerce_df, date_feat)
        assert f"{date_feat['target_col']}_year" in df_new.columns

def test_sql_engine_and_security(ecommerce_df):
    sql_engine = SQLAnalyticsModel()
    sql_engine.register_dataframe("sales", ecommerce_df)
    
    # Valid analytical query
    res = sql_engine.execute_query("SELECT region, COUNT(*) as count FROM sales GROUP BY region")
    assert res["success"] is True
    assert res["row_count"] > 0
    
    # Destructive blocked queries
    bad_res1 = sql_engine.execute_query("DROP TABLE sales")
    assert bad_res1["success"] is False
    assert "Security Violation" in bad_res1["error"] or "Unauthorized" in bad_res1["error"]
    
    bad_res2 = sql_engine.execute_query("DELETE FROM sales WHERE region = 'North'")
    assert bad_res2["success"] is False

def test_ai_insights(ecommerce_df):
    profile = DataProfilerModel.profile_dataset(ecommerce_df)
    eda = {
        "numerical": EDAModel.analyze_numerical(ecommerce_df),
        "categorical": EDAModel.analyze_categorical(ecommerce_df),
        "relationships": EDAModel.analyze_relationships(ecommerce_df),
        "time_trends": EDAModel.analyze_time_trends(ecommerce_df)
    }
    quality = DataQualityModel.calculate_quality_score(ecommerce_df)
    insights = AIInsightsModel.generate_offline_insights(profile, eda, quality)
    assert len(insights["key_insights"]) > 0
    assert len(insights["recommendations"]) > 0

def test_dashboard_model(ecommerce_df):
    kpis = DashboardModel.compute_kpis(ecommerce_df)
    assert len(kpis) >= 3
    dims = DashboardModel.detect_filter_dimensions(ecommerce_df)
    assert len(dims) > 0

def test_project_goal_and_30page_report(ecommerce_df, tmp_path):
    import pypdf
    from core.models import ReportModel
    
    # Test goal persistence
    goal = "Revenue Maximization & Growth"
    proj = ProjectModel.create_project("Test_CI_Goal", ecommerce_df, goal=goal)
    loaded = ProjectModel.get_project(proj.project_id)
    assert loaded is not None
    assert loaded.metadata.get("goal") == goal

    # Test goal-aligned AI insights
    profile = DataProfilerModel.profile_dataset(ecommerce_df)
    eda = {
        "numerical": EDAModel.analyze_numerical(ecommerce_df),
        "categorical": EDAModel.analyze_categorical(ecommerce_df),
        "relationships": EDAModel.analyze_relationships(ecommerce_df),
        "time_trends": EDAModel.analyze_time_trends(ecommerce_df)
    }
    quality = DataQualityModel.calculate_quality_score(ecommerce_df)
    insights = AIInsightsModel.generate_insights(profile, eda, quality, goal=goal)
    assert "goal_alignment" in insights
    assert len(insights["goal_alignment"]["strategic_findings"]) > 0

    # Test PDF generation >= 30 pages
    pdf_out = str(tmp_path / "ci_dossier.pdf")
    ReportModel.export_pdf(pdf_out, "Test CI", profile, quality, [], eda, insights, goal=goal)
    reader = pypdf.PdfReader(pdf_out)
    assert len(reader.pages) >= 30

def test_goal_explanation_and_hypothesis_analysis(ecommerce_df):
    goal = "Revenue Maximization & Growth"
    explanation = "Investigate regional margin leakage and evaluate profit performance across product categories."
    
    # 1. Project model persistence
    proj = ProjectModel.create_project("Test_Expl_Proj", ecommerce_df, goal=goal, goal_explanation=explanation)
    loaded = ProjectModel.get_project(proj.project_id)
    assert loaded is not None
    assert loaded.metadata.get("goal_explanation") == explanation

    # 2. AI insights evaluation on basis of explanation
    profile = DataProfilerModel.profile_dataset(ecommerce_df)
    eda = {
        "numerical": EDAModel.analyze_numerical(ecommerce_df),
        "categorical": EDAModel.analyze_categorical(ecommerce_df),
        "relationships": EDAModel.analyze_relationships(ecommerce_df),
        "time_trends": EDAModel.analyze_time_trends(ecommerce_df)
    }
    quality = DataQualityModel.calculate_quality_score(ecommerce_df)
    insights = AIInsightsModel.generate_insights(
        profile, eda, quality, goal=goal, goal_explanation=explanation
    )
    
    goal_data = insights.get("goal_alignment", {})
    assert goal_data.get("goal_explanation") == explanation
    assert len(goal_data.get("targeted_metrics", [])) > 0
    assert len(goal_data.get("hypothesis_evaluations", [])) > 0

def test_notification_model():
    from core.models.notification_model import NotificationModel
    test_user_id = "test_user_notif_999"
    
    # 1. Create notification
    notif_id = NotificationModel.create_notification(
        title="Test Ingestion Complete",
        message="Your telemetry stream has been processed successfully.",
        user_id=test_user_id,
        notification_type="success",
        icon="⚡",
        link_route="Dashboard"
    )
    assert notif_id is not None
    
    # 2. Check unread count
    unread = NotificationModel.get_unread_count(user_id=test_user_id)
    assert unread >= 1
    
    # 3. Retrieve notifications
    notifs = NotificationModel.get_notifications(user_id=test_user_id)
    found = [n for n in notifs if n["id"] == notif_id]
    assert len(found) == 1
    assert found[0]["title"] == "Test Ingestion Complete"
    
    # 4. Mark as read
    NotificationModel.mark_as_read(notif_id)
    notifs_after = NotificationModel.get_notifications(user_id=test_user_id)
    found_after = [n for n in notifs_after if n["id"] == notif_id]
    assert found_after[0]["is_read"] == 1
    
    # 5. Mark all as read
    NotificationModel.mark_all_as_read(user_id=test_user_id)
    assert NotificationModel.get_unread_count(user_id=test_user_id) == 0

def test_chart_model_themes(ecommerce_df):
    from core.models.chart_model import ChartModel
    recs = ChartModel.recommend_charts(ecommerce_df)
    assert len(recs) > 0
    rec = recs[0]
    
    # Dark theme figure
    fig_dark = ChartModel.create_plotly_figure(ecommerce_df, rec, theme="dark_futuristic")
    assert fig_dark is not None
    assert fig_dark.layout.paper_bgcolor in ["#0A0E1A", "rgba(14, 20, 36, 0.8)", "#080C16", "rgba(0,0,0,0)"]
    
    # Light theme figure
    fig_light = ChartModel.create_plotly_figure(ecommerce_df, rec, theme="light_classic")
    assert fig_light is not None
    assert fig_light.layout.paper_bgcolor in ["#FFFFFF", "rgba(0,0,0,0)"]



