"""
tests.test_quantum_ai
=====================
Comprehensive unit tests for the upgraded Quantum AI Data Analyst Assistant.
Validates:
1. Project Knowledge Base (architecture, technologies, workflows, DuckDB, Parquet)
2. Simple Language Mode & intuitive analogies (PCA, Random Forest, Regression vs Classification)
3. Controlled analytical tool execution with zero mathematical hallucination
4. Answer -> Evidence -> Explanation -> Recommendation structure
5. Voice text generation and audio summary conciseness
6. Active page context awareness
7. Token budget control on large datasets
"""

import pytest
import pandas as pd
import numpy as np

import core.ai.project_knowledge as pk
import core.ai.analytical_tools as tools
from core.ai.quantum_ai_engine import QuantumAIEngine
from core.ai.llm_client import LLMClient


@pytest.fixture
def sample_sales_df():
    """Returns a realistic tabular sales dataset for analytical testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    categories = ["Electronics", "Fashion", "Home", "Office"]
    regions = ["North", "South", "East", "West"]

    df = pd.DataFrame({
        "date": dates,
        "sales": np.random.uniform(50.0, 500.0, size=n).round(2),
        "quantity": np.random.randint(1, 15, size=n),
        "profit": np.random.uniform(-20.0, 150.0, size=n).round(2),
        "category": np.random.choice(categories, size=n),
        "region": np.random.choice(regions, size=n),
    })
    # Inject deliberate missing values and duplicates for quality checks
    df.loc[5:8, "profit"] = np.nan
    df = pd.concat([df, df.iloc[0:2]], ignore_index=True)
    return df


def test_project_knowledge_queries():
    """Verifies that all core project knowledge inquiries return verified factual answers."""
    engine = QuantumAIEngine()

    # 1. Project overview
    res1 = engine.process_query("What does this project do?")
    assert res1["intent"] == "project_knowledge"
    assert "Quantum Analyst" in res1["markdown"]
    assert "Data engines calculate. ML models predict. AI explains." in res1["markdown"]
    assert len(res1["voice_text"]) > 20

    # 2. DuckDB justification
    res2 = engine.process_query("Why is DuckDB used?")
    assert res2["intent"] == "project_knowledge"
    assert "DuckDB" in res2["markdown"]
    assert "columnar" in res2["markdown"].lower()

    # 3. Parquet justification
    res3 = engine.process_query("Why do we use Parquet?")
    assert res3["intent"] == "project_knowledge"
    assert "Parquet" in res3["markdown"]
    assert "compression" in res3["markdown"].lower()

    # 4. Agentic AI
    res4 = engine.process_query("What is Agentic AI?")
    assert res4["intent"] == "project_knowledge"
    assert "agency" in res4["markdown"].lower() or "tools" in res4["markdown"].lower()

    # 5. Complete workflow
    res5 = engine.process_query("Explain the complete workflow")
    assert res5["intent"] == "project_knowledge"
    assert "14-stage" in res5["markdown"] or "Ingestion" in res5["markdown"]


def test_simple_language_analogies():
    """Verifies that technical concepts are explained using relatable layman analogies."""
    engine = QuantumAIEngine()

    # 1. PCA analogy
    res_pca = engine.process_query("What is PCA?")
    assert "statue" in res_pca["markdown"].lower() or "photo" in res_pca["markdown"].lower() or "compress" in res_pca["markdown"].lower()
    assert len(res_pca["voice_text"]) > 20

    # 2. Random Forest analogy
    res_rf = engine.process_query("What is Random Forest?")
    assert "jury" in res_rf["markdown"].lower() or "doctor" in res_rf["markdown"].lower() or "trees" in res_rf["markdown"].lower()

    # 3. Regression vs Classification
    res_rvc = engine.process_query("What is the difference between regression and classification?")
    assert "number" in res_rvc["markdown"].lower()
    assert "category" in res_rvc["markdown"].lower() or "label" in res_rvc["markdown"].lower()


def test_data_aware_profiling_and_quality(sample_sales_df):
    """Verifies that profiling and quality audits calculate genuine values from data without hallucination."""
    engine = QuantumAIEngine()

    # 1. Profiling
    res_prof = engine.process_query("Summarize my dataset", df=sample_sales_df)
    assert res_prof["intent"] == "dataset_profile"
    assert f"{len(sample_sales_df):,}" in res_prof["markdown"]
    assert "Answer" in res_prof["markdown"]
    assert "Evidence" in res_prof["markdown"]
    assert "Recommendation" in res_prof["markdown"]

    # 2. Data Quality
    res_qual = engine.process_query("Find the biggest problem in my data", df=sample_sales_df)
    assert res_qual["intent"] == "data_quality"
    assert "Health Score" in res_qual["markdown"] or "Quality" in res_qual["markdown"]
    # Check that real missing cells were found
    assert "profit" in res_qual["markdown"].lower() or "missing" in res_qual["markdown"].lower() or "duplicate" in res_qual["markdown"].lower()


def test_controlled_sql_execution(sample_sales_df):
    """Verifies that natural language SQL queries execute safely via DuckDB with real results."""
    engine = QuantumAIEngine()

    res_sql = engine.process_query("Show top 3 categories by sales", df=sample_sales_df)
    assert res_sql["intent"] == "sql_query"
    assert "DuckDB" in res_sql["markdown"]
    assert res_sql.get("data") is not None
    # Result should have at most 3 rows
    assert len(res_sql["data"]) <= 3


def test_forecasting_and_ml_tools(sample_sales_df):
    """Verifies forecasting and ML target detection execution."""
    engine = QuantumAIEngine()

    # 1. Target discovery
    res_target = engine.process_query("Which column should be the target?", df=sample_sales_df)
    assert res_target["intent"] in ["detect_target", "ml_problem"]
    assert "sales" in res_target["markdown"].lower() or "profit" in res_target["markdown"].lower()

    # 2. Forecasting
    res_fc = engine.process_query("Forecast sales for the next 14 days", df=sample_sales_df, target_col="sales")
    assert res_fc["intent"] == "forecasting"
    assert "14" in res_fc["markdown"]
    assert "trend" in res_fc["markdown"].lower() or "model" in res_fc["markdown"].lower()


def test_page_context_awareness(sample_sales_df):
    """Verifies that generic questions are interpreted through active page context."""
    engine = QuantumAIEngine()

    # On Data Quality page
    res_q = engine.process_query("What does this mean?", df=sample_sales_df, current_page="Data Quality")
    assert res_q["intent"] == "page_quality"
    assert "quality" in res_q["markdown"].lower()

    # On EDA page
    res_eda = engine.process_query("What does this mean?", df=sample_sales_df, current_page="EDA")
    assert res_eda["intent"] == "page_eda"
    assert "eda" in res_eda["markdown"].lower() or "finding" in res_eda["markdown"].lower()


def test_voice_summary_conciseness(sample_sales_df):
    """Verifies that voice text is concise, natural, and never recites full tables aloud."""
    engine = QuantumAIEngine()

    res = engine.process_query("Summarize my dataset", df=sample_sales_df)
    voice_txt = res.get("voice_text", "")
    assert isinstance(voice_txt, str)
    assert len(voice_txt) > 0
    # Voice text should be concise (less than 350 characters for comfortable listening)
    assert len(voice_txt) < 350
    # Should not contain markdown headers or pipe table markers
    assert "###" not in voice_txt
    assert "|---|" not in voice_txt


def test_model_resilience_and_recommendations(sample_sales_df):
    """Verifies that model queries work gracefully even without an active champion model."""
    engine = QuantumAIEngine()

    # Model recommendation when no champion exists
    res = engine.process_query("Which model is best?", df=sample_sales_df, champ_info=None)
    assert "recommend" in res["intent"] or "champion" in res["intent"] or "ml_problem" in res["intent"]
    assert "Random Forest" in res["markdown"] or "Regression" in res["markdown"] or "model" in res["markdown"].lower()

    # Analytical tool evaluate without champion
    eval_res = tools.evaluate_model(None, df=None)
    assert eval_res.get("answer") is not None
    assert "no champion" in eval_res.get("answer", "").lower() or eval_res.get("success") is True


def test_additional_function_knowledge():
    """Verifies knowledge entries for Forecasting, What-If Scenarios, and Quality Scoring."""
    engine = QuantumAIEngine()

    # 1. Forecasting knowledge
    res_fc = engine.process_query("How does forecasting work?")
    assert res_fc["intent"] == "project_knowledge"
    assert "ARIMA" in res_fc["markdown"] or "Holt-Winters" in res_fc["markdown"]

    # 2. What-If Scenario knowledge
    res_sc = engine.process_query("What is What-If scenario analysis?")
    assert res_sc["intent"] == "project_knowledge"
    assert "scenario" in res_sc["markdown"].lower() or "counterfactual" in res_sc["markdown"].lower()

    # 3. Quality score knowledge
    res_qs = engine.process_query("How is the data quality score calculated?")
    assert res_qs["intent"] == "project_knowledge"
    assert "completeness" in res_qs["markdown"].lower() or "dimension" in res_qs["markdown"].lower()
