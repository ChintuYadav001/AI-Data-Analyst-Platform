"""
tests.test_model_copilot
========================
Unit tests for AI Model Copilot knowledge engine, analogies, business use cases,
and voice synthesis script generation.
"""

import pytest
from core.ml.model_copilot import ModelCopilotEngine, MODEL_ENCYCLOPEDIA


@pytest.fixture
def copilot_engine():
    return ModelCopilotEngine()


def test_model_encyclopedia_coverage():
    """Verify core models have analogies, business uses, pros, and cons."""
    critical_models = [
        "linear_regression", "ridge_regression", "lasso_regression",
        "random_forest_regressor", "xgboost_regressor", "logistic_regression",
        "random_forest_classifier", "xgboost_classifier", "kmeans",
        "isolation_forest", "pca", "arima"
    ]
    for m in critical_models:
        assert m in MODEL_ENCYCLOPEDIA
        info = MODEL_ENCYCLOPEDIA[m]
        assert "analogy" in info
        assert "simple_explanation" in info
        assert len(info["business_uses"]) >= 2
        assert len(info["pros"]) >= 1
        assert len(info["cons"]) >= 1


def test_single_model_query(copilot_engine):
    # Query XGBoost
    res = copilot_engine.answer_model_question("Can you explain XGBoost in simple language?")
    assert "XGBoost" in res["title"]
    assert "Analogy" in res["markdown"]
    assert "Business Use Cases" in res["markdown"]
    assert len(res["voice_text"]) > 20

    # Query PCA
    res_pca = copilot_engine.answer_model_question("Explain PCA like I am 10")
    assert "PCA" in res_pca["title"]
    assert "photo" in res_pca["markdown"].lower() or "statue" in res_pca["markdown"].lower()

    # Query Isolation Forest
    res_anom = copilot_engine.answer_model_question("What does Isolation Forest do?")
    assert "Isolation Forest" in res_anom["markdown"]


def test_model_comparison_query(copilot_engine):
    res = copilot_engine.answer_model_question("Compare Random Forest vs XGBoost")
    assert "Random Forest vs. XGBoost" in res["markdown"]
    assert "Jury" in res["markdown"] or "archery" in res["markdown"].lower()

    res_ridge = copilot_engine.answer_model_question("Difference between Ridge vs Lasso")
    assert "Ridge vs. Lasso" in res_ridge["markdown"]


def test_metrics_query(copilot_engine):
    res = copilot_engine.answer_model_question("Explain Precision and Recall simply")
    assert "Precision" in res["markdown"]
    assert "Recall" in res["markdown"]


def test_all_models_query(copilot_engine):
    res = copilot_engine.answer_model_question("Explain all models in the platform")
    assert "Complete Algorithm Catalog" in res["markdown"]


def test_champion_context_explanation(copilot_engine):
    mock_champ = {
        "model_id": "random_forest_classifier",
        "model_name": "Random Forest Classifier",
        "task_type": "binary_classification",
        "target_col": "churn",
        "metrics": {"accuracy": 0.92, "f1": 0.89},
        "feature_importance": [
            {"feature": "annual_income", "importance": 0.45},
            {"feature": "age", "importance": 0.32},
        ],
    }
    res = copilot_engine.answer_model_question("Why did my champion model win?", champ_info=mock_champ)
    assert "Random Forest Classifier" in res["markdown"]
    assert "churn" in res["markdown"]
    assert "annual_income" in res["markdown"]


def test_voice_text_cleaning(copilot_engine):
    raw = "**Hello** [click here](http://example.com) `code_text` *italic*"
    cleaned = copilot_engine._clean_for_voice(raw)
    assert "*" not in cleaned
    assert "[" not in cleaned
    assert "`" not in cleaned
    assert "Hello" in cleaned
    assert "code_text" in cleaned
