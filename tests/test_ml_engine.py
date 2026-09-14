"""
tests.test_ml_engine
====================
Comprehensive test suite verifying ML readiness, preprocessing, model registry,
trainer, evaluation, explainability, forecasting, inference, and experiment tracking.
"""

import os
import shutil
import numpy as np
import pandas as pd
import pytest

from core.ml.ml_readiness import audit_ml_readiness, classify_task_type, detect_target_candidates, detect_target_leakage
from core.ml.preprocessing import build_preprocessor, get_feature_names_after_preprocessing
from core.ml.model_registry import REGISTRY, get_candidate_models, create_model_instance
from core.ml.trainer import train_single_model
from core.ml.model_selection import run_benchmark
from core.ml.evaluation import evaluate_regression, evaluate_classification
from core.ml.explainability import extract_global_feature_importance, explain_local_prediction
from core.ml.forecasting import run_time_series_forecast
from core.ml.prediction import predict_single_record, predict_batch
from core.ml.experiment_tracker import save_experiment, load_model, list_experiments, get_experiment


@pytest.fixture
def classification_data():
    """Generates synthetic classification dataset."""
    np.random.seed(42)
    n = 200
    age = np.random.randint(18, 70, size=n)
    income = np.random.uniform(20000, 120000, size=n)
    category = np.random.choice(["Retail", "Tech", "Finance"], size=n)
    # Target correlated with income and age
    prob = 1 / (1 + np.exp(-(0.00005 * income + 0.02 * age - 2.5)))
    churn = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": age,
        "annual_income": income,
        "segment": category,
        "churn": churn,
    })


@pytest.fixture
def regression_data():
    """Generates synthetic sales regression dataset."""
    np.random.seed(42)
    n = 250
    units = np.random.randint(1, 50, size=n)
    unit_price = np.random.uniform(10, 100, size=n)
    marketing_spend = np.random.uniform(100, 2000, size=n)
    region = np.random.choice(["North", "South", "East", "West"], size=n)
    revenue = units * unit_price + 0.5 * marketing_spend + np.random.normal(0, 50, size=n)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "order_id": [f"ORD_{i:04d}" for i in range(n)],
        "order_date": dates,
        "units": units,
        "unit_price": unit_price,
        "marketing_spend": marketing_spend,
        "region": region,
        "revenue": revenue,
    })


def test_ml_readiness_and_target_detection(classification_data, regression_data):
    # Test classification readiness
    clf_report = audit_ml_readiness(classification_data)
    assert clf_report.is_ready is True
    assert clf_report.overall_score >= 80
    assert "customer_id" in clf_report.id_columns
    assert clf_report.selected_target == "churn"
    assert clf_report.detected_task == "binary_classification"

    # Test regression readiness
    reg_report = audit_ml_readiness(regression_data)
    assert reg_report.is_ready is True
    assert reg_report.selected_target == "revenue"
    assert reg_report.detected_task == "regression"


def test_target_leakage_detection():
    df = pd.DataFrame({
        "sales": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "leaked_sales": [10.01, 19.99, 30.02, 39.98, 50.01, 60.02, 69.99, 80.01, 89.98, 100.01],
        "regular_feature": [1, 5, 2, 8, 3, 7, 4, 6, 2, 9],
    })
    leaks = detect_target_leakage(df, "sales")
    assert len(leaks) >= 1
    assert leaks[0]["column"] == "leaked_sales"
    assert leaks[0]["risk_level"] == "HIGH"


def test_preprocessing_pipeline(regression_data):
    feature_cols = ["order_date", "units", "unit_price", "marketing_spend", "region"]
    preprocessor, col_types = build_preprocessor(regression_data, feature_cols)
    preprocessor.fit(regression_data[feature_cols])

    trans = preprocessor.transform(regression_data[feature_cols])
    feature_names = get_feature_names_after_preprocessing(preprocessor)

    assert trans.shape[0] == len(regression_data)
    assert trans.shape[1] == len(feature_names)
    assert len(feature_names) > len(feature_cols)  # Due to one-hot and datetime features


def test_classification_training_and_evaluation(classification_data):
    feature_cols = ["age", "annual_income", "segment"]
    res = train_single_model(
        df=classification_data,
        target_col="churn",
        feature_cols=feature_cols,
        task_type="binary_classification",
        model_id="logistic_regression",
    )

    assert "accuracy" in res["metrics"]
    assert "f1_weighted" in res["metrics"]
    assert res["metrics"]["accuracy"] >= 0.5
    assert res["confusion_matrix"] is not None
    assert len(res["feature_importance"]) > 0
    assert res["pipeline"] is not None


def test_regression_training_and_evaluation(regression_data):
    feature_cols = ["units", "unit_price", "marketing_spend", "region"]
    res = train_single_model(
        df=regression_data,
        target_col="revenue",
        feature_cols=feature_cols,
        task_type="regression",
        model_id="ridge_regression",
    )

    assert "r2" in res["metrics"]
    assert res["metrics"]["r2"] > 0.85
    assert "mae" in res["metrics"]
    assert "rmse" in res["metrics"]
    assert res["is_overfitting"] is False


def test_model_selection_and_champion(regression_data):
    feature_cols = ["units", "unit_price", "marketing_spend"]
    bench = run_benchmark(
        df=regression_data,
        target_col="revenue",
        feature_cols=feature_cols,
        task_type="regression",
        compute_profile="Fast",
    )

    assert len(bench["leaderboard"]) >= 2
    assert bench["champion_id"] is not None
    assert bench["leaderboard"][0]["is_champion"] is True
    assert bench["leaderboard"][0]["score"] > 0.80


def test_explainability(regression_data):
    feature_cols = ["units", "unit_price", "marketing_spend"]
    res = train_single_model(
        df=regression_data,
        target_col="revenue",
        feature_cols=feature_cols,
        task_type="regression",
        model_id="ridge_regression",
    )

    importance = res["feature_importance"]
    assert len(importance) > 0
    top_feature = importance[0]["feature"]
    assert top_feature in ["units", "unit_price", "marketing_spend"]

    # Local explanation
    input_row = {"units": 45, "unit_price": 95, "marketing_spend": 1500}
    loc_exp = explain_local_prediction(
        model=res["estimator"],
        preprocessor=res["preprocessor"],
        input_row=input_row,
        baseline_df=regression_data[feature_cols],
        predicted_value=5000.0,
        feature_names=feature_cols,
    )

    assert "summary" in loc_exp
    assert len(loc_exp["positive_drivers"]) > 0


def test_time_series_forecasting(regression_data):
    fc = run_time_series_forecast(
        df=regression_data,
        date_col="order_date",
        target_col="revenue",
        horizon=10,
    )

    assert len(fc["forecast"]) == 10
    assert "metrics" in fc
    assert "mae" in fc["metrics"]
    assert "trend_summary" in fc
    assert fc["forecast"][0]["lower_ci"] <= fc["forecast"][0]["predicted"] <= fc["forecast"][0]["upper_ci"]


def test_inference_and_immutability(regression_data):
    feature_cols = ["units", "unit_price", "marketing_spend"]
    res = train_single_model(
        df=regression_data,
        target_col="revenue",
        feature_cols=feature_cols,
        task_type="regression",
        model_id="ridge_regression",
    )

    pipeline = res["pipeline"]

    # Single Record
    single_res = predict_single_record(
        pipeline=pipeline,
        input_data={"units": 10, "unit_price": 50, "marketing_spend": 200},
        task_type="regression",
    )
    assert isinstance(single_res["prediction"], float)
    assert single_res["prediction"] > 0

    # Batch prediction & Immutability test
    orig_cols = list(regression_data.columns)
    batch_out = predict_batch(
        pipeline=pipeline,
        batch_df=regression_data,
        feature_cols=feature_cols,
        task_type="regression",
    )

    assert "model_prediction" in batch_out.columns
    assert len(batch_out) == len(regression_data)
    # Verify original dataframe is untouched
    assert list(regression_data.columns) == orig_cols
    assert "model_prediction" not in regression_data.columns


def test_experiment_tracker(classification_data):
    feature_cols = ["age", "annual_income"]
    res = train_single_model(
        df=classification_data,
        target_col="churn",
        feature_cols=feature_cols,
        task_type="binary_classification",
        model_id="logistic_regression",
    )

    exp_id = save_experiment(
        pipeline=res["pipeline"],
        model_id="logistic_regression",
        model_name="Logistic Regression",
        task_type="binary_classification",
        target_col="churn",
        feature_cols=feature_cols,
        metrics=res["metrics"],
        dataset_name="test_churn_dataset",
        is_champion=True,
    )

    manifest = get_experiment(exp_id)
    assert manifest is not None
    assert manifest["experiment_id"] == exp_id
    assert manifest["is_champion"] is True

    # Reload model from disk
    loaded_pipe = load_model(exp_id)
    test_pred = loaded_pipe.predict(pd.DataFrame([{"age": 35, "annual_income": 60000}]))
    assert len(test_pred) == 1


def test_xgboost_and_lightgbm_training(regression_data, classification_data):
    # Test XGBoost Regressor
    res_xgb = train_single_model(
        df=regression_data,
        target_col="revenue",
        feature_cols=["units", "unit_price", "marketing_spend"],
        task_type="regression",
        model_id="xgboost_regressor",
    )
    assert "r2" in res_xgb["metrics"]
    assert res_xgb["metrics"]["r2"] > 0.70

    # Test LightGBM Classifier
    res_lgb = train_single_model(
        df=classification_data,
        target_col="churn",
        feature_cols=["age", "annual_income"],
        task_type="binary_classification",
        model_id="lightgbm_classifier",
    )
    assert "accuracy" in res_lgb["metrics"]
    assert res_lgb["metrics"]["accuracy"] >= 0.50


def test_string_target_and_dataset_switching():
    # Test dataset with string targets like 'Female', 'Male'
    df_finance = pd.DataFrame({
        "age": [25, 34, 45, 52, 28, 38, 48, 60],
        "Investment_Avenues": ["Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "income": [40000, 60000, 75000, 90000, 45000, 80000, 70000, 110000],
        "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    })

    # Train XGBoost with string targets
    res = train_single_model(
        df=df_finance,
        target_col="gender",
        feature_cols=["age", "Investment_Avenues", "income"],
        task_type="binary_classification",
        model_id="xgboost_classifier",
    )
    assert "accuracy" in res["metrics"]
    assert res["pipeline"] is not None

    # Verify single prediction returns string label
    pred_res = predict_single_record(
        pipeline=res["pipeline"],
        input_data={"age": 30, "Investment_Avenues": "Yes", "income": 55000},
        task_type="binary_classification",
    )
    assert pred_res["prediction"] in ["Female", "Male"]

    # Test dataset switching resilience: batch prediction on a new dataset missing 'Investment_Avenues'
    df_new = pd.DataFrame({
        "age": [22, 33],
        "income": [35000, 65000],
    })
    batch_out = predict_batch(
        pipeline=res["pipeline"],
        batch_df=df_new,
        feature_cols=["age", "Investment_Avenues", "income"],
        task_type="binary_classification",
    )
    assert "model_prediction" in batch_out.columns
    assert len(batch_out) == 2
