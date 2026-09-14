"""
tests.test_ml_advanced
=======================
Unit and integration tests for PCA & dimensionality reduction, feature selection,
anomaly detection, advanced feature engineering, multi-model forecasting,
what-if scenario analysis, report generation, and AI controlled tools.
"""

import numpy as np
import pandas as pd
import pytest

from core.ml.dimensionality import run_dimensionality_reduction
from core.ml.feature_selection import select_best_features
from core.ml.anomaly_detection import run_anomaly_detection
from core.ml.clustering import run_clustering, CLUSTERING_ALGORITHMS
from core.ml.feature_engineering import (
    apply_power_transform,
    apply_quantile_transform,
    generate_polynomial_features,
    generate_binned_features,
    generate_time_series_features,
)
from core.ml.forecasting import run_time_series_forecast, auto_detect_time_series_meta
from core.ml.scenario_analysis import simulate_what_if_scenario
from core.ml.report_generator import generate_ml_pdf_report, generate_ml_html_report, generate_ml_csv_summary
from core.ml.trainer import train_single_model
from core.ml.model_registry import REGISTRY, get_candidate_models, create_model_instance
import core.ml.ai_ml_tools as ai_tools


@pytest.fixture
def sample_numeric_df():
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(50, 15, n)
    x2 = x1 * 0.8 + np.random.normal(0, 5, n)
    x3 = np.random.exponential(10, n)
    x4 = np.random.uniform(1, 100, n)
    y = 2.5 * x1 + 1.2 * x2 - 0.5 * x3 + np.random.normal(0, 10, n)
    target_class = (y > np.median(y)).astype(str)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "feat_1": x1,
        "feat_2": x2,
        "feat_3": x3,
        "feat_4": x4,
        "target_reg": y,
        "target_clf": target_class,
    })


def test_dimensionality_reduction(sample_numeric_df):
    num_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
    # 1. PCA
    pca_res = run_dimensionality_reduction(sample_numeric_df, feature_cols=num_cols, method="pca", n_components=2)
    assert pca_res["method"] == "pca"
    assert len(pca_res["explained_variance_ratio"]) == 2
    assert len(pca_res["scree_data"]) == 2
    assert len(pca_res["proj_2d"]) == len(sample_numeric_df)
    assert pca_res["comparison_summary"]["original_features_count"] == 4
    assert pca_res["comparison_summary"]["reduced_dimensions_count"] == 2
    assert pca_res["comparison_summary"]["compression_ratio_pct"] == 50.0

    # 2. Truncated SVD
    svd_res = run_dimensionality_reduction(sample_numeric_df, feature_cols=num_cols, method="truncated_svd", n_components=2)
    assert svd_res["method"] == "truncated_svd"

    # 3. Fast ICA
    ica_res = run_dimensionality_reduction(sample_numeric_df, feature_cols=num_cols, method="fast_ica", n_components=2)
    assert ica_res["method"] == "fast_ica"

    # 4. NMF
    nmf_res = run_dimensionality_reduction(sample_numeric_df, feature_cols=num_cols, method="nmf", n_components=2)
    assert nmf_res["method"] == "nmf"


def test_feature_selection(sample_numeric_df):
    # Test mutual info
    mi_res = select_best_features(sample_numeric_df, target_col="target_reg", method="mutual_info", k=2)
    assert len(mi_res["selected_features"]) == 2
    assert len(mi_res["ranked_table"]) >= 4

    # Test correlation
    corr_res = select_best_features(sample_numeric_df, target_col="target_reg", method="correlation", k=2)
    assert len(corr_res["selected_features"]) == 2

    # Test tree importance
    tree_res = select_best_features(sample_numeric_df, target_col="target_reg", method="tree_importance", k=3)
    assert len(tree_res["selected_features"]) == 3

    # Test variance threshold
    var_res = select_best_features(sample_numeric_df, target_col="target_reg", method="variance_threshold", k=2)
    assert len(var_res["selected_features"]) == 2


def test_anomaly_detection(sample_numeric_df):
    num_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
    # Isolation Forest
    iso_res = run_anomaly_detection(sample_numeric_df, feature_cols=num_cols, algorithm="isolation_forest", contamination=0.05)
    assert iso_res["anomaly_count"] > 0
    assert len(iso_res["scatter_points"]) > 0

    # LOF
    lof_res = run_anomaly_detection(sample_numeric_df, feature_cols=num_cols, algorithm="local_outlier_factor", contamination=0.05)
    assert lof_res["anomaly_count"] > 0

    # Elliptic Envelope
    ee_res = run_anomaly_detection(sample_numeric_df, feature_cols=num_cols, algorithm="elliptic_envelope", contamination=0.05)
    assert ee_res["anomaly_count"] > 0


def test_feature_engineering(sample_numeric_df):
    num_cols = ["feat_1", "feat_2"]
    # Power transform
    df_pt = apply_power_transform(sample_numeric_df, num_cols)
    assert "feat_1_yeojohnson" in df_pt.columns

    # Quantile transform
    df_qt = apply_quantile_transform(sample_numeric_df, num_cols)
    assert "feat_1_quantile_normal" in df_qt.columns

    # Polynomial interaction
    df_poly = generate_polynomial_features(sample_numeric_df, num_cols, degree=2)
    assert any("poly_" in col for col in df_poly.columns)

    # Binning
    df_bin = generate_binned_features(sample_numeric_df, num_cols, n_bins=4)
    assert "feat_1_bin_4" in df_bin.columns

    # Time series lag features
    df_ts = generate_time_series_features(sample_numeric_df, target_col="target_reg", lags=[1, 2], date_col="date")
    assert "target_reg_lag_1" in df_ts.columns
    assert "date_year" in df_ts.columns


def test_forecasting_engine(sample_numeric_df):
    meta = auto_detect_time_series_meta(sample_numeric_df, date_col="date", target_col="target_reg")
    assert meta["valid"] is True
    assert "trend" in meta
    assert "stationarity" in meta

    fc_res = run_time_series_forecast(
        df=sample_numeric_df,
        date_col="date",
        target_col="target_reg",
        horizon=7,
        method="auto",
    )
    assert len(fc_res["forecast"]) == 7
    assert fc_res["best_model"] is not None
    assert "mae" in fc_res["metrics"]
    assert len(fc_res["leaderboard"]) >= 3


def test_expanded_algorithm_training(sample_numeric_df):
    feat_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
    # Huber Regressor
    huber_res = train_single_model(
        df=sample_numeric_df,
        target_col="target_reg",
        feature_cols=feat_cols,
        task_type="regression",
        model_id="huber_regressor",
    )
    assert "mae" in huber_res["metrics"]

    # Bayesian Ridge
    bayes_res = train_single_model(
        df=sample_numeric_df,
        target_col="target_reg",
        feature_cols=feat_cols,
        task_type="regression",
        model_id="bayesian_ridge",
    )
    assert "r2" in bayes_res["metrics"]

    # KNN Regressor
    knn_res = train_single_model(
        df=sample_numeric_df,
        target_col="target_reg",
        feature_cols=feat_cols,
        task_type="regression",
        model_id="knn_regressor",
    )
    assert "rmse" in knn_res["metrics"]

    # SGD Classifier
    sgd_res = train_single_model(
        df=sample_numeric_df,
        target_col="target_clf",
        feature_cols=feat_cols,
        task_type="binary_classification",
        model_id="sgd_classifier",
    )
    assert "accuracy" in sgd_res["metrics"]


def test_what_if_scenario(sample_numeric_df):
    feat_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
    trained = train_single_model(
        df=sample_numeric_df,
        target_col="target_reg",
        feature_cols=feat_cols,
        task_type="regression",
        model_id="ridge_regression",
    )
    pipe = trained["pipeline"]
    base_rec = sample_numeric_df[feat_cols].iloc[0].to_dict()
    scen_res = simulate_what_if_scenario(
        pipeline=pipe,
        base_record=base_rec,
        modifications={"feat_1": base_rec["feat_1"] * 1.5},
        task_type="regression",
    )
    assert scen_res["success"] is True
    assert scen_res["comparison"]["delta"] is not None
    assert len(scen_res["feature_impacts"]) >= 1


def test_report_generation():
    dummy_payload = {
        "dataset_name": "Test Dataset",
        "target_col": "target_reg",
        "task_type": "regression",
        "readiness_score": 92,
        "total_rows": 1000,
        "feature_count": 8,
        "champion_name": "Ridge Regression",
        "leaderboard": [
            {"model_id": "ridge_regression", "name": "Ridge Regression", "primary_metric_val": 0.88, "is_champion": True},
            {"model_id": "random_forest_regressor", "name": "Random Forest", "primary_metric_val": 0.85, "is_champion": False},
        ],
        "feature_importance": [
            {"rank": 1, "feature": "feat_1", "importance": 0.45},
            {"rank": 2, "feature": "feat_2", "importance": 0.32},
        ],
    }
    pdf_bytes = generate_ml_pdf_report(dummy_payload)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 200

    html_text = generate_ml_html_report(dummy_payload)
    assert "QUANTUM ANALYST" in html_text
    assert "Ridge Regression" in html_text

    csv_text = generate_ml_csv_summary(dummy_payload)
    assert "ridge_regression" in csv_text


def test_ai_ml_tools(sample_numeric_df):
    # 1. detect_ml_problem
    prob = ai_tools.detect_ml_problem(sample_numeric_df, "target_reg")
    assert prob["detected_task"] == "regression"

    # 2. detect_target
    targets = ai_tools.detect_target(sample_numeric_df)
    assert "candidates" in targets

    # 3. prepare_ml_data
    ready = ai_tools.prepare_ml_data(sample_numeric_df, "target_reg")
    assert ready["overall_score"] > 50

    # 4. select_features
    fs = ai_tools.select_features(sample_numeric_df, "target_reg", method="correlation", k=2)
    assert len(fs["selected_features"]) == 2

    # 5. run_pca
    pca_tool = ai_tools.run_pca(sample_numeric_df, n_components=2)
    assert pca_tool["n_components"] == 2

    # 6. forecast
    fc = ai_tools.forecast(sample_numeric_df, date_col="date", target_col="target_reg", horizon=5)
    assert len(fc["forecast"]) == 5

    # 7. generate_ml_report
    rep = ai_tools.generate_ml_report(sample_numeric_df, "target_reg")
    assert rep["pdf_bytes_length"] > 100

    # 8. cluster_data
    cl_res = ai_tools.cluster_data(sample_numeric_df, algorithm="kmeans", n_clusters=3)
    assert cl_res["n_clusters"] == 3
    assert "silhouette_score" in cl_res

    # 9. detect_anomalies
    anom_res = ai_tools.detect_anomalies(sample_numeric_df, algorithm="isolation_forest", contamination=0.05)
    assert anom_res["anomaly_count"] > 0


def test_clustering(sample_numeric_df):
    num_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
    # 1. K-Means
    km_res = run_clustering(sample_numeric_df, feature_cols=num_cols, algorithm="kmeans", n_clusters=3)
    assert km_res["n_clusters"] == 3
    assert len(km_res["labels"]) == len(sample_numeric_df)
    assert "silhouette_score" in km_res
    assert len(km_res["cluster_summary"]) == 3

    # 2. DBSCAN
    db_res = run_clustering(sample_numeric_df, feature_cols=num_cols, algorithm="dbscan", eps=1.5, min_samples=3)
    assert "n_clusters" in db_res
    assert "noise_count" in db_res

    # 3. Agglomerative
    agg_res = run_clustering(sample_numeric_df, feature_cols=num_cols, algorithm="agglomerative", n_clusters=2)
    assert agg_res["n_clusters"] == 2
    assert -1.0 <= agg_res["silhouette_score"] <= 1.0

