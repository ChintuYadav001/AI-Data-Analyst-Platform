"""
core.ml
=======
Institutional-grade Machine Learning & Predictive Analytics Engine for Quantum Analyst.
"""

from core.ml.ml_readiness import audit_ml_readiness, classify_task_type, detect_target_candidates
from core.ml.preprocessing import build_preprocessor
from core.ml.model_registry import REGISTRY, get_candidate_models, create_model_instance
from core.ml.trainer import train_single_model
from core.ml.model_selection import run_benchmark
from core.ml.tuning import tune_model_hyperparameters
from core.ml.evaluation import evaluate_regression, evaluate_classification, evaluate_clustering
from core.ml.explainability import extract_global_feature_importance, explain_local_prediction
from core.ml.forecasting import run_time_series_forecast, auto_detect_time_series_meta
from core.ml.prediction import predict_single_record, predict_batch
from core.ml.dimensionality import run_dimensionality_reduction
from core.ml.feature_selection import select_best_features
from core.ml.feature_engineering import (
    apply_power_transform,
    apply_quantile_transform,
    generate_polynomial_features,
    generate_binned_features,
    generate_time_series_features,
)
from core.ml.anomaly_detection import run_anomaly_detection
from core.ml.scenario_analysis import simulate_what_if_scenario
from core.ml.report_generator import generate_ml_pdf_report, generate_ml_html_report, generate_ml_csv_summary
from core.ml.experiment_tracker import (
    save_experiment,
    list_experiments,
    get_experiment,
    load_model,
    set_active_champion,
    get_active_champion,
)
from core.ml.ml_tasks import start_background_benchmark
import core.ml.ai_ml_tools as ai_ml_tools

__all__ = [
    "audit_ml_readiness",
    "classify_task_type",
    "detect_target_candidates",
    "build_preprocessor",
    "REGISTRY",
    "get_candidate_models",
    "create_model_instance",
    "train_single_model",
    "run_benchmark",
    "tune_model_hyperparameters",
    "evaluate_regression",
    "evaluate_classification",
    "evaluate_clustering",
    "extract_global_feature_importance",
    "explain_local_prediction",
    "run_time_series_forecast",
    "auto_detect_time_series_meta",
    "predict_single_record",
    "predict_batch",
    "run_dimensionality_reduction",
    "select_best_features",
    "apply_power_transform",
    "apply_quantile_transform",
    "generate_polynomial_features",
    "generate_binned_features",
    "generate_time_series_features",
    "run_anomaly_detection",
    "simulate_what_if_scenario",
    "generate_ml_pdf_report",
    "generate_ml_html_report",
    "generate_ml_csv_summary",
    "save_experiment",
    "list_experiments",
    "get_experiment",
    "load_model",
    "set_active_champion",
    "get_active_champion",
    "start_background_benchmark",
    "ai_ml_tools",
]
