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
from core.ml.forecasting import run_time_series_forecast
from core.ml.prediction import predict_single_record, predict_batch
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
    "predict_single_record",
    "predict_batch",
    "save_experiment",
    "list_experiments",
    "get_experiment",
    "load_model",
    "set_active_champion",
    "get_active_champion",
    "start_background_benchmark",
    "ai_ml_tools",
]
