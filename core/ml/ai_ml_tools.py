"""
core.ml.ai_ml_tools
===================
14 Institutional Controlled Tools for AI Assistant and automated ML orchestration.
Guarantees zero hallucination: every tool executes deterministic mathematical,
statistical, or machine learning pipelines directly on real data.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from core.ml.ml_readiness import audit_ml_readiness, classify_task_type, detect_target_candidates
from core.ml.feature_selection import select_best_features
from core.ml.dimensionality import run_dimensionality_reduction
from core.ml.trainer import train_single_model
from core.ml.model_selection import run_benchmark
from core.ml.tuning import tune_model_hyperparameters
from core.ml.evaluation import evaluate_regression, evaluate_classification
from core.ml.explainability import extract_global_feature_importance, explain_local_prediction
from core.ml.forecasting import run_time_series_forecast
from core.ml.prediction import predict_single_record, predict_batch
from core.ml.scenario_analysis import simulate_what_if_scenario
from core.ml.report_generator import generate_ml_pdf_report, generate_ml_html_report, generate_ml_csv_summary
from core.ml.experiment_tracker import (
    get_active_champion,
    load_model,
    list_experiments,
    get_experiment,
    save_experiment,
)


def detect_ml_problem(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """1. Infers whether the data suggests regression, classification, clustering, or forecasting."""
    task = classify_task_type(df, target_col)
    return {
        "detected_task": task,
        "task_name": task.replace("_", " ").title(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }


def detect_target(df: pd.DataFrame) -> Dict[str, Any]:
    """2. Scans dataset to discover semantic target variable candidates with confidence scores."""
    candidates = detect_target_candidates(df)
    return {
        "candidates": [
            {
                "column": c.column,
                "task_type": c.task_type,
                "confidence": c.confidence,
                "reason": c.reason,
            }
            for c in candidates
        ],
        "top_recommendation": candidates[0].column if candidates else None,
    }


def prepare_ml_data(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """3. Performs deep health audit, missingness, cardinality, leakage, and ML readiness score."""
    report = audit_ml_readiness(df, target_col)
    return report.to_dict()


def select_features(
    df: pd.DataFrame,
    target_col: str,
    method: str = "mutual_info",
    k: int = 10,
) -> Dict[str, Any]:
    """4. Ranks and filters top predictors using statistical or tree-based feature selection."""
    return select_best_features(df, target_col, method=method, k=k)


def run_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    method: str = "pca",
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """5. Decomposes multi-dimensional feature space via PCA, SVD, Kernel PCA, ICA, or NMF."""
    res = run_dimensionality_reduction(df, method=method, n_components=n_components, target_col=target_col)
    return {
        "method": res["method"],
        "n_components": res["n_components"],
        "variance_retained": res["comparison_summary"].get("variance_retained_pct"),
        "compression_ratio": res["comparison_summary"].get("compression_ratio_pct"),
        "top_drivers": res.get("top_drivers", {}),
        "summary": f"Reduced {len(res['feature_cols'])} features to {res['n_components']} components retaining {res['comparison_summary'].get('variance_retained_pct')}% variance.",
    }


def train_models(
    df: pd.DataFrame,
    target_col: str,
    model_ids: Optional[List[str]] = None,
    compute_profile: str = "Balanced",
) -> Dict[str, Any]:
    """6. Trains multiple candidate algorithms across holdout validation and ranks them."""
    task_type = classify_task_type(df, target_col)
    res = run_benchmark(
        df=df,
        target_col=target_col,
        task_type=task_type,
        compute_profile=compute_profile,
        model_subset=model_ids,
    )
    return {
        "task_type": task_type,
        "champion_id": res["champion_id"],
        "champion_reason": res["champion_reason"],
        "leaderboard": res["leaderboard"],
    }


def compare_models(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """7. Compares model families and returns benchmark metrics table."""
    return train_models(df, target_col, compute_profile="Balanced")


def tune_model(
    df: pd.DataFrame,
    target_col: str,
    model_id: str,
    search_type: str = "random",
) -> Dict[str, Any]:
    """8. Optimizes hyperparameters for a specified model via Cross-Validated Grid/Random Search."""
    task_type = classify_task_type(df, target_col)
    return tune_model_hyperparameters(
        df=df,
        target_col=target_col,
        task_type=task_type,
        model_id=model_id,
        search_type=search_type,
    )


def evaluate_model(experiment_id: str) -> Dict[str, Any]:
    """9. Retrieves holdout validation metrics, confusion matrix, and overfitting diagnostics."""
    exp = get_experiment(experiment_id)
    if not exp:
        return {"error": f"Experiment '{experiment_id}' not found."}
    return {
        "experiment_id": experiment_id,
        "model_name": exp.get("model_name"),
        "metrics": exp.get("metrics"),
        "warnings": exp.get("warnings"),
        "is_overfitting": exp.get("is_overfitting"),
    }


def explain_model(experiment_id: str, row_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """10. Computes global feature importance or local prediction explanation."""
    exp = get_experiment(experiment_id)
    if not exp:
        return {"error": f"Experiment '{experiment_id}' not found."}
    if not row_data:
        return {
            "model_name": exp.get("model_name"),
            "global_importance": exp.get("feature_importance", []),
        }
    # Local attribution
    pipe = load_model(experiment_id)
    local_exp = explain_local_prediction(pipe, row_data, exp.get("task_type", "regression"))
    return {
        "model_name": exp.get("model_name"),
        "local_attribution": local_exp.get("attributions", []),
        "prediction": local_exp.get("prediction"),
    }


def predict(input_data: Dict[str, Any], experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """11. Generates model prediction for single record from active champion or specified experiment."""
    if not experiment_id:
        champ = get_active_champion()
        if not champ:
            return {"error": "No active champion model found. Train a model first."}
        experiment_id = champ["experiment_id"]
        task_type = champ["task_type"]
    else:
        exp = get_experiment(experiment_id)
        task_type = exp.get("task_type", "regression") if exp else "regression"

    pipe = load_model(experiment_id)
    return predict_single_record(pipeline=pipe, input_data=input_data, task_type=task_type)


def forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 14,
    model_type: str = "auto",
) -> Dict[str, Any]:
    """12. Computes future time series projection with confidence intervals and diagnostics."""
    return run_time_series_forecast(
        df=df,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        method=model_type,
    )


def run_scenario(
    base_record: Dict[str, Any],
    alterations: Dict[str, Any],
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """13. Evaluates counterfactual What-If scenario against trained pipeline."""
    if not experiment_id:
        champ = get_active_champion()
        if not champ:
            return {"error": "No active champion model found. Train a model first."}
        experiment_id = champ["experiment_id"]
        task_type = champ["task_type"]
    else:
        exp = get_experiment(experiment_id)
        task_type = exp.get("task_type", "regression") if exp else "regression"

    pipe = load_model(experiment_id)
    return simulate_what_if_scenario(
        pipeline=pipe,
        base_record=base_record,
        modifications=alterations,
        task_type=task_type,
    )


def generate_ml_report(
    df: pd.DataFrame,
    target_col: str,
    experiment_id: Optional[str] = None,
    forecast_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """14. Compiles institutional ML audit report in PDF, HTML, and CSV formats."""
    readiness = prepare_ml_data(df, target_col)
    exp = get_experiment(experiment_id) if experiment_id else get_active_champion()

    report_payload = {
        "dataset_name": "Active Dataset",
        "target_col": target_col,
        "task_type": readiness.get("detected_task"),
        "readiness_score": readiness.get("overall_score"),
        "total_rows": len(df),
        "feature_count": len(df.columns) - 1,
        "champion_name": exp.get("model_name") if exp else "None",
        "leaderboard": [exp] if exp else [],
        "feature_importance": exp.get("feature_importance", []) if exp else [],
        "forecast_results": forecast_res,
    }

    pdf_bytes = generate_ml_pdf_report(report_payload)
    html_str = generate_ml_html_report(report_payload)
    csv_str = generate_ml_csv_summary(report_payload)

    return {
        "pdf_bytes_length": len(pdf_bytes),
        "html_report": html_str,
        "csv_summary": csv_str,
        "summary": "Institutional report generated across PDF, HTML, and CSV formats.",
    }
