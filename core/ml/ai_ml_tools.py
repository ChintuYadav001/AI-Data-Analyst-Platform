"""
core.ml.ai_ml_tools
===================
Deterministic tools for AI Assistant and automated ML orchestration.
Zero hallucination: only executes real mathematical and statistical pipelines.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from core.ml.ml_readiness import audit_ml_readiness, classify_task_type
from core.ml.model_selection import run_benchmark
from core.ml.forecasting import run_time_series_forecast
from core.ml.prediction import predict_single_record
from core.ml.experiment_tracker import get_active_champion, load_model, list_experiments, get_experiment


def prepare_ml_data(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Audit dataset health and readiness for machine learning."""
    report = audit_ml_readiness(df, target_col)
    return report.to_dict()


def train_and_benchmark(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    compute_profile: str = "Fast",
) -> Dict[str, Any]:
    """Train and rank multiple candidate models, returning champion and leaderboard."""
    task_type = classify_task_type(df, target_col)
    if feature_cols is None:
        report = audit_ml_readiness(df, target_col)
        # Exclude IDs, constant cols, and leakage
        exclude = set(report.id_columns + report.constant_columns + [target_col])
        for leak in report.leakage_warnings:
            if leak.get("risk_level") in ["CRITICAL", "HIGH"]:
                exclude.add(leak["column"])
        feature_cols = [c for c in df.columns if c not in exclude]

    res = run_benchmark(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        task_type=task_type,
        compute_profile=compute_profile,
    )

    # Return clean serializable summary
    return {
        "task_type": task_type,
        "champion_id": res["champion_id"],
        "champion_reason": res["champion_reason"],
        "leaderboard": res["leaderboard"],
    }


def forecast_metric(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 14,
) -> Dict[str, Any]:
    """Execute time-series forecast for a given metric and date column."""
    return run_time_series_forecast(
        df=df,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
    )


def predict_from_champion(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Use the currently active champion model to predict an input record."""
    champ = get_active_champion()
    if not champ:
        return {"error": "No active champion model registered yet. Please train a model first."}

    exp_id = champ["experiment_id"]
    pipeline = load_model(exp_id)
    return predict_single_record(
        pipeline=pipeline,
        input_data=input_dict,
        task_type=champ["task_type"],
    )


def generate_ml_executive_summary(df: pd.DataFrame, target_col: str) -> str:
    """Generate institutional executive briefing regarding ML capability on dataset."""
    report = audit_ml_readiness(df, target_col)
    task = report.detected_task.replace("_", " ").title()

    lines = [
        f"### 🤖 Quantum ML Executive Briefing: `{target_col}`",
        f"- **Identified ML Task**: {task}",
        f"- **Readiness Health Score**: {report.overall_score}/100 ({'Optimal' if report.overall_score >= 80 else 'Acceptable'})",
        f"- **Training Sample Scale**: {report.total_rows:,} rows across {report.total_columns} features.",
    ]

    if report.warnings:
        lines.append(f"- **Risk Disclosures**: {'; '.join(report.warnings[:2])}")

    if report.recommendations:
        lines.append(f"- **Recommended Next Action**: {report.recommendations[0]}")

    return "\n".join(lines)
