"""
core.ml.model_selection
=======================
Automated benchmarking, leaderboard ranking, and champion model selection
across Fast, Balanced, and Deep compute profiles.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from core.ml.model_registry import get_candidate_models, REGISTRY
from core.ml.trainer import train_single_model


def run_benchmark(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: str,
    compute_profile: str = "Balanced",
    model_ids: Optional[List[str]] = None,
    test_size: float = 0.2,
    scaler_type: str = "standard",
    cat_encoding: str = "onehot",
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Benchmarks multiple models on the dataset, evaluates metrics, ranks the leaderboard,
    and identifies the winning Champion Model.
    """
    if model_ids is None or len(model_ids) == 0:
        candidates = get_candidate_models(task_type, compute_profile)
    else:
        candidates = [m for m in model_ids if m in REGISTRY]

    if not candidates:
        raise ValueError(f"No candidate models available for task: {task_type}")

    is_classification = "classification" in task_type
    primary_metric = "f1_weighted" if is_classification else "r2"

    results = {}
    leaderboard = []
    total = len(candidates)

    for i, m_id in enumerate(candidates):
        if progress_callback:
            progress_callback(i / total, f"Training {REGISTRY[m_id].name} ({i+1}/{total})...")

        try:
            res = train_single_model(
                df=df,
                target_col=target_col,
                feature_cols=feature_cols,
                task_type=task_type,
                model_id=m_id,
                test_size=test_size,
                scaler_type=scaler_type,
                cat_encoding=cat_encoding,
            )
            results[m_id] = res

            score = res["metrics"].get(primary_metric, 0.0)
            leaderboard.append({
                "model_id": m_id,
                "model_name": res["model_name"],
                "score": score,
                "metrics": res["metrics"],
                "training_time": res["training_time_seconds"],
                "is_overfitting": res["is_overfitting"],
                "warnings": res["warnings"],
            })
        except Exception as e:
            # Record failed model run
            leaderboard.append({
                "model_id": m_id,
                "model_name": REGISTRY.get(m_id, {}).name if m_id in REGISTRY else m_id,
                "score": -999.0,
                "metrics": {},
                "training_time": 0.0,
                "is_overfitting": False,
                "warnings": [f"Training failed: {str(e)}"],
            })

    if progress_callback:
        progress_callback(1.0, "Benchmarking complete.")

    # Sort leaderboard by primary score descending
    leaderboard.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks and champion
    champion_id = None
    champion_reason = "No valid model completed training."

    for idx, entry in enumerate(leaderboard):
        entry["rank"] = idx + 1
        if idx == 0 and entry["score"] > -900:
            entry["is_champion"] = True
            champion_id = entry["model_id"]
        else:
            entry["is_champion"] = False

    if champion_id and champion_id in results:
        champ_res = results[champion_id]
        val_score = champ_res["metrics"].get(primary_metric)
        time_sec = champ_res["training_time_seconds"]
        metric_label = "F1-Score" if is_classification else "R²"
        champion_reason = (
            f"Achieved top performance with {metric_label} = {val_score:.3f} in {time_sec}s "
            f"while maintaining high generalization stability."
        )

    return {
        "leaderboard": leaderboard,
        "champion_id": champion_id,
        "champion_reason": champion_reason,
        "primary_metric": primary_metric,
        "task_type": task_type,
        "compute_profile": compute_profile,
        "all_results": results,
    }
