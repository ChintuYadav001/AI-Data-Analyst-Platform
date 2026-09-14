"""
core.ml.tuning
==============
Automated hyperparameter optimization using RandomizedSearchCV and GridSearchCV.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from core.ml.model_registry import REGISTRY, create_model_instance
from core.ml.trainer import train_single_model


def tune_model_hyperparameters(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: str,
    model_id: str,
    search_type: str = "random",  # random or grid
    n_iter: int = 8,
    cv_folds: int = 3,
    scaler_type: str = "standard",
) -> Dict[str, Any]:
    """
    Tunes hyperparameters for a specified model using the curated parameter search space.
    """
    model_info = REGISTRY.get(model_id)
    if not model_info or not model_info.param_grid:
        # If model has no tuning grid, return standard training
        res = train_single_model(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
            model_id=model_id,
            scaler_type=scaler_type,
        )
        return {
            "tuned": False,
            "message": f"No hyperparameter search space defined for {model_id}.",
            "best_params": model_info.default_params if model_info else {},
            "result": res,
        }

    start_time = time.perf_counter()

    # Train baseline first for metric comparison
    baseline_res = train_single_model(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        task_type=task_type,
        model_id=model_id,
        scaler_type=scaler_type,
    )

    # Preprocess training data
    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df[feature_cols]
    y = clean_df[target_col]
    if "classification" not in task_type:
        y = y.astype(float)

    # Use the fitted preprocessor from baseline
    preprocessor = baseline_res["preprocessor"]
    X_trans = preprocessor.transform(X)

    raw_estimator = create_model_instance(model_id)
    scoring = "f1_weighted" if "classification" in task_type else "r2"

    if search_type == "grid":
        search = GridSearchCV(
            estimator=raw_estimator,
            param_grid=model_info.param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )
    else:
        # Cap iterations to total grid size
        total_combos = 1
        for v in model_info.param_grid.values():
            total_combos *= len(v)
        actual_iter = min(n_iter, total_combos)

        search = RandomizedSearchCV(
            estimator=raw_estimator,
            param_distributions=model_info.param_grid,
            n_iter=actual_iter,
            cv=cv_folds,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
        )

    search.fit(X_trans, y)
    best_params = search.best_params_

    # Train final single model with best params
    tuned_res = train_single_model(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        task_type=task_type,
        model_id=model_id,
        custom_params=best_params,
        scaler_type=scaler_type,
    )

    elapsed = round(time.perf_counter() - start_time, 2)
    base_score = baseline_res["metrics"].get(scoring, 0.0)
    tuned_score = tuned_res["metrics"].get(scoring, 0.0)
    improvement = round(tuned_score - base_score, 4)

    return {
        "tuned": True,
        "best_params": best_params,
        "base_score": base_score,
        "tuned_score": tuned_score,
        "improvement": improvement,
        "elapsed_seconds": elapsed,
        "primary_metric": scoring,
        "baseline_result": baseline_res,
        "tuned_result": tuned_res,
    }
