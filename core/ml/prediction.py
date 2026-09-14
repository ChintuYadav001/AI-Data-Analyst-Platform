"""
core.ml.prediction
==================
Inference engine supporting single-record interactive simulation and batch CSV/XLSX
predictions with zero mutation of source datasets.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def predict_single_record(
    pipeline: Any,
    input_data: Dict[str, Any],
    task_type: str,
) -> Dict[str, Any]:
    """
    Executes inference for a single input record.
    Returns predicted target and class probabilities (for classification).
    """
    row_df = pd.DataFrame([input_data])
    is_classification = "classification" in task_type

    raw_pred = pipeline.predict(row_df)[0]

    probabilities = {}
    confidence = None

    if is_classification:
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(row_df)[0]
                classes = pipeline.classes_ if hasattr(pipeline, "classes_") else [f"Class_{i}" for i in range(len(proba))]
                for c, p in zip(classes, proba):
                    probabilities[str(c)] = round(float(p), 4)
                confidence = float(np.max(proba))
            except Exception:
                pass

        prediction_val = str(raw_pred)
    else:
        prediction_val = round(float(raw_pred), 4)

    return {
        "prediction": prediction_val,
        "confidence": confidence,
        "probabilities": probabilities,
        "task_type": task_type,
    }


def predict_batch(
    pipeline: Any,
    batch_df: pd.DataFrame,
    feature_cols: List[str],
    task_type: str,
    output_pred_col: str = "model_prediction",
) -> pd.DataFrame:
    """
    Executes batch inference over an entire dataset.
    Returns a copy of batch_df with added prediction and probability columns.
    Original dataframe is NEVER mutated.
    """
    result_df = batch_df.copy()
    is_classification = "classification" in task_type

    # Ensure required features exist; fill missing columns with NaN
    eval_df = pd.DataFrame()
    for col in feature_cols:
        if col in result_df.columns:
            eval_df[col] = result_df[col]
        else:
            eval_df[col] = np.nan

    # Run predictions
    preds = pipeline.predict(eval_df)
    result_df[output_pred_col] = preds

    # Add probabilities if classifier
    if is_classification and hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(eval_df)
            if probas.shape[1] == 2:
                # Binary: record probability of positive class
                result_df[f"{output_pred_col}_confidence"] = np.round(np.max(probas, axis=1), 4)
                result_df[f"{output_pred_col}_positive_proba"] = np.round(probas[:, 1], 4)
            else:
                result_df[f"{output_pred_col}_confidence"] = np.round(np.max(probas, axis=1), 4)
        except Exception:
            pass

    return result_df
