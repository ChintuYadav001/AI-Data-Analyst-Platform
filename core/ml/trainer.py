"""
core.ml.trainer
===============
Training and validation orchestrator for Quantum Analyst ML. Handles reproducible
train/validation splitting, pipeline assembly, metric evaluation, and CV scoring.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from core.ml.preprocessing import build_preprocessor, get_feature_names_after_preprocessing
from core.ml.model_registry import create_model_instance, REGISTRY
from core.ml.evaluation import evaluate_regression, evaluate_classification
from core.ml.explainability import extract_global_feature_importance


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps any base classifier (e.g. XGBoost, LightGBM, RandomForest, LogisticRegression)
    to automatically encode targets to [0, 1, ...] and inverse-transform predictions,
    ensuring string, boolean, and object targets work flawlessly across all algorithms.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator
        self.label_encoder_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        # Convert to string array for robust encoding
        y_str = np.asarray(y).astype(str)
        y_enc = self.label_encoder_.fit_transform(y_str)
        self.classes_ = self.label_encoder_.classes_
        self.base_estimator.fit(X, y_enc)
        return self

    def predict(self, X):
        preds_enc = self.base_estimator.predict(X)
        try:
            return self.label_encoder_.inverse_transform(np.asarray(preds_enc, dtype=int))
        except Exception:
            return preds_enc

    def predict_proba(self, X):
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X)
        raise AttributeError("Underlying estimator does not support predict_proba")

    @property
    def feature_importances_(self):
        return getattr(self.base_estimator, "feature_importances_", None)

    @property
    def coef_(self):
        return getattr(self.base_estimator, "coef_", None)


def train_single_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: str,
    model_id: str,
    custom_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    cv_folds: int = 3,
    scaler_type: str = "standard",
    cat_encoding: str = "onehot",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and evaluates a single model pipeline with zero data leakage.
    Returns trained pipeline, validation metrics, feature importance, and diagnostic alerts.
    """
    start_time = time.perf_counter()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    # Clean missing values in target
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 8:
        raise ValueError(f"Insufficient valid rows ({len(clean_df)}) for training after dropping missing targets.")

    # Filter features
    valid_features = [c for c in feature_cols if c in clean_df.columns and c != target_col]
    if not valid_features:
        raise ValueError("No valid feature columns provided for training.")

    X = clean_df[valid_features]
    y = clean_df[target_col]

    is_classification = "classification" in task_type
    if is_classification:
        val_counts = y.value_counts()
        min_count = val_counts.min()
        stratify = y if min_count >= 2 else None
    else:
        stratify = None
        y = pd.to_numeric(y, errors="coerce")
        # Drop rows where y is NaN after numeric coercion
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask].astype(float)
        if len(y) < 8:
            raise ValueError(f"Insufficient valid numeric values in target '{target_col}'.")

    # Dynamic test size adjustment for small datasets
    actual_test_size = test_size
    if len(X) * test_size < 3:
        actual_test_size = max(0.15, min(0.35, 3.0 / len(X)))

    # Train / Test split with robust stratification fallback
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=actual_test_size, random_state=random_state, stratify=stratify
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=actual_test_size, random_state=random_state, stratify=None
        )

    # Build and fit preprocessing pipeline on X_train only
    preprocessor, col_types = build_preprocessor(
        df=X_train,
        feature_cols=valid_features,
        scaler_type=scaler_type,
        cat_encoding=cat_encoding,
    )
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    feature_names_out = get_feature_names_after_preprocessing(preprocessor)

    # Instantiate base estimator
    raw_estimator = create_model_instance(model_id, custom_params=custom_params)

    # Adjust KNN n_neighbors if training set is small
    if ("knn" in model_id) and hasattr(raw_estimator, "n_neighbors"):
        raw_estimator.n_neighbors = min(raw_estimator.n_neighbors, max(1, len(X_train) - 1))

    # Wrap classifier with LabelEncodedClassifier for robust string/boolean target handling
    if is_classification:
        estimator = LabelEncodedClassifier(base_estimator=raw_estimator)
        estimator.fit(X_train_trans, y_train.values)
    else:
        estimator = raw_estimator
        estimator.fit(X_train_trans, y_train.values)

    # Evaluate predictions
    y_pred_val = estimator.predict(X_val_trans)
    y_pred_train = estimator.predict(X_train_trans)

    if is_classification:
        y_proba_val = None
        if hasattr(estimator, "predict_proba"):
            try:
                y_proba_val = estimator.predict_proba(X_val_trans)
            except Exception:
                pass

        eval_result = evaluate_classification(
            y_true=y_val.values.astype(str),
            y_pred=y_pred_val.astype(str),
            y_proba=y_proba_val,
            classes=estimator.classes_ if hasattr(estimator, "classes_") else None,
            y_train_true=y_train.values.astype(str),
            y_train_pred=y_pred_train.astype(str),
        )
    else:
        eval_result = evaluate_regression(
            y_true=y_val.values,
            y_pred=y_pred_val,
            y_train_true=y_train.values,
            y_train_pred=y_pred_train,
        )

    # Compute Feature Importances
    feat_exp = extract_global_feature_importance(
        model=estimator,
        feature_names=feature_names_out,
        X_val=X_val_trans,
        y_val=y_val.values,
    )

    # Assemble End-to-End Pipeline for standalone inference
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    elapsed = round(time.perf_counter() - start_time, 3)

    return {
        "model_id": model_id,
        "model_name": REGISTRY.get(model_id, {}).name if model_id in REGISTRY else model_id,
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": valid_features,
        "preprocessed_features": feature_names_out,
        "metrics": eval_result["metrics"],
        "warnings": eval_result["warnings"],
        "is_overfitting": eval_result["is_overfitting"],
        "feature_importance": feat_exp["ranked_features"],
        "importance_disclaimer": feat_exp["disclaimer"],
        "pipeline": full_pipeline,
        "estimator": estimator,
        "preprocessor": preprocessor,
        "training_time_seconds": elapsed,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "confusion_matrix": eval_result.get("confusion_matrix"),
        "classes": eval_result.get("classes"),
        "scatter_sample": eval_result.get("scatter_sample"),
    }
