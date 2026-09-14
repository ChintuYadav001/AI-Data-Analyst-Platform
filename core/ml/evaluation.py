"""
core.ml.evaluation
==================
Evaluation metrics, diagnostic warnings, confusion matrices, and residual
diagnostics for regression, classification, and clustering.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    max_error,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_train_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive regression metrics and diagnostic warnings."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Filter NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = 0.0

    mx_err = float(max_error(y_true, y_pred))

    residuals = (y_true - y_pred).tolist()
    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    std_residual = float(np.std(residuals)) if residuals else 0.0

    metrics = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "max_error": round(mx_err, 4),
        "mean_residual": round(mean_residual, 4),
        "std_residual": round(std_residual, 4),
    }

    # Overfitting Check
    overfitting_alert = False
    train_r2 = None
    if y_train_true is not None and y_train_pred is not None:
        tr_mask = ~np.isnan(y_train_true) & ~np.isnan(y_train_pred)
        train_r2 = float(r2_score(y_train_true[tr_mask], y_train_pred[tr_mask]))
        metrics["train_r2"] = round(train_r2, 4)
        if (train_r2 - r2) > 0.15 and train_r2 > 0.70:
            overfitting_alert = True

    warnings = []
    if r2 < 0:
        warnings.append(f"Negative R² ({r2:.3f}): Model performs worse than a horizontal mean line. Features may lack predictive signal.")
    elif r2 < 0.20:
        warnings.append(f"Low R² ({r2:.3f}): Model captures limited variance. Consider engineering stronger interaction features.")
    if overfitting_alert:
        warnings.append(f"Overfitting Detected: Train R² is {train_r2:.3f} vs Validation R² of {r2:.3f} (gap > 0.15).")

    # Subsample scatter points for clean UI charting (up to 300 points)
    n = len(y_true)
    step = max(1, n // 300)
    scatter_data = [
        {"actual": round(float(y_true[i]), 3), "predicted": round(float(y_pred[i]), 3), "residual": round(float(residuals[i]), 3)}
        for i in range(0, n, step)
    ]

    return {
        "metrics": metrics,
        "warnings": warnings,
        "is_overfitting": overfitting_alert,
        "scatter_sample": scatter_data,
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[Any]] = None,
    y_train_true: Optional[np.ndarray] = None,
    y_train_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics, confusion matrix, and alerts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec_weighted, 4),
        "recall": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
    }

    # ROC AUC & PR AUC
    roc_auc = None
    pr_auc = None
    if y_proba is not None:
        try:
            unq = np.unique(y_true)
            if len(unq) == 2:
                # Binary classification
                # Use probability of positive class
                pos_proba = y_proba[:, 1] if (y_proba.ndim == 2 and y_proba.shape[1] >= 2) else y_proba
                roc_auc = float(roc_auc_score(y_true, pos_proba))
                pr_auc = float(average_precision_score(y_true, pos_proba))
            else:
                # Multiclass
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass

    if roc_auc is not None:
        metrics["roc_auc"] = round(roc_auc, 4)
    if pr_auc is not None:
        metrics["pr_auc"] = round(pr_auc, 4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [str(c) for c in (classes if classes is not None else np.unique(y_true))]

    # Overfitting Check
    overfitting_alert = False
    train_acc = None
    if y_train_true is not None and y_train_pred is not None:
        train_acc = float(accuracy_score(y_train_true, y_train_pred))
        metrics["train_accuracy"] = round(train_acc, 4)
        if (train_acc - acc) > 0.15 and train_acc > 0.80:
            overfitting_alert = True

    warnings = []
    if acc < 0.55:
        warnings.append(f"Low Accuracy ({acc:.1%}): Model is marginally better than random guessing.")
    if abs(prec_weighted - rec_weighted) > 0.25:
        warnings.append("High Precision/Recall disparity: Inspect confusion matrix for class-specific false positives/negatives.")
    if overfitting_alert:
        warnings.append(f"Overfitting Detected: Train Accuracy is {train_acc:.1%} vs Validation Accuracy of {acc:.1%} (gap > 15%).")

    return {
        "metrics": metrics,
        "warnings": warnings,
        "is_overfitting": overfitting_alert,
        "confusion_matrix": cm.tolist(),
        "classes": class_labels,
    }


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Calculate unsupervised clustering metrics."""
    # Need at least 2 distinct clusters and less than total samples
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(X):
        return {
            "metrics": {"clusters": n_labels},
            "warnings": ["Clustering produced invalid number of clusters (<2 or >=N)."],
        }

    # Subsample if X is very large to compute silhouette score quickly
    if len(X) > 3000:
        idx = np.random.choice(len(X), 3000, replace=False)
        sil = float(silhouette_score(X[idx], labels[idx]))
    else:
        sil = float(silhouette_score(X, labels))

    cal = float(calinski_harabasz_score(X, labels))
    dav = float(davies_bouldin_score(X, labels))

    return {
        "metrics": {
            "clusters": int(n_labels),
            "silhouette_score": round(sil, 4),
            "calinski_harabasz_score": round(cal, 2),
            "davies_bouldin_index": round(dav, 4),
        },
        "warnings": [],
    }
