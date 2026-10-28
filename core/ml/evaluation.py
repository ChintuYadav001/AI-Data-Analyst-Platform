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

    # Ratio of RMSE to MAE: > 1.40 indicates high-magnitude tail outliers dominate RMSE
    rmse_mae_ratio = round(rmse / (mae + 1e-6), 2)

    metrics = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "max_error": round(mx_err, 4),
        "mean_residual": round(mean_residual, 4),
        "std_residual": round(std_residual, 4),
        "rmse_mae_ratio": rmse_mae_ratio,
    }

    # Train vs Test metrics comparison (Underfitting / Overfitting diagnosis)
    train_metrics = {}
    train_r2 = None
    fit_status = "optimal"
    fit_message = "Model achieves balanced generalization between training and test sets."

    if y_train_true is not None and y_train_pred is not None:
        tr_mask = ~np.isnan(y_train_true) & ~np.isnan(y_train_pred)
        y_tr_t = np.asarray(y_train_true)[tr_mask]
        y_tr_p = np.asarray(y_train_pred)[tr_mask]

        if len(y_tr_t) > 0:
            train_r2 = float(r2_score(y_tr_t, y_tr_p))
            train_mae = float(mean_absolute_error(y_tr_t, y_tr_p))
            train_rmse = float(np.sqrt(mean_squared_error(y_tr_t, y_tr_p)))
            metrics["train_r2"] = round(train_r2, 4)
            metrics["train_mae"] = round(train_mae, 4)
            metrics["train_rmse"] = round(train_rmse, 4)
            train_metrics = {
                "train_r2": round(train_r2, 4),
                "train_mae": round(train_mae, 4),
                "train_rmse": round(train_rmse, 4),
            }

            # Check underfitting vs overfitting
            if (train_r2 < 0.40 and r2 < 0.40) or (r2 < 0):
                fit_status = "underfitting"
                fit_message = (
                    f"Underfitting Detected: Train R² is {train_r2:.3f} and Holdout R² is {r2:.3f}. "
                    "The model is too simple to capture patterns. Try non-linear ensembles (Random Forest, Gradient Boosting, XGBoost) and feature engineering."
                )
            elif (train_r2 - r2) > 0.15 and train_r2 > 0.65:
                fit_status = "overfitting"
                fit_message = (
                    f"Overfitting Detected: Train R² is {train_r2:.3f} vs Holdout R² of {r2:.3f} (gap > 0.15). "
                    "Model memorized training noise. Consider stronger regularization or pruning tree depth."
                )

    overfitting_alert = (fit_status == "overfitting")
    underfitting_alert = (fit_status == "underfitting")

    # Target Skewness Analysis
    try:
        from scipy.stats import skew as calc_skew
        target_skew = float(calc_skew(y_true))
    except Exception:
        # Fallback manual Fisher-Pearson skewness
        n_pts = len(y_true)
        if n_pts > 2 and np.std(y_true) > 0:
            target_skew = float((np.sum((y_true - np.mean(y_true)) ** 3) / n_pts) / (np.std(y_true) ** 3))
        else:
            target_skew = 0.0

    target_non_negative = bool(np.all(y_true >= 0)) if len(y_true) > 0 else False
    log_transform_recommended = (target_skew > 0.75 and target_non_negative)

    # High-Sales / Tail Underestimation Analysis
    # Look at top 10-15% highest actual observations
    has_high_sales_underestimation = False
    top_underestimated_records = []
    top_10_avg_underestimation = 0.0
    top_10_underestimated_pct = 0.0

    if len(y_true) >= 8:
        sorted_indices = np.argsort(y_true)
        top_k = max(2, int(len(y_true) * 0.15))
        top_indices = sorted_indices[-top_k:]

        top_actuals = y_true[top_indices]
        top_predictions = y_pred[top_indices]
        top_residuals = top_actuals - top_predictions  # positive means underestimation (actual > predicted)

        underestimated_mask = top_residuals > 0
        top_10_underestimated_pct = round(float(np.mean(underestimated_mask) * 100), 1)
        top_10_avg_underestimation = round(float(np.mean(top_residuals)), 3)

        # Flag underestimation if average residual on top sales is substantially positive
        std_y = np.std(y_true)
        if top_10_avg_underestimation > 0.12 * std_y and top_10_underestimated_pct >= 60.0:
            has_high_sales_underestimation = True

        # Extract worst underestimated records (highest positive residual)
        worst_order = np.argsort(top_residuals)[::-1]
        for w_idx in worst_order[:5]:
            act_val = float(top_actuals[w_idx])
            pred_val = float(top_predictions[w_idx])
            diff = act_val - pred_val
            pct_err = round((diff / (abs(act_val) + 1e-6)) * 100, 1)
            top_underestimated_records.append({
                "actual": round(act_val, 2),
                "predicted": round(pred_val, 2),
                "underestimation_amount": round(diff, 2),
                "percent_underestimated": f"+{pct_err}%",
            })

    # Assemble Warnings
    warnings = []
    if has_high_sales_underestimation:
        warnings.append(
            f"High-Sales Underestimation Detected: Model underestimates top 15% highest observations "
            f"by an average of {top_10_avg_underestimation:,.2f} units ({top_10_underestimated_pct}% of top observations underpredicted). "
            f"These tail errors inflate RMSE ({rmse:,.2f}) relative to MAE ({mae:,.2f})."
        )
    if fit_status == "underfitting":
        warnings.append(fit_message)
    elif fit_status == "overfitting":
        warnings.append(fit_message)
    if r2 < 0:
        warnings.append(f"Negative R² ({r2:.3f}): Model performs worse than a horizontal mean line. Features may lack predictive signal.")
    elif r2 < 0.20:
        warnings.append(f"Low R² ({r2:.3f}): Model captures limited variance. Consider engineering stronger interaction features.")

    # Subsample scatter points for clean UI charting (up to 300 points)
    n = len(y_true)
    step = max(1, n // 300)
    scatter_data = [
        {
            "actual": round(float(y_true[i]), 3),
            "predicted": round(float(y_pred[i]), 3),
            "residual": round(float(residuals[i]), 3),
            "is_top_sales": bool(i in sorted_indices[-top_k:]) if len(y_true) >= 8 else False,
        }
        for i in range(0, n, step)
    ]

    # Model improvement diagnostic summary package
    improvement_diagnostics = {
        "step_1_evaluation": {
            "test_mae": round(mae, 4),
            "test_rmse": round(rmse, 4),
            "test_r2": round(r2, 4),
            "train_mae": train_metrics.get("train_mae"),
            "train_rmse": train_metrics.get("train_rmse"),
            "train_r2": train_metrics.get("train_r2"),
            "rmse_mae_ratio": rmse_mae_ratio,
        },
        "step_2_outliers_and_high_sales": {
            "has_high_sales_underestimation": has_high_sales_underestimation,
            "top_10_avg_underestimation": top_10_avg_underestimation,
            "top_10_underestimated_pct": top_10_underestimated_pct,
            "worst_underpredicted_records": top_underestimated_records,
            "outlier_impact_explanation": (
                f"RMSE ({rmse:,.2f}) is {rmse_mae_ratio}x larger than MAE ({mae:,.2f}). "
                "Because RMSE squares errors, extreme underpredictions on high-sales observations heavily inflate RMSE."
            ),
        },
        "step_5_underfitting_check": {
            "status": fit_status,
            "is_underfitting": underfitting_alert,
            "is_overfitting": overfitting_alert,
            "train_r2": train_r2,
            "test_r2": round(r2, 4),
            "r2_gap": round(train_r2 - r2, 4) if train_r2 is not None else None,
            "diagnosis_message": fit_message,
        },
        "step_6_log_transformation": {
            "target_skew": round(target_skew, 3),
            "is_right_skewed": target_skew > 0.75,
            "target_non_negative": target_non_negative,
            "log_transform_recommended": log_transform_recommended,
            "recommendation_text": (
                f"Target skewness is {target_skew:.2f} (right-skewed). Training with np.log1p(y) and predicting "
                "with np.expm1() will compress extreme tail values, drastically stabilizing variance and preventing high-sales underestimation."
                if log_transform_recommended else
                f"Target skewness is {target_skew:.2f} (symmetrical). Standard linear or ensemble scaling is sufficient."
            ),
        },
    }

    return {
        "metrics": metrics,
        "warnings": warnings,
        "is_overfitting": overfitting_alert,
        "is_underfitting": underfitting_alert,
        "fit_status": fit_status,
        "scatter_sample": scatter_data,
        "improvement_diagnostics": improvement_diagnostics,
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
