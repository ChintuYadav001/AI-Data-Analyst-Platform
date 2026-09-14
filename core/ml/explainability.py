"""
core.ml.explainability
======================
Global feature importance, local prediction attribution, permutation importance,
and statistical causation disclaimers for Quantum Analyst ML.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


STATISTICAL_DISCLAIMER = (
    "⚠️ Statistical Disclaimer: Feature importance and attribution reflect mathematical "
    "correlation and predictive contribution within this model's structure. "
    "They represent statistical associations and do not prove real-world causality."
)


def extract_global_feature_importance(
    model: Any,
    feature_names: List[str],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    max_features: int = 20,
) -> Dict[str, Any]:
    """
    Extract global feature importance using direct model attributes (tree importances,
    linear coefficients) or permutation importance fallback.
    """
    importances = None
    method = "direct"

    # Unwrap pipeline if needed
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model.steps[-1][1])

    # 1. Check tree-based feature_importances_
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        method = "tree_feature_importances"

    # 2. Check linear model coef_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
        method = "linear_coefficients"

    # 3. Fallback to permutation importance if validation set provided
    elif X_val is not None and y_val is not None and len(X_val) > 10:
        try:
            # Subsample if large
            idx = np.random.choice(len(X_val), min(500, len(X_val)), replace=False)
            perm = permutation_importance(estimator, X_val[idx], y_val[idx], n_repeats=3, random_state=42)
            importances = np.maximum(0, perm.importances_mean)
            method = "permutation_importance"
        except Exception:
            importances = None

    if importances is None or len(importances) == 0:
        # Equal importance default fallback
        importances = np.ones(len(feature_names)) / max(1, len(feature_names))
        method = "uniform_fallback"

    # Normalize to percentage
    importances = np.asarray(importances, dtype=float)
    total = np.sum(importances)
    if total > 0:
        norm_importances = (importances / total) * 100.0
    else:
        norm_importances = np.ones(len(importances)) * (100.0 / len(importances))

    # Match with feature names
    ranked = []
    for i, name in enumerate(feature_names[:len(norm_importances)]):
        ranked.append({
            "feature": name,
            "importance": round(float(norm_importances[i]), 2),
            "raw_score": round(float(importances[i]), 5),
        })

    # Sort descending
    ranked.sort(key=lambda x: x["importance"], reverse=True)

    # Add cumulative importance and rank
    cum = 0.0
    for r, item in enumerate(ranked, 1):
        cum += item["importance"]
        item["rank"] = r
        item["cumulative_importance"] = round(cum, 2)

    return {
        "ranked_features": ranked[:max_features],
        "method": method,
        "total_features": len(feature_names),
        "disclaimer": STATISTICAL_DISCLAIMER,
    }


def explain_local_prediction(
    model: Any,
    preprocessor: Any,
    input_row: Dict[str, Any],
    baseline_df: pd.DataFrame,
    predicted_value: Any,
    feature_names: List[str],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Produce intuitive local prediction explanation for a single record
    by calculating individual feature attribution relative to the population baseline.
    """
    row_df = pd.DataFrame([input_row])
    
    # Extract baseline means for numeric features
    pos_drivers = []
    neg_drivers = []

    # Get global feature importance weights
    global_exp = extract_global_feature_importance(model, feature_names)
    weight_map = {item["feature"]: item["importance"] for item in global_exp["ranked_features"]}

    # Identify features that diverge from baseline
    for col in input_row:
        if col not in baseline_df.columns:
            continue
        val = input_row[col]
        
        if pd.api.types.is_numeric_dtype(baseline_df[col]):
            baseline_mean = float(baseline_df[col].mean())
            baseline_std = float(baseline_df[col].std()) if baseline_df[col].std() > 0 else 1.0
            
            try:
                num_val = float(val)
                z_score = (num_val - baseline_mean) / baseline_std
                weight = weight_map.get(col, 5.0)
                attribution = round(z_score * (weight / 10.0), 3)

                entry = {
                    "feature": col,
                    "value": num_val,
                    "baseline_mean": round(baseline_mean, 2),
                    "attribution": attribution,
                    "direction": "Positive" if attribution > 0 else "Negative",
                }

                if attribution > 0.05:
                    pos_drivers.append(entry)
                elif attribution < -0.05:
                    neg_drivers.append(entry)
            except (ValueError, TypeError):
                pass
        else:
            # Categorical value match
            mode_val = baseline_df[col].mode()
            mode_str = str(mode_val.iloc[0]) if len(mode_val) > 0 else ""
            if str(val) != mode_str:
                entry = {
                    "feature": col,
                    "value": str(val),
                    "baseline_mean": f"Modal: {mode_str}",
                    "attribution": 1.0,
                    "direction": "Categorical Shift",
                }
                pos_drivers.append(entry)

    pos_drivers.sort(key=lambda x: abs(x["attribution"]), reverse=True)
    neg_drivers.sort(key=lambda x: abs(x["attribution"]), reverse=True)

    summary = (
        f"Prediction of {predicted_value} is most heavily supported by "
        f"{', '.join([d['feature'] for d in pos_drivers[:top_k]]) or 'baseline characteristics'}."
    )

    return {
        "predicted_value": predicted_value,
        "positive_drivers": pos_drivers[:top_k],
        "negative_drivers": neg_drivers[:top_k],
        "summary": summary,
        "disclaimer": STATISTICAL_DISCLAIMER,
    }
