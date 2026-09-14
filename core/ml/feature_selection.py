"""
core.ml.feature_selection
=========================
Advanced Feature Selection Engine for Quantum Analyst ML.
Supports Variance Threshold, Target Correlation, Mutual Information,
Chi-Square, RFE/RFECV, L1 Sparsity, and Tree Feature Importance.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    SelectKBest,
    RFE,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer


FEATURE_SELECTION_METHODS = {
    "mutual_info": "Mutual Information (Non-linear Dependency)",
    "correlation": "Pearson / Spearman Target Correlation",
    "tree_importance": "Tree-based MDI (Extra Trees Ensemble)",
    "variance_threshold": "Variance Threshold (Information Entropy)",
    "chi2": "Chi-Square Independence Test",
    "rfe": "Recursive Feature Elimination (RFE)",
    "l1_penalty": "L1 Lasso / Sparse Logistic Regularization",
}


def select_best_features(
    df: pd.DataFrame,
    target_col: str,
    method: str = "mutual_info",
    k: int = 10,
    task_type: Optional[str] = None,
    variance_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Ranks and selects the top-k most predictive features using statistical and model-driven techniques.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Clean target missingness
    clean_df = df.dropna(subset=[target_col]).copy()
    candidate_features = [c for c in clean_df.columns if c != target_col and pd.api.types.is_numeric_dtype(clean_df[c])]

    if len(candidate_features) < 2:
        raise ValueError("Feature selection requires at least 2 numeric feature candidates.")

    # Determine task type if not provided
    if not task_type:
        n_unique = clean_df[target_col].nunique()
        task_type = "classification" if n_unique <= 10 or not pd.api.types.is_numeric_dtype(clean_df[target_col]) else "regression"

    is_classif = "classification" in task_type

    X_raw = clean_df[candidate_features]
    imputer = SimpleImputer(strategy="median")
    X_mat = imputer.fit_transform(X_raw)

    y_raw = clean_df[target_col]
    if is_classif:
        # Convert target to integer codes
        y_vec = pd.Categorical(y_raw).codes
    else:
        y_vec = pd.to_numeric(y_raw, errors="coerce").fillna(0.0).values

    num_features = len(candidate_features)
    k_selected = max(1, min(k, num_features))

    feature_scores: Dict[str, float] = {}

    # 1. Variance Threshold
    if method == "variance_threshold":
        vt = VarianceThreshold(threshold=variance_threshold)
        vt.fit(X_mat)
        variances = vt.variances_
        for feat, var in zip(candidate_features, variances):
            feature_scores[feat] = float(var)

    # 2. Target Correlation
    elif method == "correlation":
        for i, feat in enumerate(candidate_features):
            feat_vals = X_mat[:, i]
            if np.std(feat_vals) > 1e-9 and np.std(y_vec) > 1e-9:
                corr = float(np.corrcoef(feat_vals, y_vec)[0, 1])
                feature_scores[feat] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                feature_scores[feat] = 0.0

    # 3. Mutual Information
    elif method == "mutual_info":
        if is_classif:
            mi = mutual_info_classif(X_mat, y_vec, random_state=42)
        else:
            mi = mutual_info_regression(X_mat, y_vec, random_state=42)
        for feat, score in zip(candidate_features, mi):
            feature_scores[feat] = float(score)

    # 4. Chi-Square (Requires non-negative values)
    elif method == "chi2":
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(X_mat)
        chi2_scores, _ = chi2(X_pos, y_vec if is_classif else np.digitize(y_vec, np.percentile(y_vec, [25, 50, 75])))
        for feat, score in zip(candidate_features, chi2_scores):
            feature_scores[feat] = float(score) if not np.isnan(score) else 0.0

    # 5. Tree Importance (Extra Trees)
    elif method == "tree_importance":
        if is_classif:
            tree_model = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            tree_model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        tree_model.fit(X_mat, y_vec)
        importances = tree_model.feature_importances_
        for feat, imp in zip(candidate_features, importances):
            feature_scores[feat] = float(imp)

    # 6. L1 Penalty (Lasso / L1 Logistic)
    elif method == "l1_penalty":
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_mat)
        if is_classif:
            l1_mod = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42)
            l1_mod.fit(X_std, y_vec)
            coefs = np.mean(np.abs(l1_mod.coef_), axis=0) if l1_mod.coef_.ndim > 1 else np.abs(l1_mod.coef_[0])
        else:
            l1_mod = Lasso(alpha=0.1, random_state=42)
            l1_mod.fit(X_std, y_vec)
            coefs = np.abs(l1_mod.coef_)
        for feat, c_val in zip(candidate_features, coefs):
            feature_scores[feat] = float(c_val)

    # 7. RFE
    elif method == "rfe":
        base_est = ExtraTreesClassifier(n_estimators=30, random_state=42) if is_classif else ExtraTreesRegressor(n_estimators=30, random_state=42)
        rfe_selector = RFE(estimator=base_est, n_features_to_select=k_selected, step=1)
        rfe_selector.fit(X_mat, y_vec)
        # Higher score for lower rank (rank 1 is best)
        max_rank = max(rfe_selector.ranking_)
        for feat, rank in zip(candidate_features, rfe_selector.ranking_):
            score = float((max_rank - rank + 1) / max_rank)
            feature_scores[feat] = score
    else:
        raise ValueError(f"Unknown feature selection method '{method}'.")

    # Sort and rank features
    sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    selected_list = [item[0] for item in sorted_features[:k_selected]]
    dropped_list = [item[0] for item in sorted_features[k_selected:]]

    ranked_table = []
    for rank, (feat, score) in enumerate(sorted_features, start=1):
        is_sel = feat in selected_list
        ranked_table.append({
            "rank": rank,
            "feature": feat,
            "score": round(score, 4),
            "selected": is_sel,
        })

    summary = (
        f"Selected {len(selected_list)} of {num_features} numeric features using {FEATURE_SELECTION_METHODS.get(method, method)}. "
        f"Top predictors: {', '.join(selected_list[:3])}."
    )

    return {
        "method": method,
        "method_name": FEATURE_SELECTION_METHODS.get(method, method),
        "target_col": target_col,
        "total_evaluated": num_features,
        "k_selected": len(selected_list),
        "selected_features": selected_list,
        "dropped_features": dropped_list,
        "ranked_table": ranked_table,
        "summary": summary,
    }
