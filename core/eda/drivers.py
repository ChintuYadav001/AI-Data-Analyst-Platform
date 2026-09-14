"""
Target Driver Association Engine.
Evaluates relationship strength between candidate predictors and target variable.
Strictly labels findings as association/predictive indicators with non-causality disclaimers.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats


class EDADriverEngine:
    """Evaluates and ranks predictive association strength for target metrics."""

    @classmethod
    def analyze_drivers(
        cls,
        df: pd.DataFrame,
        target_col: str,
        task_type: str = "regression"
    ) -> List[Dict[str, Any]]:
        """Calculates association strength between features and the target variable."""
        if target_col not in df.columns or len(df) < 10:
            return []

        target_series = df[target_col].dropna()
        if len(target_series) < 10:
            return []

        drivers: List[Dict[str, Any]] = []
        is_regression = (task_type == "regression") and pd.api.types.is_numeric_dtype(target_series)

        for col in df.columns:
            if col == target_col or any(k in str(col).lower() for k in ["_id", "id_", "identifier"]):
                continue

            series = df[col].dropna()
            if len(series) < 10:
                continue

            paired_df = df[[target_col, col]].dropna()
            if len(paired_df) < 10:
                continue

            # For large datasets (lakhs of rows), sample 25,000 pairs to eliminate O(N log N) rank sorting latency
            if len(paired_df) > 25000:
                eval_df = paired_df.sample(25000, random_state=42)
            else:
                eval_df = paired_df

            is_col_num = pd.api.types.is_numeric_dtype(eval_df[col])

            if is_regression:
                if is_col_num:
                    # Continuous Target vs Continuous Feature -> Pearson & Spearman
                    r_p, p_p = stats.pearsonr(eval_df[target_col], eval_df[col])
                    r_s, _ = stats.spearmanr(eval_df[target_col], eval_df[col])
                    score = abs(r_p) * 100

                    drivers.append({
                        "feature": str(col),
                        "feature_type": "Numerical",
                        "metric_type": "Pearson r & Spearman rho",
                        "correlation": round(float(r_p), 3),
                        "spearman": round(float(r_s), 3),
                        "p_value": round(float(p_p), 5),
                        "association_score": round(score, 1),
                        "relationship_type": "Positive Covariance" if r_p > 0 else "Inverse Trade-off",
                        "strength": "Strong" if score >= 60 else ("Moderate" if score >= 30 else "Mild")
                    })
                else:
                    # Continuous Target vs Categorical Feature -> ANOVA F-score
                    if 2 <= eval_df[col].nunique() <= 20:
                        groups = [g[target_col].values for _, g in eval_df.groupby(col) if len(g) >= 2]
                        if len(groups) >= 2:
                            try:
                                f_stat, f_pval = stats.f_oneway(*groups)
                                # Normalized score approximation
                                score = min(100.0, float(f_stat) * 5.0) if not np.isnan(f_stat) else 0.0
                                drivers.append({
                                    "feature": str(col),
                                    "feature_type": "Categorical",
                                    "metric_type": "One-Way ANOVA F-Statistic",
                                    "f_statistic": round(float(f_stat), 2),
                                    "p_value": round(float(f_pval), 5),
                                    "association_score": round(score, 1),
                                    "relationship_type": "Categorical Cohort Disparity",
                                    "strength": "Strong" if score >= 60 else ("Moderate" if score >= 30 else "Mild")
                                })
                            except Exception:
                                pass
            else:
                # Classification Target
                if is_col_num:
                    # Discrete Target vs Numerical Feature -> Welch t-test / ANOVA F
                    groups = [g[col].values for _, g in eval_df.groupby(target_col) if len(g) >= 2]
                    if len(groups) >= 2:
                        try:
                            f_stat, f_pval = stats.f_oneway(*groups)
                            score = min(100.0, float(f_stat) * 4.0) if not np.isnan(f_stat) else 0.0
                            drivers.append({
                                "feature": str(col),
                                "feature_type": "Numerical",
                                "metric_type": "Cohort Mean Separation F-Score",
                                "f_statistic": round(float(f_stat), 2),
                                "p_value": round(float(f_pval), 5),
                                "association_score": round(score, 1),
                                "relationship_type": "Class Divergence",
                                "strength": "Strong" if score >= 60 else ("Moderate" if score >= 30 else "Mild")
                            })
                        except Exception:
                            pass
                else:
                    # Discrete Target vs Categorical Feature -> Chi-Square & Cramer's V
                    if eval_df[col].nunique() <= 20:
                        crosstab = pd.crosstab(eval_df[target_col], eval_df[col])
                        try:
                            chi2, pval, dof, _ = stats.chi2_contingency(crosstab)
                            n = crosstab.sum().sum()
                            min_d = min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)
                            v = np.sqrt(chi2 / (n * min_d)) if min_d > 0 and n > 0 else 0.0
                            score = float(v) * 100
                            drivers.append({
                                "feature": str(col),
                                "feature_type": "Categorical",
                                "metric_type": "Cramer's V Contingency Metric",
                                "cramers_v": round(float(v), 3),
                                "p_value": round(float(pval), 5),
                                "association_score": round(score, 1),
                                "relationship_type": "Categorical Association",
                                "strength": "Strong" if score >= 40 else ("Moderate" if score >= 20 else "Mild")
                            })
                        except Exception:
                            pass

        drivers.sort(key=lambda x: x["association_score"], reverse=True)
        return drivers
