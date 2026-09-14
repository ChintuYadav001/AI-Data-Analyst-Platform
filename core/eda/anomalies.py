"""
Multi-Method Outlier Exploration & Impact Analysis Engine.
Executes Tukey IQR, Z-score, Modified Z-score (MAD), and percentile thresholds.
Never silently deletes; classifies records into error, extreme observation, or investigation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class EDAAnomalyEngine:
    """Identifies, classifies, and evaluates statistical impact of extreme observations."""

    @classmethod
    def detect_anomalies(cls, df: pd.DataFrame, num_cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scans numerical attributes using multiple anomaly detection models."""
        if df is None or df.empty:
            return []

        cols = num_cols if num_cols is not None else list(df.select_dtypes(include=[np.number]).columns)
        cols = [c for c in cols if not any(k in str(c).lower() for k in ["_id", "id_", "identifier"])]
        total_rows = len(df)
        if total_rows < 10:
            return []

        results: List[Dict[str, Any]] = []

        for col in cols:
            series = df[col].dropna()
            n_vals = len(series)
            if n_vals < 10:
                continue

            # 1. Tukey IQR & Percentile Batch Calculation
            q_vals = series.quantile([0.005, 0.25, 0.75, 0.995])
            p005 = float(q_vals.loc[0.005])
            q25 = float(q_vals.loc[0.25])
            q75 = float(q_vals.loc[0.75])
            p995 = float(q_vals.loc[0.995])

            iqr = q75 - q25
            if iqr <= 0:
                continue

            mild_lower = q25 - 1.5 * iqr
            mild_upper = q75 + 1.5 * iqr
            extreme_lower = q25 - 3.0 * iqr
            extreme_upper = q75 + 3.0 * iqr

            iqr_mild_mask = (series < mild_lower) | (series > mild_upper)
            iqr_extreme_mask = (series < extreme_lower) | (series > extreme_upper)

            n_mild = int(iqr_mild_mask.sum())
            n_extreme = int(iqr_extreme_mask.sum())

            # 2. Z-Score (|z| > 3)
            mean_val = float(series.mean())
            std_val = float(series.std()) if n_vals > 1 else 1.0
            if std_val > 0:
                z_scores = (series - mean_val) / std_val
                z_outliers = int((z_scores.abs() > 3.0).sum())
            else:
                z_outliers = 0

            # 3. Modified Z-score (MAD)
            median_val = float(series.median())
            if n_vals > 25000:
                mad_sample = series.sample(25000, random_state=42)
                mad = float(np.median(np.abs(mad_sample - median_val)))
            else:
                mad = float(np.median(np.abs(series - median_val)))

            if mad > 0:
                mod_z = 0.6745 * np.abs(series - median_val) / mad
                mad_outliers = int((mod_z > 3.5).sum())
            else:
                mad_outliers = 0

            # 4. Percentile thresholds (99.5th and 0.5th)
            p_outliers = int(((series < p005) | (series > p995)).sum())

            # 5. Impact on statistics
            non_outliers = series[~iqr_mild_mask]
            clean_mean = float(non_outliers.mean()) if not non_outliers.empty else mean_val
            clean_std = float(non_outliers.std()) if len(non_outliers) > 1 else std_val

            mean_shift_pct = round(abs((mean_val - clean_mean) / clean_mean) * 100, 2) if clean_mean != 0 else 0.0

            # 6. Classification
            has_impossible = False
            col_name_lower = str(col).lower()
            if any(k in col_name_lower for k in ["age", "revenue", "price", "sales", "quantity", "cost"]) and float(series.min()) < 0:
                has_impossible = True
            elif "rating" in col_name_lower and float(series.max()) > 5.0 and float(series.max()) < 10.0:
                has_impossible = True

            if has_impossible:
                classification = "Potential Data Error (Impossible Values Detected)"
            elif mean_shift_pct >= 10.0 or n_extreme > 0:
                classification = "Legitimate Extreme Observations (Heavy Right Tail)"
            else:
                classification = "Routine Statistical Boundary Dispersion"

            results.append({
                "column": str(col),
                "total_records": n_vals,
                "iqr_outliers_count": n_mild,
                "iqr_outliers_pct": round((n_mild / n_vals) * 100, 2),
                "extreme_outliers_count": n_extreme,
                "z_score_outliers_count": z_outliers,
                "mad_outliers_count": mad_outliers,
                "percentile_outliers_count": p_outliers,
                "lower_bound_mild": round(mild_lower, 2),
                "upper_bound_mild": round(mild_upper, 2),
                "lower_bound_extreme": round(extreme_lower, 2),
                "upper_bound_extreme": round(extreme_upper, 2),
                "original_mean": round(mean_val, 2),
                "mean_without_outliers": round(clean_mean, 2),
                "mean_shift_pct": mean_shift_pct,
                "original_std": round(std_val, 2),
                "std_without_outliers": round(clean_std, 2),
                "classification": classification,
                "recommendation": "Quarantine for segment deep-dive; do not delete from analytical baseline."
            })

        # Rank by statistical impact (mean_shift_pct descending)
        results.sort(key=lambda x: x["mean_shift_pct"], reverse=True)
        return results
