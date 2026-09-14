"""
Segmentation Breakdown & Contribution Matrix Engine.
Evaluates dimensions, computes cohort metrics, and ranks segments by metric contribution.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class EDASegmentEngine:
    """Ranks and contrasts dimensional cohorts by metric contribution and variance."""

    @classmethod
    def analyze_segments(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        cat_cols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Identifies top segmentation dimensions and ranks cohorts by metric contribution."""
        if df is None or df.empty:
            return []

        # Find target metric
        metric = target_col
        if not metric or metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if not any(k in str(col).lower() for k in ["_id", "id_", "identifier"]):
                    metric = col
                    break

        if not metric:
            return []

        # Find segmentation dimensions
        dimensions = cat_cols if cat_cols is not None else list(df.select_dtypes(include=["object", "string", "category"]).columns)
        valid_dims = [
            c for c in dimensions
            if 2 <= df[c].nunique() <= 25 and not any(k in str(c).lower() for k in ["_id", "id_", "identifier"])
        ]

        if not valid_dims:
            return []

        results: List[Dict[str, Any]] = []
        overall_mean = float(df[metric].mean())
        total_metric_sum = float(df[metric].sum()) if float(df[metric].sum()) != 0 else 1.0

        for dim in valid_dims[:4]:  # Top 4 dimensions
            grouped = df.groupby(dim)[metric].agg(["count", "sum", "mean", "median"]).reset_index()
            grouped = grouped.sort_values("sum", ascending=False)

            cohorts: List[Dict[str, Any]] = []
            for _, row in grouped.iterrows():
                c_sum = float(row["sum"])
                c_mean = float(row["mean"])
                c_count = int(row["count"])
                share_pct = round((c_sum / total_metric_sum) * 100, 2)
                vol_share = round((c_count / len(df)) * 100, 2)

                status = "Leader (Outperforming)" if c_mean >= 1.15 * overall_mean else (
                    "Laggard (Underperforming)" if c_mean <= 0.85 * overall_mean else "On-Benchmark"
                )

                cohorts.append({
                    "segment": str(row[dim]),
                    "record_count": c_count,
                    "volume_share_pct": vol_share,
                    "total_metric": round(c_sum, 2),
                    "mean_metric": round(c_mean, 2),
                    "median_metric": round(float(row["median"]), 2),
                    "contribution_pct": share_pct,
                    "status": status
                })

            # Concentration in top 2 cohorts
            top2_contrib = sum(c["contribution_pct"] for c in cohorts[:2])

            results.append({
                "dimension": str(dim),
                "metric": str(metric),
                "overall_mean": round(overall_mean, 2),
                "top2_concentration_pct": round(top2_contrib, 2),
                "top_performer": cohorts[0]["segment"] if cohorts else "N/A",
                "cohorts": cohorts
            })

        return results
