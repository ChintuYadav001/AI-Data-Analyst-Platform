"""
Auto-Granularity Chronological Waves, Growth Rates & Seasonality Engine.
Detects datetime columns, dynamically determines optimal temporal granularity,
and computes rolling averages, period-over-period growth, peaks, valleys, and seasonality.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class EDATimeSeriesEngine:
    """Enterprise time-series analysis engine with auto-granularity and seasonality detection."""

    @classmethod
    def analyze_temporal_trends(
        cls,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        metric_col: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Calculates multi-granularity trends, growth, rolling volatility, and seasonality."""
        if df is None or df.empty:
            return None

        # 1. Resolve date column
        target_date_col = date_col
        if not target_date_col:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    target_date_col = col
                    break
                if any(k in str(col).lower() for k in ["date", "time", "timestamp", "created_at"]):
                    try:
                        sample = df[col].dropna().iloc[:10]
                        if len(sample) > 0:
                            pd.to_datetime(sample)
                            target_date_col = col
                            break
                    except Exception:
                        pass

        if not target_date_col:
            return None

        # 2. Resolve target metric first to avoid copying whole dataset
        target_metric = metric_col
        if not target_metric or target_metric not in df.columns or not pd.api.types.is_numeric_dtype(df[target_metric]):
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != target_date_col and not any(k in str(col).lower() for k in ["_id", "id_", "identifier"]):
                    target_metric = col
                    break

        if not target_metric:
            return None

        # 3. Parse dates cleanly on minimal 2-column projection (prevents copying 20+ columns across lakhs of rows)
        sub_df = df[[target_date_col, target_metric]].dropna()
        sub_df["_parsed_dt"] = pd.to_datetime(sub_df[target_date_col], errors="coerce")
        temp_df = sub_df.dropna(subset=["_parsed_dt"]).sort_values("_parsed_dt")
        if len(temp_df) < 5:
            return None

        # 4. Determine auto-granularity based on timespan
        min_date = temp_df["_parsed_dt"].min()
        max_date = temp_df["_parsed_dt"].max()
        span_days = (max_date - min_date).days

        if span_days > 1460:  # > 4 years
            primary_freq = "YE"
            freq_label = "Yearly"
        elif span_days > 180:  # 6 months to 4 years
            primary_freq = "ME"
            freq_label = "Monthly"
        elif span_days > 30:  # 1 to 6 months
            primary_freq = "W"
            freq_label = "Weekly"
        else:
            primary_freq = "D"
            freq_label = "Daily"

        # Resample
        ts_indexed = temp_df.set_index("_parsed_dt")
        resampled = ts_indexed.resample(primary_freq)[target_metric].agg(["sum", "mean", "count"]).reset_index()
        resampled = resampled[resampled["count"] > 0]

        if resampled.empty:
            return None

        # Format points
        points: List[Dict[str, Any]] = []
        for _, row in resampled.iterrows():
            dt = row["_parsed_dt"]
            dt_str = dt.strftime("%Y-%m-%d") if primary_freq in ["D", "W"] else dt.strftime("%Y-%m")
            points.append({
                "period": dt_str,
                "total": round(float(row["sum"]), 2),
                "mean": round(float(row["mean"]), 2),
                "record_count": int(row["count"])
            })

        # Calculate period-over-period growth rate
        growth_rate_pct = 0.0
        if len(points) >= 2:
            last_val = points[-1]["total"]
            prev_val = points[-2]["total"]
            if prev_val != 0:
                growth_rate_pct = round(((last_val - prev_val) / abs(prev_val)) * 100, 2)

        # Rolling 3-period average smoothing
        totals_series = pd.Series([p["total"] for p in points])
        rolling_smooth = totals_series.rolling(window=min(3, len(totals_series)), min_periods=1).mean()
        for idx, p in enumerate(points):
            p["rolling_avg"] = round(float(rolling_smooth.iloc[idx]), 2)

        # Peaks and Valleys
        peak_pt = max(points, key=lambda x: x["total"])
        valley_pt = min(points, key=lambda x: x["total"])

        # Trend direction (OLS linear slope on index)
        if len(points) >= 3:
            x_vals = np.arange(len(points))
            y_vals = np.array([p["total"] for p in points])
            slope, _ = np.polyfit(x_vals, y_vals, 1)
            mean_y = float(np.mean(y_vals)) if np.mean(y_vals) != 0 else 1.0
            norm_slope = slope / abs(mean_y)
            if norm_slope > 0.05:
                trend_direction = "Strong Upward Momentum"
            elif norm_slope > 0.01:
                trend_direction = "Mild Upward Trend"
            elif norm_slope < -0.05:
                trend_direction = "Pronounced Contraction / Downward"
            elif norm_slope < -0.01:
                trend_direction = "Mild Downward Trend"
            else:
                trend_direction = "Stationary / Stable"
        else:
            trend_direction = "Insufficient periods for slope estimation"

        # Seasonality (Day of Week & Month of Year)
        day_of_week_agg: List[Dict[str, Any]] = []
        if span_days >= 14:
            temp_df["_dow"] = temp_df["_parsed_dt"].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_grouped = temp_df.groupby("_dow")[target_metric].agg(["mean", "sum"]).reindex(dow_order).dropna()
            for day, row in dow_grouped.iterrows():
                day_of_week_agg.append({
                    "day": str(day),
                    "mean": round(float(row["mean"]), 2),
                    "total": round(float(row["sum"]), 2)
                })

        return {
            "date_column": str(target_date_col),
            "metric_column": str(target_metric),
            "granularity": freq_label,
            "span_days": span_days,
            "min_date": min_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
            "trend_direction": trend_direction,
            "latest_growth_rate_pct": growth_rate_pct,
            "peak_period": peak_pt,
            "valley_period": valley_pt,
            "trend_points": points,
            "seasonality_day_of_week": day_of_week_agg
        }
