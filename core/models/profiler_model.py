import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re

from core.cache import AnalyticsCache

class DataProfilerModel:
    """Profiles datasets and individual columns, detecting statistical attributes,

    semantic data types, and suspicious columns with sub-millisecond caching.

    Optimized for large datasets (250K+ rows):
    - Sampling-based duplicate detection
    - Batch DuckDB-style vectorized column statistics
    - Capped mode/unique computation on samples for speed
    """

    # ── Sampling threshold: above this, we sample for expensive ops ──
    LARGE_DATASET_THRESHOLD = 50_000
    SAMPLE_SIZE_FOR_DUPS = 30_000
    SAMPLE_SIZE_FOR_MODE = 10_000

    @staticmethod
    def _safe_to_str(s: pd.Series) -> pd.Series:
        try:
            return s.astype(str)
        except (UnicodeDecodeError, Exception):
            def _decode(val):
                if isinstance(val, bytes):
                    try:
                        return val.decode("utf-8")
                    except Exception:
                        return val.decode("latin-1", errors="replace")
                return str(val)
            return s.map(_decode)

    @classmethod
    def _safe_nunique(cls, series: pd.Series) -> int:
        try:
            return int(series.nunique())
        except TypeError:
            try:
                return int(cls._safe_to_str(series).nunique())
            except Exception:
                return len(series.dropna())
        except Exception:
            return 0

    @classmethod
    def classify_column(cls, series: pd.Series, col_name: str, total_rows: int, unique_count: Optional[int] = None) -> str:
        name_lower = col_name.lower().strip()
        non_null_count = series.dropna().count()
        if non_null_count == 0:
            return "Empty"

        # Check for Identifier first
        if unique_count is None:
            unique_count = cls._safe_nunique(series)
        unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0
        if (unique_ratio > 0.95 and total_rows > 30) or any(k in name_lower for k in ["_id", "id_", "identifier", "uuid", "guid", "ssn"]):
            if unique_ratio > 0.7:
                return "Identifier"

        # Check for Potential Target Variable
        if any(name_lower == target for target in ["target", "label", "churn", "exited", "survived", "status", "converted", "default"]):
            return "Potential Target Variable"

        # Check for Boolean (only if unique_count <= 2)
        if unique_count <= 2:
            try:
                sample_u = series.dropna().head(100)
                if sample_u.isin([0, 1, True, False, "0", "1", "true", "false", "True", "False", "yes", "no", "Y", "N"]).all():
                    if series.dropna().isin([0, 1, True, False, "0", "1", "true", "false", "True", "False", "yes", "no", "Y", "N"]).all():
                        return "Boolean"
            except Exception:
                pass

        # Check for Date/Time
        if pd.api.types.is_datetime64_any_dtype(series):
            return "Date/time"
        if any(k in name_lower for k in ["date", "time", "timestamp", "year", "created_at", "updated_at", "dob"]):
            try:
                sample = cls._safe_to_str(series.dropna().head(30))
                pd.to_datetime(sample, errors="raise")
                return "Date/time"
            except Exception:
                pass

        # Check for Numerical
        if pd.api.types.is_numeric_dtype(series):
            return "Numerical"
        else:
            # Try converting sample to numeric
            try:
                sample = cls._safe_to_str(series.dropna().head(30)).str.replace(",", "").str.replace("$", "").str.replace("%", "")
                pd.to_numeric(sample, errors="raise")
                return "Numerical"
            except Exception:
                pass

        # Text vs Categorical
        if unique_count <= 50 or (unique_ratio < 0.2 and unique_count < 200):
            return "Categorical"

        sample_for_len = cls._safe_to_str(series.dropna().head(1000))
        avg_str_len = float(sample_for_len.str.len().mean()) if not sample_for_len.empty else 0.0
        if avg_str_len > 30:
            return "Text"

        return "Categorical"

    @classmethod
    def _fast_unique_counts(cls, df: pd.DataFrame) -> Dict[str, int]:
        """Compute unique counts efficiently — use approx for large datasets and resilient to unhashables."""
        n_rows = len(df)
        result = {}
        if n_rows > cls.LARGE_DATASET_THRESHOLD:
            # Sample-based approximate unique counts for non-numeric columns
            sample = df.sample(min(cls.SAMPLE_SIZE_FOR_MODE, n_rows), random_state=42)
            ratio = n_rows / len(sample)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    result[col] = cls._safe_nunique(df[col])
                else:
                    sample_uniq = cls._safe_nunique(sample[col])
                    if sample_uniq < 100:
                        result[col] = cls._safe_nunique(df[col])
                    else:
                        result[col] = min(int(sample_uniq * min(ratio, 3)), int(df[col].notna().sum()))
        else:
            for col in df.columns:
                result[col] = cls._safe_nunique(df[col])
        return result

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cached = AnalyticsCache.get(df, "profile")
        if cached is not None:
            return cached

        n_rows, n_cols = df.shape
        is_large = n_rows > cls.LARGE_DATASET_THRESHOLD

        # ── Memory estimation (always sampled for large) ──
        try:
            if n_rows > 5000:
                sample_n = 1000
                sample_mem = int(df.sample(sample_n, random_state=42).memory_usage(deep=True).sum())
                mem_bytes = int((sample_mem / sample_n) * n_rows)
            else:
                mem_bytes = int(df.memory_usage(deep=True).sum())
        except Exception:
            try:
                mem_bytes = int(df.memory_usage().sum())
            except Exception:
                mem_bytes = 0
        mem_mb = round(mem_bytes / (1024 * 1024), 2)

        # ── Duplicate detection (sampled for large datasets, resilient to unhashables) ──
        cached_dups = AnalyticsCache.get(df, "duplicate_rows")
        if cached_dups is not None:
            dup_rows = cached_dups
        else:
            try:
                if is_large:
                    dup_sample_size = min(cls.SAMPLE_SIZE_FOR_DUPS, n_rows)
                    sample_df = df.sample(dup_sample_size, random_state=42)
                    sample_dups = int(sample_df.duplicated().sum())
                    dup_rows = int(sample_dups * (n_rows / dup_sample_size))
                else:
                    dup_rows = int(df.duplicated().sum())
            except TypeError:
                # Handle unhashable types (dict, list, etc.)
                try:
                    hashable_cols = [c for c in df.columns if df[c].dtype != "object" or not df[c].dropna().map(lambda x: isinstance(x, (list, dict, set, bytearray))).any()]
                    if hashable_cols:
                        dup_rows = int(df.duplicated(subset=hashable_cols).sum())
                    else:
                        dup_rows = 0
                except Exception:
                    dup_rows = 0
            except Exception:
                dup_rows = 0
            AnalyticsCache.set(df, "duplicate_rows", dup_rows)
        dup_pct = round((dup_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

        # ── Missing values (vectorized — safe against duplicate columns & complex dtypes) ──
        try:
            null_counts_series = df.isna().sum()
            total_cells = n_rows * n_cols
            total_missing = int(np.nan_to_num(null_counts_series.sum()))
        except Exception:
            null_counts_series = pd.Series({c: int(df[c].isna().sum()) for c in df.columns})
            total_cells = n_rows * n_cols
            total_missing = int(null_counts_series.sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # ── Batch unique counts ──
        unique_counts = cls._fast_unique_counts(df)

        # ── Batch numeric statistics (vectorized with Skew & Outliers) ──
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75])
            for col in numeric_cols:
                if col in desc.columns:
                    col_series = df[col].dropna()
                    q25 = float(desc.loc["25%", col]) if pd.notna(desc.loc["25%", col]) else 0.0
                    q75 = float(desc.loc["75%", col]) if pd.notna(desc.loc["75%", col]) else 0.0
                    iqr = round(q75 - q25, 4)
                    
                    # Outliers (IQR method) & Skewness
                    outliers_count = 0
                    skew_val = 0.0
                    if not col_series.empty:
                        try:
                            lower_bound = q25 - 1.5 * iqr
                            upper_bound = q75 + 1.5 * iqr
                            outliers_count = int(((col_series < lower_bound) | (col_series > upper_bound)).sum())
                            skew_val = round(float(col_series.skew()), 4) if len(col_series) > 2 else 0.0
                        except Exception:
                            pass

                    numeric_stats[col] = {
                        "min": float(desc.loc["min", col]) if pd.notna(desc.loc["min", col]) else 0.0,
                        "max": float(desc.loc["max", col]) if pd.notna(desc.loc["max", col]) else 0.0,
                        "mean": round(float(desc.loc["mean", col]), 4) if pd.notna(desc.loc["mean", col]) else 0.0,
                        "median": round(float(desc.loc["50%", col]), 4) if pd.notna(desc.loc["50%", col]) else 0.0,
                        "std": round(float(desc.loc["std", col]), 4) if pd.notna(desc.loc["std", col]) else 0.0,
                        "q25": round(q25, 4),
                        "q50": round(float(desc.loc["50%", col]), 4) if pd.notna(desc.loc["50%", col]) else 0.0,
                        "q75": round(q75, 4),
                        "iqr": iqr,
                        "skewness": skew_val,
                        "outliers": outliers_count,
                        "outlier_pct": round((outliers_count / len(col_series) * 100), 2) if len(col_series) > 0 else 0.0
                    }

        # ── Per-column profiling ──
        columns_profile = []
        suspicious_columns = []

        # Pre-compute mode for all columns using sample
        if is_large:
            mode_sample = df.sample(min(cls.SAMPLE_SIZE_FOR_MODE, n_rows), random_state=42)
        else:
            mode_sample = df

        for col in df.columns:
            series = df[col]
            unique_count = unique_counts.get(col, 0)
            
            raw_null = null_counts_series.get(col, 0)
            null_count = int(raw_null.iloc[0] if isinstance(raw_null, pd.Series) else raw_null)
            col_null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

            # Classification uses sample for large datasets
            if is_large:
                classify_series = series.head(min(50_000, n_rows))
            else:
                classify_series = series
            semantic_type = cls.classify_column(classify_series, str(col), n_rows, unique_count=unique_count)

            # Mode from sample
            mode_val = None
            try:
                mode_series = mode_sample[col].mode()
                if not mode_series.empty:
                    mode_val = str(mode_series.iloc[0])
            except Exception:
                pass

            col_info: Dict[str, Any] = {
                "name": str(col),
                "dtype": str(series.dtype),
                "semantic_type": semantic_type,
                "null_count": null_count,
                "null_pct": col_null_pct,
                "unique_count": unique_count,
                "unique_pct": round((unique_count / n_rows) * 100, 2) if n_rows > 0 else 0.0,
                "mode": mode_val
            }

            # Numerical stats from pre-computed batch describe
            if col in numeric_stats:
                col_info.update(numeric_stats[col])

            # Suspicious column rules
            if unique_count == 1:
                suspicious_columns.append({
                    "column": str(col),
                    "reason": "Constant column: Contains only 1 unique value across all rows."
                })
            elif col_null_pct > 60.0:
                suspicious_columns.append({
                    "column": str(col),
                    "reason": f"High missing rate: {col_null_pct}% values are missing."
                })
            elif semantic_type == "Identifier":
                suspicious_columns.append({
                    "column": str(col),
                    "reason": "Probable identifier column: High cardinality unique key with little aggregate value."
                })

            columns_profile.append(col_info)

        # Composite Data Quality Health Score (0-100%)
        completeness_score = max(0.0, 100.0 - missing_pct)
        uniqueness_score = max(0.0, 100.0 - dup_pct)
        suspicious_penalty = min(30.0, len(suspicious_columns) * 5.0)
        health_score = round(max(5.0, min(100.0, (completeness_score * 0.5) + (uniqueness_score * 0.3) + 20.0 - suspicious_penalty)), 1)

        res = {
            "dataset_info": {
                "rows": n_rows,
                "columns": n_cols,
                "memory_mb": mem_mb,
                "duplicate_rows": dup_rows,
                "duplicate_pct": dup_pct,
                "total_missing": total_missing,
                "missing_pct": missing_pct,
                "health_score": health_score,
            },
            "columns": columns_profile,
            "suspicious_columns": suspicious_columns
        }
        AnalyticsCache.set(df, "profile", res)
        return res
