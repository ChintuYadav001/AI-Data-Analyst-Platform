import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re

from core.cache import AnalyticsCache

class DataProfilerModel:
    """Profiles datasets and individual columns, detecting statistical attributes,

    semantic data types, and suspicious columns with sub-millisecond caching.
    """

    @staticmethod
    def classify_column(series: pd.Series, col_name: str, total_rows: int, unique_count: Optional[int] = None) -> str:
        name_lower = col_name.lower().strip()
        non_null_count = series.dropna().count()
        if non_null_count == 0:
            return "Empty"

        # Check for Identifier first
        if unique_count is None:
            unique_count = int(series.nunique())
        unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0
        if (unique_ratio > 0.95 and total_rows > 30) or any(k in name_lower for k in ["_id", "id_", "identifier", "uuid", "guid", "ssn"]):
            if unique_ratio > 0.7:
                return "Identifier"

        # Check for Potential Target Variable
        if any(name_lower == target for target in ["target", "label", "churn", "exited", "survived", "status", "converted", "default"]):
            return "Potential Target Variable"

        # Check for Boolean (only if unique_count <= 2)
        if unique_count <= 2:
            sample_u = series.dropna().head(100)
            if sample_u.isin([0, 1, True, False, "0", "1", "true", "false", "True", "False", "yes", "no", "Y", "N"]).all():
                if series.dropna().isin([0, 1, True, False, "0", "1", "true", "false", "True", "False", "yes", "no", "Y", "N"]).all():
                    return "Boolean"

        # Check for Date/Time
        if pd.api.types.is_datetime64_any_dtype(series):
            return "Date/time"
        if any(k in name_lower for k in ["date", "time", "timestamp", "year", "created_at", "updated_at", "dob"]):
            try:
                sample = series.dropna().head(30).astype(str)
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
                sample = series.dropna().head(30).astype(str).str.replace(",", "").str.replace("$", "").str.replace("%", "")
                pd.to_numeric(sample, errors="raise")
                return "Numerical"
            except Exception:
                pass

        # Text vs Categorical
        if unique_count <= 50 or (unique_ratio < 0.2 and unique_count < 200):
            return "Categorical"

        sample_for_len = series.dropna().head(1000).astype(str)
        avg_str_len = float(sample_for_len.str.len().mean()) if not sample_for_len.empty else 0.0
        if avg_str_len > 30:
            return "Text"

        return "Categorical"

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cached = AnalyticsCache.get(df, "profile")
        if cached is not None:
            return cached

        n_rows, n_cols = df.shape
        if n_rows > 5000:
            sample_n = 1000
            sample_mem = int(df.sample(sample_n, random_state=42).memory_usage(deep=True).sum())
            mem_bytes = int((sample_mem / sample_n) * n_rows)
        else:
            mem_bytes = int(df.memory_usage(deep=True).sum())
        mem_mb = round(mem_bytes / (1024 * 1024), 2)
        
        cached_dups = AnalyticsCache.get(df, "duplicate_rows")
        if cached_dups is not None:
            dup_rows = cached_dups
        else:
            dup_rows = int(df.duplicated().sum())
            AnalyticsCache.set(df, "duplicate_rows", dup_rows)
        dup_pct = round((dup_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0
        
        total_cells = n_rows * n_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
        
        columns_profile = []
        suspicious_columns = []
        
        for col in df.columns:
            series = df[col]
            unique_count = int(series.nunique())
            semantic_type = cls.classify_column(series, str(col), n_rows, unique_count=unique_count)
            null_count = int(series.isna().sum())
            col_null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
            
            # Most frequent value (optimized for large datasets)
            mode_val = None
            try:
                if n_rows > 10000:
                    mode_series = series.head(5000).mode()
                else:
                    mode_series = series.mode()
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
                "mode": mode_val
            }
            
            # Numerical calculations with single-pass batch quantiles
            if pd.api.types.is_numeric_dtype(series):
                cleaned_series = series.dropna()
                if not cleaned_series.empty:
                    q_v = cleaned_series.quantile([0.25, 0.50, 0.75])
                    q25 = float(q_v.loc[0.25])
                    q50 = float(q_v.loc[0.50])
                    q75 = float(q_v.loc[0.75])
                    iqr = q75 - q25
                    col_info.update({
                        "min": float(cleaned_series.min()),
                        "max": float(cleaned_series.max()),
                        "mean": round(float(cleaned_series.mean()), 4),
                        "median": round(float(cleaned_series.median()), 4),
                        "std": round(float(cleaned_series.std()), 4) if len(cleaned_series) > 1 else 0.0,
                        "q25": round(q25, 4),
                        "q50": round(q50, 4),
                        "q75": round(q75, 4),
                        "iqr": round(iqr, 4)
                    })
            
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
            
        res = {
            "dataset_info": {
                "rows": n_rows,
                "columns": n_cols,
                "memory_mb": mem_mb,
                "duplicate_rows": dup_rows,
                "duplicate_pct": dup_pct,
                "total_missing": total_missing,
                "missing_pct": missing_pct,
            },
            "columns": columns_profile,
            "suspicious_columns": suspicious_columns
        }
        AnalyticsCache.set(df, "profile", res)
        return res
