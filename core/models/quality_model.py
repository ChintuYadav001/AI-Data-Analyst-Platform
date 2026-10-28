import pandas as pd
import numpy as np
from typing import Dict, Any, List
from core.cache import AnalyticsCache

class DataQualityModel:
    """Calculates dataset quality score from 0-100 with clear deduction breakdowns

    based on missing values, duplicates, outliers, invalid types, constant columns,
    and inconsistent categories with sub-millisecond caching.
    """

    @classmethod
    def calculate_quality_score(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cached = AnalyticsCache.get(df, "quality")
        if cached is not None:
            return cached

        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return {
                "score": 0,
                "grade": "F",
                "breakdown": [{"category": "Empty Dataset", "deduction": 100, "explanation": "Dataset is empty"}]
            }

        score = 100.0
        breakdown: List[Dict[str, Any]] = []

        # 1. Missing values check
        total_cells = n_rows * n_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = (total_missing / total_cells) * 100
        if missing_pct > 0:
            # Deduction: up to 25 points for missing values
            deduction = min(25.0, round(missing_pct * 0.8, 1))
            score -= deduction
            breakdown.append({
                "category": "Missing Values",
                "deduction": deduction,
                "severity": "High" if missing_pct > 15 else "Medium",
                "explanation": f"{total_missing} missing values ({missing_pct:.1f}% of total data points)."
            })

        # 2. Duplicate records check (reuse cached duplicates if available)
        cached_dups = AnalyticsCache.get(df, "duplicate_rows")
        if cached_dups is not None:
            dup_rows = cached_dups
        else:
            dup_rows = int(df.duplicated().sum())
            AnalyticsCache.set(df, "duplicate_rows", dup_rows)

        dup_pct = (dup_rows / n_rows) * 100
        if dup_rows > 0:
            deduction = min(15.0, round(dup_pct * 1.2, 1))
            score -= deduction
            breakdown.append({
                "category": "Duplicate Records",
                "deduction": deduction,
                "severity": "High" if dup_pct > 10 else "Medium",
                "explanation": f"{dup_rows} duplicate rows detected ({dup_pct:.1f}% of dataset)."
            })

        # 3. Constant columns (columns with only 1 unique value)
        constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            deduction = min(15.0, len(constant_cols) * 4.0)
            score -= deduction
            breakdown.append({
                "category": "Constant Columns",
                "deduction": deduction,
                "severity": "Medium",
                "explanation": f"{len(constant_cols)} column(s) have constant zero-variance values: {', '.join(constant_cols)}."
            })

        # 4. Outliers in numerical columns (using single-batch IQR)
        outlier_count = 0
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            series = df[col].dropna()
            if len(series) > 10:
                q25, q75 = series.quantile([0.25, 0.75])
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - 2.5 * iqr
                    upper = q75 + 2.5 * iqr
                    outlier_count += int(((series < lower) | (series > upper)).sum())

        outlier_pct = (outlier_count / (n_rows * len(num_cols))) * 100 if len(num_cols) > 0 else 0
        if outlier_count > 0:
            deduction = min(15.0, round(outlier_pct * 2.0, 1))
            score -= deduction
            breakdown.append({
                "category": "Extreme Outliers",
                "deduction": deduction,
                "severity": "Medium" if outlier_pct > 5 else "Low",
                "explanation": f"{outlier_count} extreme outlier values found across numeric columns."
            })

        # 5. Inconsistent category casings / spaces (sample-accelerated on large datasets)
        cat_issues = 0
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        for col in cat_cols:
            raw_s = df[col].dropna()
            if n_rows > 10000:
                raw_s = raw_s.head(5000)
            try:
                sample_series = raw_s.astype(str)
            except Exception:
                sample_series = raw_s.map(lambda x: x.decode("latin-1", errors="replace") if isinstance(x, bytes) else str(x))
            raw_unique = sample_series.nunique()
            if raw_unique <= 1 or raw_unique > 200:
                continue
            cleaned_unique = sample_series.str.strip().str.lower().nunique()
            if raw_unique > cleaned_unique:
                diff = raw_unique - cleaned_unique
                cat_issues += diff

        if cat_issues > 0:
            deduction = min(10.0, float(cat_issues * 2.0))
            score -= deduction
            breakdown.append({
                "category": "Inconsistent Categories",
                "deduction": deduction,
                "severity": "Low",
                "explanation": f"{cat_issues} categorical inconsistencies found due to mixed casing or trailing spaces."
            })

        # 6. Type mismatch / numeric columns stored as strings (Sampled to max 100 for instant evaluation)
        type_issues = []
        for col in cat_cols:
            raw_s = df[col].dropna().head(100)
            try:
                sample_series = raw_s.astype(str)
            except Exception:
                sample_series = raw_s.map(lambda x: x.decode("latin-1", errors="replace") if isinstance(x, bytes) else str(x))
            if not sample_series.empty:
                numeric_matches = sample_series.str.replace(r"[\$,%]", "", regex=True).str.strip().str.match(r"^-?\d+(\.\d+)?$").sum()
                if (numeric_matches / len(sample_series)) > 0.85:
                    type_issues.append(str(col))

        if type_issues:
            deduction = min(10.0, len(type_issues) * 3.0)
            score -= deduction
            breakdown.append({
                "category": "Suboptimal Data Types",
                "deduction": deduction,
                "severity": "Medium",
                "explanation": f"Columns {', '.join(type_issues)} appear to be numeric but are stored as text/object."
            })

        final_score = max(0, min(100, round(score)))

        if final_score >= 90:
            grade = "Excellent"
        elif final_score >= 75:
            grade = "Good"
        elif final_score >= 50:
            grade = "Fair"
        else:
            grade = "Needs Attention"

        res = {
            "score": final_score,
            "grade": grade,
            "breakdown": breakdown,
            "total_issues": len(breakdown)
        }
        AnalyticsCache.set(df, "quality", res)
        return res
