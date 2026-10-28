"""
core.ml.ml_readiness
====================
Dataset suitability audit, automatic target detection, task classification,
and target leakage detection for Quantum Analyst ML engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class TargetCandidate:
    column: str
    task_type: str
    confidence: float
    reason: str
    unique_count: int
    data_type: str


@dataclass
class ReadinessReport:
    is_ready: bool
    overall_score: int  # 0 to 100
    total_rows: int
    total_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    id_columns: List[str]
    constant_columns: List[str]
    high_missing_columns: List[Dict[str, Any]]
    high_cardinality_columns: List[Dict[str, Any]]
    target_candidates: List[TargetCandidate]
    selected_target: Optional[str]
    detected_task: str
    leakage_warnings: List[Dict[str, Any]]
    warnings: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "overall_score": self.overall_score,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "id_columns": self.id_columns,
            "constant_columns": self.constant_columns,
            "high_missing_columns": self.high_missing_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "target_candidates": [
                {
                    "column": c.column,
                    "task_type": c.task_type,
                    "confidence": c.confidence,
                    "reason": c.reason,
                    "unique_count": c.unique_count,
                    "data_type": c.data_type,
                }
                for c in self.target_candidates
            ],
            "selected_target": self.selected_target,
            "detected_task": self.detected_task,
            "leakage_warnings": self.leakage_warnings,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


# Target semantic keywords with associated tasks & base confidence
TARGET_KEYWORDS = {
    # Classification keywords
    "churn": ("binary_classification", 0.95),
    "target": ("auto", 0.92),
    "label": ("auto", 0.90),
    "status": ("classification", 0.85),
    "class": ("classification", 0.88),
    "category": ("classification", 0.75),
    "fraud": ("binary_classification", 0.95),
    "default": ("binary_classification", 0.92),
    "converted": ("binary_classification", 0.92),
    "buy": ("binary_classification", 0.85),
    "purchased": ("binary_classification", 0.88),
    "survived": ("binary_classification", 0.95),
    "outcome": ("classification", 0.85),
    "result": ("classification", 0.80),
    "sentiment": ("classification", 0.85),
    "approved": ("binary_classification", 0.90),
    "attrition": ("binary_classification", 0.92),
    "subscription": ("binary_classification", 0.80),

    # Regression keywords
    "price": ("regression", 0.92),
    "sales": ("regression", 0.90),
    "revenue": ("regression", 0.92),
    "salary": ("regression", 0.90),
    "cost": ("regression", 0.88),
    "profit": ("regression", 0.90),
    "amount": ("regression", 0.85),
    "score": ("regression", 0.80),
    "rating": ("regression", 0.82),
    "val": ("regression", 0.70),
    "total": ("regression", 0.75),
    "quantity": ("regression", 0.78),
    "loss": ("regression", 0.85),
}


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Detect potential identifier or index columns."""
    id_cols = []
    n = len(df)
    work_df = df.sample(50_000, random_state=42) if n > 50_000 else df
    w_n = len(work_df)
    for col in df.columns:
        col_lower = str(col).lower()
        # Name heuristic
        is_id_named = any(k in col_lower for k in ["_id", "id_", "uuid", "guid", "key", "index", "code"]) or col_lower in ["id", "uid"]
        
        # Uniqueness check on sample
        nunique = work_df[col].nunique(dropna=True)
        if w_n > 50 and nunique >= 0.98 * w_n and (is_id_named or work_df[col].dtype == object):
            id_cols.append(col)
        elif is_id_named and (work_df[col].dtype == object or nunique >= 0.5 * w_n):
            id_cols.append(col)
    return id_cols


def detect_constant_columns(df: pd.DataFrame) -> List[str]:
    """Detect zero-variance or constant columns."""
    constant_cols = []
    work_df = df.sample(10_000, random_state=42) if len(df) > 10_000 else df
    for col in df.columns:
        if work_df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)
    return constant_cols


def classify_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Formally classify the machine learning task for a target column:
    'binary_classification', 'multiclass_classification', 'regression', or 'unsupported'.
    """
    if target_col not in df.columns:
        return "unsupported"

    series = df[target_col].dropna()
    if len(series) == 0:
        return "unsupported"

    # Sample for large datasets
    sample_series = series.sample(50_000, random_state=42) if len(series) > 50_000 else series
    nunique = sample_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(sample_series)

    # Check for boolean or 2 distinct classes
    if nunique == 2:
        return "binary_classification"

    # String / categorical targets with 3 to 50 classes
    if not is_numeric:
        if 2 < nunique <= 50:
            return "multiclass_classification"
        else:
            return "unsupported"

    # Numeric target
    if is_numeric:
        # Check if integer codes represent a discrete multiclass target
        is_integer_like = (series.astype(float) == series.astype(int)).all() if len(series) > 0 else False
        if is_integer_like and 2 < nunique <= 15:
            return "multiclass_classification"
        # Continuous numeric target
        if nunique > 15 or not is_integer_like:
            return "regression"
        if 2 < nunique <= 15:
            return "multiclass_classification"

    return "regression"


def detect_target_candidates(df: pd.DataFrame) -> List[TargetCandidate]:
    """Auto-detect potential target candidates with confidence scores and rationales."""
    candidates = []
    n = len(df)
    if n == 0:
        return candidates

    work_df = df.sample(50_000, random_state=42) if n > 50_000 else df
    id_cols = set(detect_id_columns(work_df))
    constant_cols = set(detect_constant_columns(work_df))

    for col in df.columns:
        if col in id_cols or col in constant_cols:
            continue

        col_lower = str(col).lower()
        series = work_df[col].dropna()
        if len(series) == 0:
            continue

        nunique = series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        task_type = classify_task_type(work_df, col)
        
        confidence = 0.20
        reason_parts = []

        # Check keyword matches
        for kw, (k_task, k_conf) in TARGET_KEYWORDS.items():
            if kw in col_lower:
                confidence = max(confidence, k_conf)
                reason_parts.append(f"Contains target keyword '{kw}'")
                if k_task != "auto" and k_task != "classification":
                    task_type = k_task
                break

        # Check binary outcome heuristic
        if nunique == 2:
            confidence = max(confidence, 0.70)
            reason_parts.append("Binary distinct values (e.g. 0/1, Yes/No)")
            task_type = "binary_classification"

        # Position heuristic: last column is very frequently target in tabular datasets
        if col == df.columns[-1]:
            confidence = min(1.0, confidence + 0.15)
            reason_parts.append("Positioned as final column")

        # Missingness penalty
        missing_pct = df[col].isna().mean()
        if missing_pct > 0.4:
            confidence = max(0.1, confidence - 0.3)
            reason_parts.append(f"High missingness ({missing_pct:.1%})")

        reason = "; ".join(reason_parts) if reason_parts else f"Candidate variable with {nunique} unique values"
        
        candidates.append(
            TargetCandidate(
                column=col,
                task_type=task_type,
                confidence=round(confidence, 2),
                reason=reason,
                unique_count=nunique,
                data_type=str(df[col].dtype),
            )
        )

    # Sort candidates by confidence descending
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def detect_target_leakage(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """
    Detect potential target leakage:
    - Extremely high correlation (|r| > 0.98)
    - Exact duplicate values
    - Column name similarity
    """
    leakage_warnings = []
    if target_col not in df.columns or len(df) < 5:
        return leakage_warnings

    work_df = df.sample(30_000, random_state=42) if len(df) > 30_000 else df
    target_series = work_df[target_col]
    target_is_num = pd.api.types.is_numeric_dtype(target_series)

    for col in work_df.columns:
        if col == target_col:
            continue

        series = work_df[col]
        # Check duplicate
        if series.equals(target_series):
            leakage_warnings.append({
                "column": col,
                "target": target_col,
                "correlation": 1.0,
                "risk_level": "CRITICAL",
                "message": f"Column '{col}' is an exact duplicate of target '{target_col}' (Direct Leakage)."
            })
            continue

        # Check correlation if both numeric
        if target_is_num and pd.api.types.is_numeric_dtype(series):
            try:
                valid_mask = target_series.notna() & series.notna()
                if valid_mask.sum() >= 5:
                    corr = float(series[valid_mask].corr(target_series[valid_mask]))
                    if not np.isnan(corr) and abs(corr) >= 0.98:
                        leakage_warnings.append({
                            "column": col,
                            "target": target_col,
                            "correlation": round(corr, 4),
                            "risk_level": "HIGH",
                            "message": f"Correlation with target is {corr:.3f} (|r| >= 0.98). Risk of look-ahead target leakage."
                        })
                    elif not np.isnan(corr) and abs(corr) >= 0.92:
                        leakage_warnings.append({
                            "column": col,
                            "target": target_col,
                            "correlation": round(corr, 4),
                            "risk_level": "MODERATE",
                            "message": f"High correlation with target ({corr:.3f}). Verify this feature is available before prediction time."
                        })
            except Exception:
                pass

    return leakage_warnings


def audit_ml_readiness(df: pd.DataFrame, target_col: Optional[str] = None) -> ReadinessReport:
    """
    Comprehensive ML Readiness and Data Health Audit.
    Evaluates dataset scale, column types, missingness, cardinality, leakage, and task viability.
    """
    total_rows, total_columns = df.shape
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    # Sample for heavy cardinality and leakage scans on large datasets
    sample_df = df.sample(50_000, random_state=42) if total_rows > 50_000 else df
    id_cols = detect_id_columns(sample_df)
    constant_cols = detect_constant_columns(sample_df)

    high_missing = []
    for col in df.columns:
        pct = float(sample_df[col].isna().mean())
        if pct > 0.30:
            high_missing.append({"column": col, "missing_pct": round(pct * 100, 1)})

    high_cardinality = []
    for col in categorical_cols:
        unq = sample_df[col].nunique(dropna=True)
        if unq > 50 and col not in id_cols:
            high_cardinality.append({"column": col, "unique_count": unq})

    target_candidates = detect_target_candidates(sample_df)

    # Determine selected target
    selected_target = target_col
    if not selected_target and target_candidates:
        # Default to highest confidence candidate if >= 0.60
        if target_candidates[0].confidence >= 0.60:
            selected_target = target_candidates[0].column

    # Detect task
    detected_task = "clustering"
    if selected_target and selected_target in df.columns:
        detected_task = classify_task_type(df, selected_target)

    # Leakage
    leakage_warnings = detect_target_leakage(df, selected_target) if selected_target else []

    # Calculate readiness score out of 100
    score = 100
    warnings = []
    recommendations = []

    # Row count penalties
    if total_rows < 20:
        score -= 40
        warnings.append(f"Very small dataset ({total_rows} rows). Models may easily overfit.")
        recommendations.append("Acquire more data or use simple linear baselines.")
    elif total_rows < 50:
        score -= 20
        warnings.append(f"Small dataset ({total_rows} rows). Cross-validation recommended.")

    # High missing penalties
    if high_missing:
        score -= min(20, len(high_missing) * 5)
        warnings.append(f"{len(high_missing)} column(s) have >30% missing values.")
        recommendations.append("Impute or drop columns with excessive missingness.")

    # Constant columns penalty
    if constant_cols:
        score -= min(15, len(constant_cols) * 5)
        warnings.append(f"{len(constant_cols)} constant column(s) detected with zero variance: {constant_cols[:3]}.")
        recommendations.append("Exclude constant columns from model training.")

    # ID columns penalty
    if id_cols:
        score -= min(10, len(id_cols) * 3)
        warnings.append(f"{len(id_cols)} identifier column(s) detected: {id_cols[:3]}.")
        recommendations.append("Exclude ID columns from features to prevent memorization.")

    # High leakage penalty
    critical_leaks = [l for l in leakage_warnings if l.get("risk_level") == "CRITICAL"]
    high_leaks = [l for l in leakage_warnings if l.get("risk_level") == "HIGH"]
    if critical_leaks or high_leaks:
        score -= 25
        warnings.append(f"Target leakage detected: {[l['column'] for l in (critical_leaks + high_leaks)]}.")
        recommendations.append("Remove leaked features prior to model fitting.")

    # Target class imbalance warning
    if selected_target and detected_task == "binary_classification":
        val_counts = df[selected_target].value_counts(normalize=True)
        if len(val_counts) == 2 and (val_counts.min() < 0.10):
            minority_pct = val_counts.min() * 100
            score -= 10
            warnings.append(f"Severe class imbalance: minority class is only {minority_pct:.1f}% of data.")
            recommendations.append("Use Stratified K-Fold validation and evaluate with PR-AUC / balanced metrics.")

    score = max(10, min(100, score))
    is_ready = score >= 50 and total_rows >= 10

    if not recommendations:
        recommendations.append("Dataset is in optimal health for automated model training.")

    return ReadinessReport(
        is_ready=is_ready,
        overall_score=score,
        total_rows=total_rows,
        total_columns=total_columns,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        id_columns=id_cols,
        constant_columns=constant_cols,
        high_missing_columns=high_missing,
        high_cardinality_columns=high_cardinality,
        target_candidates=target_candidates,
        selected_target=selected_target,
        detected_task=detected_task,
        leakage_warnings=leakage_warnings,
        warnings=warnings,
        recommendations=recommendations,
    )
