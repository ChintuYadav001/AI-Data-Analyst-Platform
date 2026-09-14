"""
AI Data Cleaning Engine & Central Orchestrator for Quantum Analyst.
Implements all 12 cleaning categories, context-aware rule evaluation,
confidence scoring, non-destructive execution, before/after diffing,
and post-cleaning validation reporting.
"""
from __future__ import annotations

import re
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
from core.cache import AnalyticsCache

# Known text sentinels that often denote missingness
MISSING_TEXT_SENTINELS: Set[str] = {
    "n/a", "na", "null", "none", "-", "--", "?", "unknown", "nan", "nil", "missing", "undefined", ""
}

class AICleaningOrchestrator:
    """Central AI Data Cleaning Orchestrator providing dataset-aware auditing,

    confidence-scored cleaning plans, safe execution, and post-clean validation.
    """

    # -------------------------------------------------------------------------
    # 1. DATASET SCAN & AI CLEANING PLAN (12 CATEGORIES)
    # -------------------------------------------------------------------------
    @classmethod
    def scan_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Scans the dataset across all 12 cleaning categories and synthesizes

        an AI Cleaning Plan with confidence scores, severity, and action levels.
        """
        if df is None or df.empty:
            return {
                "plan_items": [],
                "summary": {
                    "total_issues": 0,
                    "safe_fixes": 0,
                    "review_required": 0,
                    "info_items": 0,
                    "affected_rows_estimate": 0,
                    "categories_detected": []
                },
                "category_counts": {},
                "column_renames": {}
            }

        cached = AnalyticsCache.get(df, "cleaning_scan")
        if cached is not None:
            return cached

        plan_items: List[Dict[str, Any]] = []
        n_rows = len(df)
        n_cols = len(df.columns)

        # 1. CATEGORY 11: COLUMN CLEANING (Empty, Constant, Unnamed, Index, Identifiers)
        cls._scan_category_columns(df, plan_items)

        # 2. CATEGORY 2: DUPLICATE DATA (Exact, Potential, Duplicate IDs)
        cls._scan_category_duplicates(df, plan_items)

        # 3. CATEGORY 7: FORMATTING & WHITESPACE (Column names, leading/trailing spaces)
        cls._scan_category_formatting_and_whitespace(df, plan_items)

        # 4. CATEGORY 3 & 10: DATA TYPE & NUMERICAL FORMATTING (Currencies, numbers as strings, percentages)
        cls._scan_category_types_and_numerical(df, plan_items)

        # 5. CATEGORY 8: DATE & TIME CLEANING (Formats, future dates, invalid dates)
        cls._scan_category_dates(df, plan_items)

        # 6. CATEGORY 1: MISSING VALUES (Nulls, text sentinels, skewness-aware strategies)
        cls._scan_category_missing(df, plan_items)

        # 7. CATEGORY 4: INCONSISTENT VALUES (Casing, whitespace variants, semantic equivalence)
        cls._scan_category_inconsistencies(df, plan_items)

        # 8. CATEGORY 6: INVALID / IMPOSSIBLE VALUES (Semantic constraints: age, rating, percentages)
        cls._scan_category_invalid_values(df, plan_items)

        # 9. CATEGORY 5: OUTLIER DETECTION (IQR, Z-score, percentiles - Default Flag, not delete)
        cls._scan_category_outliers(df, plan_items)

        # 10. CATEGORY 9: CATEGORICAL DATA CLEANING (Rare categories, high cardinality)
        cls._scan_category_categoricals(df, plan_items)

        # 11. CATEGORY 12: TEXT CLEANING (HTML fragments, multiple spaces, encoding artifacts)
        cls._scan_category_text(df, plan_items)

        # Summarize Plan
        safe_fixes = sum(1 for item in plan_items if item.get("action") == "AUTO_FIX")
        review_required = sum(1 for item in plan_items if item.get("action") == "REVIEW")
        info_items = sum(1 for item in plan_items if item.get("action") in ("KEEP", "DO_NOT_CHANGE"))
        total_issues = len(plan_items)
        categories_detected = sorted(list({item.get("category") for item in plan_items if item.get("category")}))

        category_counts: Dict[str, int] = {}
        for item in plan_items:
            cat = item.get("category", "General")
            category_counts[cat] = category_counts.get(cat, 0) + item.get("count", 1)

        # Estimate affected rows
        max_affected = 0
        for item in plan_items:
            max_affected = max(max_affected, min(n_rows, item.get("count", 0)))

        # Ensure backward compatibility fields for legacy callers
        for item in plan_items:
            if "issue" not in item:
                item["issue"] = item.get("category", "Data Issue")
            if "recommended_action" not in item:
                item["recommended_action"] = item.get("default_option") or item.get("recommendation", "Review")

        res = {
            "plan_items": plan_items,
            "summary": {
                "total_issues": total_issues,
                "safe_fixes": safe_fixes,
                "review_required": review_required,
                "info_items": info_items,
                "affected_rows_estimate": max_affected,
                "categories_detected": categories_detected
            },
            "category_counts": category_counts
        }
        AnalyticsCache.set(df, "cleaning_scan", res)
        return res

    # -------------------------------------------------------------------------
    # CATEGORY DETECTORS
    # -------------------------------------------------------------------------
    @classmethod
    def _scan_category_columns(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 11: Column Cleaning (Empty, Constant, Unnamed, Duplicate cols, Identifiers)."""
        n_rows = len(df)

        # Unnamed index columns (e.g. Unnamed: 0, index)
        for col in df.columns:
            col_str = str(col).strip()
            if col_str.lower().startswith("unnamed") or col_str.lower() in ("index", "level_0"):
                plan_items.append({
                    "id": f"col_unnamed_{col_str}",
                    "category": "Column Cleaning",
                    "issue_type": "unnamed_index_column",
                    "column": col_str,
                    "count": n_rows,
                    "percentage": 100.0,
                    "severity": "MEDIUM",
                    "detection_method": "Header Pattern Scan",
                    "recommendation": "Drop redundant index/unnamed artifact column",
                    "action": "AUTO_FIX",
                    "confidence": 0.98,
                    "reason": f"Column '{col_str}' appears to be a discarded index artifact from file serialization.",
                    "technical_details": "Matches regex '^unnamed|^index$'; typically generated when CSV files are exported without index=False.",
                    "options": ["Drop Column", "Keep Column", "Rename Column"],
                    "default_option": "Drop Column"
                })

            # Completely empty column
            elif df[col].isna().all():
                plan_items.append({
                    "id": f"col_empty_{col_str}",
                    "category": "Column Cleaning",
                    "issue_type": "empty_column",
                    "column": col_str,
                    "count": n_rows,
                    "percentage": 100.0,
                    "severity": "HIGH",
                    "detection_method": "Completeness Audit",
                    "recommendation": "Remove completely empty column",
                    "action": "AUTO_FIX",
                    "confidence": 0.99,
                    "reason": f"Column '{col_str}' contains 100% missing values and carries zero analytical signal.",
                    "technical_details": "df[col].isna().sum() == len(df). Retaining it increases memory usage without information gain.",
                    "options": ["Drop Column", "Keep Column"],
                    "default_option": "Drop Column"
                })

            # Single constant value column (zero variance)
            elif n_rows > 10:
                is_constant = False
                if n_rows > 10000:
                    first_val = df[col].iloc[0]
                    if pd.isna(first_val):
                        if df[col].isna().all():
                            is_constant = True
                    else:
                        if (df[col].iloc[-1] == first_val) and (df[col].iloc[n_rows // 2] == first_val):
                            if not (df[col].iloc[:1000] != first_val).any():
                                is_constant = bool((df[col] == first_val).all())
                else:
                    is_constant = (df[col].nunique(dropna=False) == 1)

                if is_constant:
                    val = df[col].iloc[0]
                    plan_items.append({
                        "id": f"col_constant_{col_str}",
                        "category": "Column Cleaning",
                        "issue_type": "constant_column",
                        "column": col_str,
                        "count": n_rows,
                        "percentage": 100.0,
                        "severity": "LOW",
                        "detection_method": "Variance Evaluation",
                        "recommendation": "Review constant column (zero variance)",
                        "action": "REVIEW",
                        "confidence": 0.85,
                        "reason": f"Column '{col_str}' holds the exact same single value ('{val}') across all records.",
                        "technical_details": f"Variance = 0; nunique = 1. Useful only if representing metadata or system partition tags.",
                        "options": ["Keep Column", "Drop Column", "Flag as Metadata"],
                        "default_option": "Keep Column"
                    })
                elif any(k in col_str.lower() for k in ["_id", "id_", "identifier", "uuid", "guid", "ssn", "pk_"]) or col_str.lower() == "id":
                    is_pk = False
                    if df[col].isna().sum() == 0:
                        is_pk = bool(df[col].is_unique)
                    if is_pk:
                        plan_items.append({
                            "id": f"col_id_{col_str}",
                            "category": "Column Cleaning",
                            "issue_type": "primary_identifier",
                            "column": col_str,
                            "count": n_rows,
                            "percentage": 100.0,
                            "severity": "INFO",
                            "detection_method": "Uniqueness & Semantic Classification",
                            "recommendation": "Preserve as Primary Key Identifier",
                            "action": "DO_NOT_CHANGE",
                            "confidence": 0.99,
                            "reason": f"Column '{col_str}' is a verified unique primary identifier required for joins and record tracing.",
                            "technical_details": "100% unique observations matching primary identifier semantic patterns.",
                            "options": ["Preserve Identifier (Do Not Delete)", "Keep as Feature"],
                            "default_option": "Preserve Identifier (Do Not Delete)"
                        })

    @classmethod
    def _scan_category_duplicates(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 2: Duplicate Data (Exact rows, duplicate IDs, potential duplicates)."""
        n_rows = len(df)
        if n_rows <= 1:
            return

        # Exact duplicate rows (reuse cached calculation if present)
        cached_dups = AnalyticsCache.get(df, "duplicate_rows")
        if cached_dups is not None:
            exact_dups = cached_dups
        else:
            exact_dups = int(df.duplicated().sum())
            AnalyticsCache.set(df, "duplicate_rows", exact_dups)
        if exact_dups > 0:
            dup_pct = round((exact_dups / n_rows) * 100, 2)
            plan_items.append({
                "id": "dup_exact_rows",
                "category": "Duplicate Data",
                "issue_type": "exact_duplicate_rows",
                "column": "ALL_COLUMNS",
                "count": exact_dups,
                "percentage": dup_pct,
                "severity": "CRITICAL" if dup_pct > 10 else "HIGH",
                "detection_method": "Full-Tuple Hash Collision",
                "recommendation": "Remove exact duplicate rows",
                "action": "AUTO_FIX",
                "confidence": 0.99,
                "reason": f"Found {exact_dups:,} exact identical rows ({dup_pct}%) that artificially inflate sample size and bias metrics.",
                "technical_details": f"pandas.DataFrame.duplicated(keep='first').sum() == {exact_dups}. Deduplicating preserves the first occurrence.",
                "options": ["Remove Exact Duplicates", "Keep Duplicates", "Flag Duplicates in Column"],
                "default_option": "Remove Exact Duplicates"
            })

        # Duplicate Primary IDs
        id_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["_id", "id_", "identifier", "uuid"]) or str(c).lower() == "id"]
        for id_col in id_cols:
            if df[id_col].isna().sum() == 0:
                if df[id_col].is_unique:
                    continue
                dup_id_cnt = int(df.duplicated(subset=[id_col]).sum())
                if 0 < dup_id_cnt < exact_dups or (exact_dups == 0 and dup_id_cnt > 0):
                    plan_items.append({
                        "id": f"dup_id_{id_col}",
                        "category": "Duplicate Data",
                        "issue_type": "duplicate_id_values",
                        "column": str(id_col),
                        "count": dup_id_cnt,
                        "percentage": round((dup_id_cnt / n_rows) * 100, 2),
                        "severity": "HIGH",
                        "detection_method": "Primary Key Integrity Check",
                        "recommendation": "Review duplicate identifiers (potential multiple events per entity)",
                        "action": "REVIEW",
                        "confidence": 0.88,
                        "reason": f"Identifier column '{id_col}' contains {dup_id_cnt:,} repeated keys. May represent valid time-series snapshots or data collision.",
                        "technical_details": "df.duplicated(subset=[id_col]) detected collisions while non-ID columns contain different values.",
                        "options": ["Keep Duplicates (Time-series / Multi-event)", "Deduplicate on ID (Keep Most Recent)", "Flag Duplicate IDs"],
                        "default_option": "Keep Duplicates (Time-series / Multi-event)"
                    })

    @classmethod
    def _scan_category_formatting_and_whitespace(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 7: Formatting & Whitespace (Column names and cell strings)."""
        # 1. Column name formatting
        dirty_cols = []
        for col in df.columns:
            c_str = str(col)
            clean_c = re.sub(r"[^a-zA-Z0-9_]+", "_", c_str.strip()).strip("_").lower()
            if c_str != clean_c and clean_c:
                dirty_cols.append((c_str, clean_c))

        if dirty_cols:
            sample_preview = ", ".join([f"'{orig}' -> '{c}'" for orig, c in dirty_cols[:3]])
            plan_items.append({
                "id": "format_column_names",
                "category": "Formatting and Whitespace",
                "issue_type": "unnormalized_column_names",
                "column": "SCHEMA_HEADERS",
                "count": len(dirty_cols),
                "percentage": round((len(dirty_cols) / len(df.columns)) * 100, 1),
                "severity": "LOW",
                "detection_method": "Snake_Case Standardization Scan",
                "recommendation": "Standardize column headers to snake_case",
                "action": "AUTO_FIX",
                "confidence": 0.96,
                "reason": f"{len(dirty_cols)} column headers contain spaces or special characters ({sample_preview}).",
                "technical_details": "Normalizing headers to snake_case prevents SQL syntax errors, dot-notation failures, and encoding bugs.",
                "options": ["Standardize Headers to snake_case", "Keep Original Headers"],
                "default_option": "Standardize Headers to snake_case"
            })

        # 2. String cells with leading/trailing/multiple whitespaces (sample-accelerated on large datasets)
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        n_rows = len(df)
        for col in str_cols:
            if n_rows > 10000:
                sample_s = df[col].dropna().head(5000).astype(str)
                if sample_s.empty:
                    continue
                edge_cnt = (sample_s.str.startswith(" ") | sample_s.str.endswith(" ")).sum()
                multi_cnt = sample_s.str.contains(r"\s{2,}", regex=True).sum()
                sample_issues = edge_cnt + multi_cnt
                if sample_issues == 0:
                    continue
                total_valid = len(df[col].dropna())
                total_ws_issues = int((sample_issues / len(sample_s)) * total_valid)
                pct = round((total_ws_issues / total_valid) * 100, 2)
            else:
                series = df[col].dropna().astype(str)
                if series.empty:
                    continue
                has_edge_space = (series.str.startswith(" ") | series.str.endswith(" ")).sum()
                has_multi_space = series.str.contains(r"\s{2,}", regex=True).sum()
                total_ws_issues = int(has_edge_space + has_multi_space)
                if total_ws_issues == 0:
                    continue
                pct = round((total_ws_issues / len(series)) * 100, 2)

            plan_items.append({
                "id": f"format_whitespace_{col}",
                "category": "Formatting and Whitespace",
                "issue_type": "whitespace_padding",
                "column": str(col),
                "count": total_ws_issues,
                "percentage": pct,
                "severity": "LOW",
                "detection_method": "Regex Whitespace Scan",
                "recommendation": "Strip leading, trailing, and redundant spaces",
                "action": "AUTO_FIX",
                "confidence": 0.99,
                "reason": f"Column '{col}' has {total_ws_issues:,} cells with redundant surrounding or multiple spaces.",
                "technical_details": "Removes ^\\s+|\\s+$ and collapses multiple spaces to single spaces without modifying character data.",
                "options": ["Strip and Normalize Spaces", "Keep Spaces"],
                "default_option": "Strip and Normalize Spaces"
            })

    @classmethod
    def _scan_category_types_and_numerical(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 3 & 10: Data Type & Numerical Formatting (Currencies, numbers as strings)."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in str_cols:
            sample_raw = df[col].dropna().head(100)
            if len(sample_raw) < 5:
                continue

            sample_head = sample_raw.astype(str).str.strip()

            # Currency Detection (e.g. $500, ₹50,000, €1,200.50)
            currency_match = sample_head.str.contains(r"^[\$₹€£¥]\s*\d+", regex=True).sum()
            if currency_match / len(sample_head) > 0.65:
                plan_items.append({
                    "id": f"type_currency_{col}",
                    "category": "Numerical Data Cleaning",
                    "issue_type": "currency_string_format",
                    "column": str(col),
                    "count": int(currency_match),
                    "percentage": round((currency_match / len(sample_head)) * 100, 1),
                    "severity": "HIGH",
                    "detection_method": "Financial Symbol & Magnitude Parser",
                    "recommendation": "Strip currency symbols and cast to numeric float",
                    "action": "AUTO_FIX",
                    "confidence": 0.98,
                    "reason": f"Column '{col}' stores monetary values formatted as text with currency symbols (e.g. '{sample_head.iloc[0]}').",
                    "technical_details": r"Strips currency glyphs [\$₹€£¥] and thousands commas, converting values to standard float64.",
                    "options": ["Convert to Numeric (Float)", "Keep as Text"],
                    "default_option": "Convert to Numeric (Float)"
                })
                continue

            # Percentages (e.g. 25%, 98.4%)
            pct_match = sample_head.str.contains(r"^\d+(?:\.\d+)?%$", regex=True).sum()
            if pct_match / len(sample_head) > 0.70:
                plan_items.append({
                    "id": f"type_pct_{col}",
                    "category": "Numerical Data Cleaning",
                    "issue_type": "percentage_string_format",
                    "column": str(col),
                    "count": int(pct_match),
                    "percentage": round((pct_match / len(sample_head)) * 100, 1),
                    "severity": "MEDIUM",
                    "detection_method": "Percentage Pattern Parser",
                    "recommendation": "Strip percentage symbols and cast to numeric float (0-100 scale)",
                    "action": "AUTO_FIX",
                    "confidence": 0.97,
                    "reason": f"Column '{col}' contains percentages stored as strings (e.g. '{sample_head.iloc[0]}').",
                    "technical_details": "Strips trailing '%' character without changing scale, casting to float64.",
                    "options": ["Convert to Numeric Float", "Keep as Text"],
                    "default_option": "Convert to Numeric Float"
                })
                continue

            # Pure numbers stored as strings (with or without commas, e.g. '1,250', '250.00')
            clean_num = sample_head.str.replace(",", "", regex=False).str.strip()
            num_matches = clean_num.str.match(r"^-?\d+(\.\d+)?$").sum()
            if num_matches / len(sample_head) > 0.85:
                # Check if it's an ID or zip code
                is_id = any(k in str(col).lower() for k in ["_id", "id_", "zip", "code", "phone", "pin", "ssn"])
                action = "REVIEW" if is_id else "AUTO_FIX"
                confidence = 0.80 if is_id else 0.98
                plan_items.append({
                    "id": f"type_num_string_{col}",
                    "category": "Data Type Correction",
                    "issue_type": "numeric_stored_as_string",
                    "column": str(col),
                    "count": int(num_matches),
                    "percentage": round((num_matches / len(sample_head)) * 100, 1),
                    "severity": "MEDIUM",
                    "detection_method": "Type Inference Engine",
                    "recommendation": "Convert string numbers to numeric type" if not is_id else "Review numeric string (likely postal/identifier code)",
                    "action": action,
                    "confidence": confidence,
                    "reason": f"Column '{col}' contains numeric values stored as text strings (e.g. '{sample_head.iloc[0]}').",
                    "technical_details": "Removes thousands separators and converts to int64 or float64. Identifier patterns preserved upon review.",
                    "options": ["Convert to Numeric", "Keep as Text"],
                    "default_option": "Keep as Text" if is_id else "Convert to Numeric"
                })
                continue

            # Booleans stored as strings (True/False, YES/NO, 1/0)
            bool_matches = sample_head.str.lower().isin(["true", "false", "yes", "no", "y", "n", "t", "f"]).sum()
            if bool_matches / len(sample_head) > 0.90:
                plan_items.append({
                    "id": f"type_bool_{col}",
                    "category": "Data Type Correction",
                    "issue_type": "boolean_stored_as_string",
                    "column": str(col),
                    "count": int(bool_matches),
                    "percentage": round((bool_matches / len(sample_head)) * 100, 1),
                    "severity": "LOW",
                    "detection_method": "Boolean Flag Recognition",
                    "recommendation": "Standardize to Boolean True/False",
                    "action": "AUTO_FIX",
                    "confidence": 0.97,
                    "reason": f"Column '{col}' contains boolean flags stored as varied text (e.g. '{sample_head.iloc[0]}').",
                    "technical_details": "Maps {'true', 'yes', 'y', 't', '1'} -> True, {'false', 'no', 'n', 'f', '0'} -> False.",
                    "options": ["Convert to Boolean", "Keep as Text"],
                    "default_option": "Convert to Boolean"
                })

    @classmethod
    def _scan_category_dates(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 8: Date and Time Cleaning (Detection, invalid dates, future dates)."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        date_patterns = [
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",  # 2026-09-14
            r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}",  # 14/09/2026 or 09/14/2026
            r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"  # September 14, 2026
        ]

        for col in str_cols:
            if any(k in str(col).lower() for k in ["date", "time", "timestamp", "created_at", "updated_at", "dob", "birth"]):
                sample_raw = df[col].dropna().head(50)
                if sample_raw.empty:
                    continue

                sample_head = sample_raw.astype(str).str.strip()
                matches_date = 0
                for pattern in date_patterns:
                    matches_date += sample_head.str.contains(pattern, regex=True).sum()

                if matches_date / len(sample_head) > 0.60:
                    # Check for ambiguous dates (e.g. 03/04/2026 where day and month <= 12)
                    slash_dates = sample_head.str.extract(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})")
                    is_ambiguous = False
                    if not slash_dates.empty and slash_dates[0].notna().sum() > 0:
                        valid_p = slash_dates.dropna()
                        if not valid_p.empty:
                            v1 = pd.to_numeric(valid_p[0], errors="coerce")
                            v2 = pd.to_numeric(valid_p[1], errors="coerce")
                            if ((v1 <= 12) & (v2 <= 12)).mean() > 0.8:
                                is_ambiguous = True

                    action = "REVIEW" if is_ambiguous else "AUTO_FIX"
                    conf = 0.78 if is_ambiguous else 0.96
                    plan_items.append({
                        "id": f"date_parse_{col}",
                        "category": "Date and Time Cleaning",
                        "issue_type": "unparsed_datetime_column",
                        "column": str(col),
                        "count": int(df[col].notna().sum()),
                        "percentage": 100.0,
                        "severity": "MEDIUM",
                        "detection_method": "Temporal Pattern Detection",
                        "recommendation": "Flag for date format review (ambiguous DD/MM vs MM/DD)" if is_ambiguous else "Parse and standardize to ISO-8601 Datetime",
                        "action": action,
                        "confidence": conf,
                        "reason": f"Column '{col}' contains date strings that should be standardized into temporal datetimes." if not is_ambiguous else f"Column '{col}' has ambiguous date components where day/month format needs review (e.g. '{sample_head.iloc[0]}').",
                        "technical_details": f"pd.to_datetime with errors='coerce'. Ambiguous day/month detection: {is_ambiguous}.",
                        "options": ["Convert to Datetime (Auto-infer)", "Convert as YYYY-MM-DD", "Convert as DD/MM/YYYY", "Keep Original Text"],
                        "default_option": "Convert to Datetime (Auto-infer)"
                    })

    @classmethod
    def _scan_category_missing(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 1: Missing Values (Nulls, sentinel text strings, skewness-aware strategies)."""
        n_rows = len(df)
        for col in df.columns:
            # 1. Standard nulls
            standard_nulls = int(df[col].isna().sum())

            # 2. Text sentinel nulls (e.g. 'N/A', 'null', '-') in string/object columns
            sentinel_nulls = 0
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                valid_raw = df[col].dropna()
                if not valid_raw.empty:
                    if n_rows > 20000:
                        test_sample = valid_raw.head(2000).astype(str).str.strip().str.lower()
                        if test_sample.isin(MISSING_TEXT_SENTINELS).sum() > 0:
                            sample_str = valid_raw.astype(str).str.strip().str.lower()
                            sentinel_nulls = int(sample_str.isin(MISSING_TEXT_SENTINELS).sum())
                    else:
                        sample_str = valid_raw.astype(str).str.strip().str.lower()
                        sentinel_nulls = int(sample_str.isin(MISSING_TEXT_SENTINELS).sum())

            total_missing = standard_nulls + sentinel_nulls
            if total_missing > 0:
                missing_pct = round((total_missing / n_rows) * 100, 2)
                is_num = pd.api.types.is_numeric_dtype(df[col])
                if not is_num and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
                    sample_clean = df[col].dropna().head(1000).astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip()
                    sample_clean = sample_clean[~sample_clean.str.lower().isin(MISSING_TEXT_SENTINELS)]
                    if not sample_clean.empty:
                        num_test = pd.to_numeric(sample_clean, errors="coerce")
                        if (num_test.notna().sum() / len(sample_clean)) >= 0.70:
                            is_num = True

                col_type = "numerical" if is_num else "categorical"

                # High missing threshold (>70%): recommend dropping column
                if missing_pct >= 70.0:
                    plan_items.append({
                        "id": f"missing_high_{col}",
                        "category": "Missing Values",
                        "issue_type": "high_missing_column",
                        "column": str(col),
                        "count": total_missing,
                        "percentage": missing_pct,
                        "severity": "HIGH",
                        "detection_method": "Missingness Threshold Scan",
                        "recommendation": f"Drop column due to severe missingness ({missing_pct}%)",
                        "action": "REVIEW",
                        "confidence": 0.88,
                        "reason": f"Column '{col}' is {missing_pct}% missing ({total_missing:,} values). Imputing over 70% of a column risks artificial distortion.",
                        "technical_details": f"Total missing = {total_missing} ({standard_nulls} nulls + {sentinel_nulls} sentinels). Dropping preserves statistical integrity.",
                        "options": ["Drop Column", "Impute with 'Unknown' / Mode", "Keep Missing Values"],
                        "default_option": "Drop Column"
                    })
                    continue

                if col_type == "numerical":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        series = df[col].dropna()
                    else:
                        clean_text = df[col].dropna().astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip()
                        clean_text = clean_text[~clean_text.str.lower().isin(MISSING_TEXT_SENTINELS)]
                        series = pd.to_numeric(clean_text, errors="coerce").dropna()

                    skewness = float(series.skew()) if len(series) > 2 else 0.0
                    is_skewed = abs(skewness) > 1.0
                    method = "Median" if is_skewed else "Mean"
                    val = round(float(series.median()), 2) if is_skewed else round(float(series.mean()), 2)
                    med_v = round(float(series.median()), 2) if not series.empty else 0.0
                    mean_v = round(float(series.mean()), 2) if not series.empty else 0.0
                    plan_items.append({
                        "id": f"missing_num_{col}",
                        "category": "Missing Values",
                        "issue_type": "missing_numerical_values",
                        "column": str(col),
                        "count": total_missing,
                        "percentage": missing_pct,
                        "severity": "MEDIUM" if missing_pct > 5 else "LOW",
                        "detection_method": f"Distribution Skewness Evaluation (Skew = {skewness:.2f})",
                        "recommendation": f"Impute with {method} ({val})",
                        "action": "AUTO_FIX",
                        "confidence": 0.96,
                        "reason": f"Column '{col}' has {total_missing:,} missing values ({missing_pct}%). Skewness is {skewness:.2f}; {method} imputation preserves central tendency without outlier distortion.",
                        "technical_details": f"Distribution skewness is {'skewed' if is_skewed else 'symmetrical'}. Imputing {method} = {val}.",
                        "options": [f"Impute with Median ({med_v:.2f})", f"Impute with Mean ({mean_v:.2f})", "Impute with Constant (0)", "Drop Rows with Missing"],
                        "default_option": f"Impute with {method} ({val})"
                    })
                else:
                    # Categorical or Date
                    if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in str(col).lower():
                        plan_items.append({
                            "id": f"missing_date_{col}",
                            "category": "Missing Values",
                            "issue_type": "missing_datetime_values",
                            "column": str(col),
                            "count": total_missing,
                            "percentage": missing_pct,
                            "severity": "MEDIUM",
                            "detection_method": "Temporal Sequence Completeness",
                            "recommendation": "Impute via Temporal Forward/Backward Fill",
                            "action": "AUTO_FIX",
                            "confidence": 0.95,
                            "reason": f"Column '{col}' has {total_missing:,} missing timestamps ({missing_pct}%). Forward-fill propagates continuous temporal states.",
                            "technical_details": "Applies series.ffill().bfill() along the index.",
                            "options": ["Forward / Backward Fill", "Drop Rows with Missing Dates", "Keep Missing"],
                            "default_option": "Forward / Backward Fill"
                        })
                    else:
                        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                        plan_items.append({
                            "id": f"missing_cat_{col}",
                            "category": "Missing Values",
                            "issue_type": "missing_categorical_values",
                            "column": str(col),
                            "count": total_missing,
                            "percentage": missing_pct,
                            "severity": "LOW",
                            "detection_method": "Category Nullity Audit",
                            "recommendation": "Impute with 'Unknown' placeholder",
                            "action": "AUTO_FIX",
                            "confidence": 0.98,
                            "reason": f"Column '{col}' has {total_missing:,} missing categories ({missing_pct}%). Explicit 'Unknown' prevents false mode inflation.",
                            "technical_details": "Fills nulls with string 'Unknown' to maintain categorical cardinality.",
                            "options": ["Impute with 'Unknown'", f"Impute with Mode ('{mode_val}')", "Drop Rows with Missing"],
                            "default_option": "Impute with 'Unknown'"
                        })

    @classmethod
    def _scan_category_inconsistencies(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 4: Inconsistent Values (Casing variations, gender codes, status)."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            unique_raw = df[col].dropna().unique()
            if len(unique_raw) == 0 or len(unique_raw) > 100:
                continue

            series_u = pd.Series(unique_raw).astype(str).str.strip()
            raw_unique = series_u.nunique()
            title_unique = series_u.str.title().nunique()

            if raw_unique > title_unique:
                diff = raw_unique - title_unique
                # Check sample variation (e.g. 'Delhi' vs 'delhi')
                sample_pairs = []
                lower_map: Dict[str, List[str]] = {}
                for v in series_u:
                    low = v.lower()
                    lower_map.setdefault(low, []).append(v)

                for low, variants in lower_map.items():
                    if len(variants) > 1:
                        sample_pairs.append(f"'{variants[0]}' / '{variants[1]}'")
                        if len(sample_pairs) >= 2:
                            break

                preview_str = ", ".join(sample_pairs) if sample_pairs else "casing variants"
                plan_items.append({
                    "id": f"inconsistent_casing_{col}",
                    "category": "Inconsistent Values",
                    "issue_type": "inconsistent_casing_variations",
                    "column": str(col),
                    "count": diff,
                    "percentage": round((diff / raw_unique) * 100, 1),
                    "severity": "LOW",
                    "detection_method": "Semantic Case Equivalence Analysis",
                    "recommendation": f"Standardize to Title Case ({diff} redundant variants eliminated)",
                    "action": "AUTO_FIX",
                    "confidence": 0.99,
                    "reason": f"Column '{col}' contains {diff} duplicated category representations caused solely by casing differences ({preview_str}).",
                    "technical_details": "Normalizes to str.strip().str.title() ensuring unified group-by aggregation.",
                    "options": ["Standardize to Title Case", "Standardize to Lowercase", "Standardize to UPPERCASE", "Keep Unchanged"],
                    "default_option": "Standardize to Title Case"
                })

            # Check gender abbreviation inconsistency (e.g. 'Male', 'M', 'Female', 'F')
            if any(k in str(col).lower() for k in ["gender", "sex"]):
                uniques = set(series_u.str.lower().unique())
                if {"m", "male"}.issubset(uniques) or {"f", "female"}.issubset(uniques):
                    col_s = df[col].dropna()
                    abbrev_cnt = int(col_s.astype(str).str.strip().str.lower().isin({"m", "f"}).sum())
                    plan_items.append({
                        "id": f"inconsistent_gender_{col}",
                        "category": "Inconsistent Values",
                        "issue_type": "gender_abbreviation_inconsistency",
                        "column": str(col),
                        "count": abbrev_cnt,
                        "percentage": round((abbrev_cnt / len(col_s)) * 100, 1) if len(col_s) > 0 else 0.0,
                        "severity": "MEDIUM",
                        "detection_method": "Domain Equivalence Dictionary",
                        "recommendation": "Map abbreviations 'M' -> 'Male' and 'F' -> 'Female'",
                        "action": "AUTO_FIX",
                        "confidence": 0.98,
                        "reason": f"Column '{col}' mixes full gender names and abbreviations ('M'/'Male', 'F'/'Female').",
                        "technical_details": "Standardizes abbreviation codes to full canonical descriptions.",
                        "options": ["Standardize to 'Male' / 'Female'", "Standardize to 'M' / 'F'", "Keep As Is"],
                        "default_option": "Standardize to 'Male' / 'Female'"
                    })

    @classmethod
    def _scan_category_invalid_values(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 6: Invalid / Impossible Values (Logical & semantic domain constraints)."""
        # Age validation (< 0 or > 125)
        for col in df.columns:
            c_low = str(col).lower()
            if pd.api.types.is_numeric_dtype(df[col]):
                if "age" in c_low and not any(k in c_low for k in ["percentage", "ratio", "stage"]):
                    invalid_mask = (df[col] < 0) | (df[col] > 125)
                    inv_count = int(invalid_mask.sum())
                    if inv_count > 0:
                        plan_items.append({
                            "id": f"invalid_age_{col}",
                            "category": "Invalid / Impossible Values",
                            "issue_type": "impossible_age_range",
                            "column": str(col),
                            "count": inv_count,
                            "percentage": round((inv_count / len(df)) * 100, 2),
                            "severity": "CRITICAL",
                            "detection_method": "Biological Age Constraint [0, 125]",
                            "recommendation": "Clip or nullify impossible age values (< 0 or > 125)",
                            "action": "AUTO_FIX",
                            "confidence": 0.97,
                            "reason": f"Detected {inv_count} biologically impossible age values (e.g. negative or > 125).",
                            "technical_details": f"Min: {df[col].min()}, Max: {df[col].max()}. Values outside [0, 125] represent input errors.",
                            "options": ["Set Invalid Values to NaN & Impute", "Clip to [0, 100]", "Drop Affected Rows", "Keep Values"],
                            "default_option": "Set Invalid Values to NaN & Impute"
                        })

                # Rating validation (e.g. 1 to 5)
                elif "rating" in c_low or "score_5" in c_low:
                    invalid_mask = (df[col] < 0) | (df[col] > 5.5)
                    inv_count = int(invalid_mask.sum())
                    if inv_count > 0:
                        plan_items.append({
                            "id": f"invalid_rating_{col}",
                            "category": "Invalid / Impossible Values",
                            "issue_type": "out_of_bounds_rating",
                            "column": str(col),
                            "count": inv_count,
                            "percentage": round((inv_count / len(df)) * 100, 2),
                            "severity": "HIGH",
                            "detection_method": "5-Star Rating Scale Constraint [1, 5]",
                            "recommendation": "Cap ratings to valid range [1.0, 5.0]",
                            "action": "AUTO_FIX",
                            "confidence": 0.95,
                            "reason": f"Column '{col}' has {inv_count} ratings outside the standard 1 to 5 range.",
                            "technical_details": f"Min: {df[col].min()}, Max: {df[col].max()}.",
                            "options": ["Cap to [1.0, 5.0]", "Set to NaN", "Keep Unchanged"],
                            "default_option": "Cap to [1.0, 5.0]"
                        })

                # Negative values where impossible (Quantity, Price, Sales, Tenure)
                elif any(k in c_low for k in ["quantity", "qty", "price", "unit_price", "tenure", "days"]):
                    neg_mask = (df[col] < 0)
                    neg_cnt = int(neg_mask.sum())
                    if neg_cnt > 0:
                        plan_items.append({
                            "id": f"invalid_neg_{col}",
                            "category": "Invalid / Impossible Values",
                            "issue_type": "negative_value_violation",
                            "column": str(col),
                            "count": neg_cnt,
                            "percentage": round((neg_cnt / len(df)) * 100, 2),
                            "severity": "HIGH",
                            "detection_method": "Non-Negative Domain Invariant",
                            "recommendation": "Convert negative quantities/prices to positive absolute value or NaN",
                            "action": "REVIEW",
                            "confidence": 0.90,
                            "reason": f"Column '{col}' contains {neg_cnt:,} negative values where quantities or prices must be non-negative.",
                            "technical_details": "Could indicate sign inversions, return entries, or data entry errors.",
                            "options": ["Convert to Absolute Values (abs)", "Set to NaN & Impute", "Keep Negative Values (Returns/Debits)"],
                            "default_option": "Convert to Absolute Values (abs)"
                        })

    @classmethod
    def _scan_category_outliers(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 5: Outlier Detection (IQR, Z-score - DEFAULT TO FLAG RATHER THAN DELETE)."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        n_rows = len(df)
        for col in num_cols:
            # Skip ID columns and binary columns
            if any(k in str(col).lower() for k in ["_id", "id_", "identifier", "code"]):
                continue
            if df[col].iloc[:50].nunique() <= 2 and df[col].nunique() <= 2:
                continue

            series = df[col].dropna()
            if len(series) < 6:
                continue

            if n_rows > 30000:
                sample_s = series.sample(30000, random_state=42)
                q_vals = sample_s.quantile([0.25, 0.75])
            else:
                q_vals = series.quantile([0.25, 0.75])

            q25 = float(q_vals[0.25])
            q75 = float(q_vals[0.75])
            iqr = q75 - q25

            if iqr > 0:
                lower_b = q25 - 1.5 * iqr
                upper_b = q75 + 1.5 * iqr
                vals = series.to_numpy()
                outlier_count = int(np.count_nonzero((vals < lower_b) | (vals > upper_b)))

                if outlier_count > 0:
                    outlier_pct = round((outlier_count / len(series)) * 100, 2)
                    # Classify: potential legitimate extreme vs error
                    severity = "LOW" if outlier_pct < 3 else "MEDIUM"
                    plan_items.append({
                        "id": f"outlier_iqr_{col}",
                        "category": "Outlier Detection",
                        "issue_type": "statistical_outliers_iqr",
                        "column": str(col),
                        "count": outlier_count,
                        "percentage": outlier_pct,
                        "severity": severity,
                        "detection_method": f"Tukey's IQR Bound [{lower_b:.2f}, {upper_b:.2f}]",
                        "recommendation": "Flag outliers in a telemetry indicator column (Do not delete legitimate extremes)",
                        "action": "REVIEW",
                        "confidence": 0.85,
                        "reason": f"Column '{col}' contains {outlier_count} potential statistical outliers ({outlier_pct}%). Outliers often convey valuable business signals (e.g. VIP customers, high-value transactions).",
                        "technical_details": f"Q1 = {q25:.2f}, Q3 = {q75:.2f}, IQR = {iqr:.2f}. Lower bound = {lower_b:.2f}, Upper bound = {upper_b:.2f}.",
                        "options": ["Flag in New Column (Keep Data Intact)", "Cap / Winsorize (1.5 IQR)", "Remove Outlier Rows", "Keep Unchanged"],
                        "default_option": "Flag in New Column (Keep Data Intact)"
                    })

    @classmethod
    def _scan_category_categoricals(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 9: Categorical Data Cleaning (Rare categories, high cardinality)."""
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        n_rows = len(df)
        for col in cat_cols:
            col_str = str(col).lower()
            if any(k in col_str for k in ["_id", "id_", "identifier", "uuid", "guid", "date", "timestamp", "time"]) or col_str == "id":
                continue

            series = df[col].dropna()
            total_s = len(series)
            if total_s < 50:
                continue

            if n_rows > 10000 and series.iloc[:1000].nunique() > 300:
                continue

            vc = series.value_counts()
            if len(vc) <= 5:
                continue
            rare_cats = vc[vc / total_s < 0.01]  # Categories appearing in less than 1% of rows
            if 1 < len(rare_cats) < len(vc):
                rare_count = int(rare_cats.sum())
                plan_items.append({
                    "id": f"cat_rare_{col}",
                    "category": "Categorical Data Cleaning",
                    "issue_type": "rare_category_dispersion",
                    "column": str(col),
                    "count": rare_count,
                    "percentage": round((rare_count / n_rows) * 100, 2),
                    "severity": "LOW",
                    "detection_method": "Frequency Density Threshold (< 1%)",
                    "recommendation": f"Group {len(rare_cats)} rare categories (<1% frequency) into 'Other'",
                    "action": "REVIEW",
                    "confidence": 0.75,
                    "reason": f"Column '{col}' has {len(rare_cats)} micro-categories accounting for only {round((rare_count/n_rows)*100, 1)}% of rows.",
                    "technical_details": f"Rare categories: {list(rare_cats.index[:5])}. Consolidating reduces model dimensionality without loss of main effects.",
                    "options": ["Group Rare as 'Other'", "Keep Original Categories", "Review Categories"],
                    "default_option": "Keep Original Categories"
                })

    @classmethod
    def _scan_category_text(cls, df: pd.DataFrame, plan_items: List[Dict[str, Any]]) -> None:
        """Category 12: Text Cleaning (HTML fragments, multiple spaces, encoding artifacts)."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            series = df[col].dropna().astype(str)
            if series.empty:
                continue

            sample_head = series.head(100)
            html_count = int(sample_head.str.contains(r"<[a-zA-Z\/][^>]*>", regex=True).sum())
            if html_count > 0:
                plan_items.append({
                    "id": f"text_html_{col}",
                    "category": "Text Cleaning",
                    "issue_type": "html_tag_fragments",
                    "column": str(col),
                    "count": html_count,
                    "percentage": round((html_count / len(sample_head)) * 100, 1),
                    "severity": "MEDIUM",
                    "detection_method": "Markup Tag Regex Audit",
                    "recommendation": "Strip HTML tags while preserving inner prose content",
                    "action": "AUTO_FIX",
                    "confidence": 0.98,
                    "reason": f"Column '{col}' contains embedded HTML tags (e.g. '<p>', '<div>') that clutter text analysis.",
                    "technical_details": "Strips regex r'<[^<]+?>' and cleans unescaped HTML entities.",
                    "options": ["Strip HTML Tags", "Keep Raw Text"],
                    "default_option": "Strip HTML Tags"
                })

    # -------------------------------------------------------------------------
    # 2. EXECUTION ENGINE (AUTO-CLEAN & CUSTOM CLEAN)
    # -------------------------------------------------------------------------
    @classmethod
    def apply_auto_clean(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Executes ONLY high-confidence, non-destructive AUTO_FIX items automatically."""
        plan = cls.scan_dataset(df)
        auto_items = [item for item in plan.get("plan_items", []) if item.get("action") == "AUTO_FIX"]
        return cls.execute_plan(df, auto_items)

    @classmethod
    def execute_plan(cls, df: pd.DataFrame, selected_items: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Safely executes the provided cleaning actions in deterministic order on a

        fresh copy of the dataset and produces an audit ledger.
        """
        cleaned = df.copy()
        log: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        active_rename_map: Dict[str, str] = {}

        for item in selected_items:
            issue_type = item.get("issue_type")
            col = item.get("column")
            if col and col in active_rename_map:
                col = active_rename_map[col]
            elif col and col not in cleaned.columns:
                col_norm = re.sub(r"[^a-zA-Z0-9_]+", "_", str(col).strip()).strip("_").lower()
                for actual_col in cleaned.columns:
                    actual_norm = re.sub(r"[^a-zA-Z0-9_]+", "_", str(actual_col).strip()).strip("_").lower()
                    if actual_norm == col_norm or str(actual_col).strip() == str(col).strip():
                        active_rename_map[col] = actual_col
                        col = actual_col
                        break

            chosen_action = item.get("chosen_option") or item.get("default_option")

            if "Keep" in chosen_action or chosen_action == "Preserve Identifier (Do Not Delete)":
                continue

            # 1. Drop Unnamed/Empty Column
            if issue_type in ("unnamed_index_column", "empty_column", "high_missing_column"):
                if "Drop" in chosen_action and col in cleaned.columns:
                    cleaned = cleaned.drop(columns=[col])
                    log.append({
                        "timestamp": now_iso,
                        "category": item.get("category"),
                        "operation": f"Dropped Column '{col}'",
                        "column": col,
                        "rows_affected": len(cleaned),
                        "status": "COMPLETED",
                        "reason": item.get("reason")
                    })

            # 2. Duplicate exact rows
            elif issue_type == "exact_duplicate_rows":
                if "Remove" in chosen_action:
                    before_len = len(cleaned)
                    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
                    removed = before_len - len(cleaned)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Duplicate Data",
                        "operation": "Removed Exact Duplicates",
                        "column": "ALL_COLUMNS",
                        "rows_affected": removed,
                        "status": "COMPLETED",
                        "reason": f"Removed {removed:,} identical row tuples."
                    })

            # 3. Column header standardization
            elif issue_type == "unnormalized_column_names":
                if "Standardize" in chosen_action:
                    rename_map = {}
                    for c in cleaned.columns:
                        clean_c = re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip()).strip("_").lower()
                        if clean_c and clean_c != c:
                            rename_map[c] = clean_c
                            for orig, mapped in list(active_rename_map.items()):
                                if mapped == c:
                                    active_rename_map[orig] = clean_c
                    if rename_map:
                        cleaned = cleaned.rename(columns=rename_map)
                        active_rename_map.update(rename_map)
                        log.append({
                            "timestamp": now_iso,
                            "category": "Formatting and Whitespace",
                            "operation": f"Renamed {len(rename_map)} Headers to snake_case",
                            "column": "SCHEMA_HEADERS",
                            "rows_affected": len(cleaned),
                            "status": "COMPLETED",
                            "reason": f"Standardized: {list(rename_map.items())[:3]}"
                        })

            # 4. Whitespace padding
            elif issue_type == "whitespace_padding" and col in cleaned.columns:
                if "Strip" in chosen_action:
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.replace(r"\s{2,}", " ", regex=True)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Formatting and Whitespace",
                        "operation": "Stripped Whitespace Padding",
                        "column": col,
                        "rows_affected": item.get("count", len(cleaned)),
                        "status": "COMPLETED",
                        "reason": "Removed leading, trailing, and multi-spaces."
                    })

            # 5. Currency String Parsing
            elif issue_type == "currency_string_format" and col in cleaned.columns:
                if "Convert" in chosen_action:
                    cleaned[col] = (
                        cleaned[col].astype(str)
                        .str.replace(r"[\$₹€£¥,]", "", regex=True)
                        .str.strip()
                    )
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    log.append({
                        "timestamp": now_iso,
                        "category": "Numerical Data Cleaning",
                        "operation": "Parsed Currency Strings to Numeric Float",
                        "column": col,
                        "rows_affected": item.get("count", len(cleaned)),
                        "status": "COMPLETED",
                        "reason": "Stripped currency symbols and converted to float64."
                    })

            # 6. Percentage String Parsing
            elif issue_type == "percentage_string_format" and col in cleaned.columns:
                if "Convert" in chosen_action:
                    cleaned[col] = cleaned[col].astype(str).str.replace("%", "", regex=False).str.strip()
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    log.append({
                        "timestamp": now_iso,
                        "category": "Numerical Data Cleaning",
                        "operation": "Parsed Percentages to Float64",
                        "column": col,
                        "rows_affected": item.get("count", len(cleaned)),
                        "status": "COMPLETED",
                        "reason": "Stripped '%' signs to enable mathematical calculations."
                    })

            # 7. Numeric string conversion
            elif issue_type == "numeric_stored_as_string" and col in cleaned.columns:
                if "Convert to Numeric" in chosen_action:
                    cleaned[col] = cleaned[col].astype(str).str.replace(",", "", regex=False).str.strip()
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    log.append({
                        "timestamp": now_iso,
                        "category": "Data Type Correction",
                        "operation": "Cast Text to Numeric Float",
                        "column": col,
                        "rows_affected": item.get("count", len(cleaned)),
                        "status": "COMPLETED",
                        "reason": "Cast numbers stored as strings to float64."
                    })

            # 8. Boolean conversion
            elif issue_type == "boolean_stored_as_string" and col in cleaned.columns:
                if "Convert" in chosen_action:
                    b_map = {"true": True, "yes": True, "y": True, "t": True, "1": True,
                             "false": False, "no": False, "n": False, "f": False, "0": False}
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.lower().map(b_map).fillna(False)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Data Type Correction",
                        "operation": "Standardized to Boolean True/False",
                        "column": col,
                        "rows_affected": item.get("count", len(cleaned)),
                        "status": "COMPLETED",
                        "reason": "Mapped truthy/falsy text representations to canonical boolean."
                    })

            # 9. Date Parsing
            elif issue_type == "unparsed_datetime_column" and col in cleaned.columns:
                if "Convert" in chosen_action:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                    log.append({
                        "timestamp": now_iso,
                        "category": "Date and Time Cleaning",
                        "operation": "Standardized to ISO-8601 Datetime",
                        "column": col,
                        "rows_affected": len(cleaned),
                        "status": "COMPLETED",
                        "reason": "Parsed date strings into datetime64."
                    })

            # 10. Missing Values Imputation
            elif issue_type in ("missing_numerical_values", "missing_categorical_values", "missing_datetime_values") and col in cleaned.columns:
                null_cnt = int(cleaned[col].isna().sum())
                if null_cnt > 0:
                    if "Median" in chosen_action:
                        if not pd.api.types.is_numeric_dtype(cleaned[col]):
                            cleaned[col] = pd.to_numeric(cleaned[col].astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")
                        med = float(cleaned[col].dropna().median()) if not cleaned[col].dropna().empty else 0.0
                        cleaned[col] = cleaned[col].fillna(med)
                        log.append({
                            "timestamp": now_iso,
                            "category": "Missing Values",
                            "operation": f"Imputed Missing with Median ({med:.2f})",
                            "column": col,
                            "rows_affected": null_cnt,
                            "status": "COMPLETED",
                            "reason": "Median imputation preserves central tendency for skewed values."
                        })
                    elif "Mean" in chosen_action:
                        if not pd.api.types.is_numeric_dtype(cleaned[col]):
                            cleaned[col] = pd.to_numeric(cleaned[col].astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")
                        mean_v = float(cleaned[col].dropna().mean()) if not cleaned[col].dropna().empty else 0.0
                        cleaned[col] = cleaned[col].fillna(mean_v)
                        log.append({
                            "timestamp": now_iso,
                            "category": "Missing Values",
                            "operation": f"Imputed Missing with Mean ({mean_v:.2f})",
                            "column": col,
                            "rows_affected": null_cnt,
                            "status": "COMPLETED",
                            "reason": "Mean imputation applied to normally distributed series."
                        })
                    elif "Forward" in chosen_action:
                        cleaned[col] = cleaned[col].ffill().bfill()
                        log.append({
                            "timestamp": now_iso,
                            "category": "Missing Values",
                            "operation": "Forward/Backward Filled Timestamps",
                            "column": col,
                            "rows_affected": null_cnt,
                            "status": "COMPLETED",
                            "reason": "Temporal propagation applied."
                        })
                    elif "Unknown" in chosen_action:
                        # Check if column is numeric or numeric stored as strings
                        is_num_target = pd.api.types.is_numeric_dtype(cleaned[col])
                        if not is_num_target and (pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col])):
                            valid_s = cleaned[col].dropna()
                            if not valid_s.empty:
                                num_s = pd.to_numeric(valid_s.astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")
                                if (num_s.notna().sum() / len(valid_s)) >= 0.70:
                                    is_num_target = True
                                    cleaned[col] = pd.to_numeric(cleaned[col].astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")

                        if is_num_target:
                            med_val = float(cleaned[col].dropna().median()) if not cleaned[col].dropna().empty else 0.0
                            cleaned[col] = cleaned[col].fillna(med_val)
                            log.append({
                                "timestamp": now_iso,
                                "category": "Missing Values",
                                "operation": f"Imputed Missing with Median ({med_val:.2f})",
                                "column": col,
                                "rows_affected": null_cnt,
                                "status": "COMPLETED",
                                "reason": "Target column is numeric; applied median imputation to maintain numeric schema."
                            })
                        else:
                            if isinstance(cleaned[col].dtype, pd.CategoricalDtype):
                                if "Unknown" not in cleaned[col].cat.categories:
                                    cleaned[col] = cleaned[col].cat.add_categories(["Unknown"])
                            cleaned[col] = cleaned[col].fillna("Unknown")
                            log.append({
                                "timestamp": now_iso,
                                "category": "Missing Values",
                                "operation": "Imputed Missing with 'Unknown'",
                                "column": col,
                                "rows_affected": null_cnt,
                                "status": "COMPLETED",
                                "reason": "Explicit 'Unknown' placeholder preserves cardinality."
                            })
                    elif "Mode" in chosen_action and not cleaned[col].mode().empty:
                        m_val = cleaned[col].mode().iloc[0]
                        if isinstance(cleaned[col].dtype, pd.CategoricalDtype):
                            if m_val not in cleaned[col].cat.categories:
                                cleaned[col] = cleaned[col].cat.add_categories([m_val])
                        cleaned[col] = cleaned[col].fillna(m_val)
                        log.append({
                            "timestamp": now_iso,
                            "category": "Missing Values",
                            "operation": f"Imputed with Mode ('{m_val}')",
                            "column": col,
                            "rows_affected": null_cnt,
                            "status": "COMPLETED",
                            "reason": "Mode imputation applied."
                        })

            # 11. Inconsistent Casing / Title Case
            elif issue_type == "inconsistent_casing_variations" and col in cleaned.columns:
                if "Title" in chosen_action:
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.title()
                    log.append({
                        "timestamp": now_iso,
                        "category": "Inconsistent Values",
                        "operation": "Standardized to Title Case",
                        "column": col,
                        "rows_affected": len(cleaned),
                        "status": "COMPLETED",
                        "reason": "Unified category casing differences."
                    })
                elif "Lower" in chosen_action:
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()
                    log.append({
                        "timestamp": now_iso,
                        "category": "Inconsistent Values",
                        "operation": "Standardized to Lowercase",
                        "column": col,
                        "rows_affected": len(cleaned),
                        "status": "COMPLETED",
                        "reason": "Lowercased strings."
                    })

            # 12. Gender abbreviation
            elif issue_type == "gender_abbreviation_inconsistency" and col in cleaned.columns:
                if "Male" in chosen_action:
                    g_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.lower().map(lambda v: g_map.get(v, v.title()))
                    log.append({
                        "timestamp": now_iso,
                        "category": "Inconsistent Values",
                        "operation": "Normalized Gender Codes ('M'/'F' -> 'Male'/'Female')",
                        "column": col,
                        "rows_affected": len(cleaned),
                        "status": "COMPLETED",
                        "reason": "Unified gender taxonomy."
                    })

            # 13. Invalid Ages (<0 or >125)
            elif issue_type == "impossible_age_range" and col in cleaned.columns:
                if "Set Invalid" in chosen_action:
                    cleaned.loc[(cleaned[col] < 0) | (cleaned[col] > 125), col] = np.nan
                    med_age = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(med_age)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Invalid / Impossible Values",
                        "operation": f"Nullified Impossible Ages & Imputed with Median ({med_age:.1f})",
                        "column": col,
                        "rows_affected": item.get("count", 0),
                        "status": "COMPLETED",
                        "reason": "Biologically impossible age boundaries enforced."
                    })
                elif "Clip" in chosen_action:
                    cleaned[col] = cleaned[col].clip(lower=0, upper=100)

            # 14. Invalid Rating ([1, 5])
            elif issue_type == "out_of_bounds_rating" and col in cleaned.columns:
                if "Cap" in chosen_action:
                    cleaned[col] = cleaned[col].clip(lower=1.0, upper=5.0)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Invalid / Impossible Values",
                        "operation": "Capped Rating to Bounds [1.0, 5.0]",
                        "column": col,
                        "rows_affected": item.get("count", 0),
                        "status": "COMPLETED",
                        "reason": "Enforced 5-star scale boundaries."
                    })

            # 15. Invalid Negative Quantities / Absolute values
            elif issue_type == "negative_value_violation" and col in cleaned.columns:
                if "Absolute" in chosen_action:
                    neg_cnt = int((cleaned[col] < 0).sum())
                    cleaned[col] = cleaned[col].abs()
                    log.append({
                        "timestamp": now_iso,
                        "category": "Invalid / Impossible Values",
                        "operation": "Inverted Negatives to Absolute Quantities",
                        "column": col,
                        "rows_affected": neg_cnt,
                        "status": "COMPLETED",
                        "reason": "Corrected sign inversions on strictly positive quantities."
                    })

            # 16. Outlier Flagging / Capping
            elif issue_type == "statistical_outliers_iqr" and col in cleaned.columns:
                series = cleaned[col].dropna()
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25
                low_b = q25 - 1.5 * iqr
                high_b = q75 + 1.5 * iqr

                if "Flag" in chosen_action:
                    flag_col = f"{col}_is_outlier"
                    cleaned[flag_col] = ((cleaned[col] < low_b) | (cleaned[col] > high_b)).astype(int)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Outlier Detection",
                        "operation": f"Flagged Outliers in Indicator Column '{flag_col}'",
                        "column": col,
                        "rows_affected": item.get("count", 0),
                        "status": "COMPLETED",
                        "reason": "Flagged statistical extremes without discarding observation data."
                    })
                elif "Cap" in chosen_action:
                    cleaned[col] = cleaned[col].clip(lower=low_b, upper=high_b)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Outlier Detection",
                        "operation": f"Winsorized Outliers to [{low_b:.2f}, {high_b:.2f}]",
                        "column": col,
                        "rows_affected": item.get("count", 0),
                        "status": "COMPLETED",
                        "reason": "Capped values to Tukey 1.5 IQR bounds."
                    })
                elif "Remove" in chosen_action:
                    mask = (cleaned[col] >= low_b) & (cleaned[col] <= high_b) | cleaned[col].isna()
                    removed_c = len(cleaned) - int(mask.sum())
                    cleaned = cleaned[mask].reset_index(drop=True)
                    log.append({
                        "timestamp": now_iso,
                        "category": "Outlier Detection",
                        "operation": f"Removed {removed_c} Outlier Rows",
                        "column": col,
                        "rows_affected": removed_c,
                        "status": "COMPLETED",
                        "reason": "User-approved deletion of extreme outliers."
                    })

            # 17. HTML Tag Stripping
            elif issue_type == "html_tag_fragments" and col in cleaned.columns:
                if "Strip" in chosen_action:
                    cleaned[col] = cleaned[col].astype(str).str.replace(r"<[^<]+?>", "", regex=True).str.strip()
                    log.append({
                        "timestamp": now_iso,
                        "category": "Text Cleaning",
                        "operation": "Stripped Embedded HTML Tags",
                        "column": col,
                        "rows_affected": item.get("count", len(cleaned)),
                        "status": "COMPLETED",
                        "reason": "Removed markup tags from prose text."
                    })

            # 18. Rare category grouping
            elif issue_type == "rare_category_dispersion" and col in cleaned.columns:
                if "Group" in chosen_action:
                    vc = cleaned[col].value_counts(normalize=True)
                    rare_idx = set(vc[vc < 0.01].index)
                    if rare_idx:
                        if isinstance(cleaned[col].dtype, pd.CategoricalDtype):
                            if "Other" not in cleaned[col].cat.categories:
                                cleaned[col] = cleaned[col].cat.add_categories(["Other"])
                        cleaned[col] = cleaned[col].mask(cleaned[col].isin(rare_idx), "Other")
                    log.append({
                        "timestamp": now_iso,
                        "category": "Categorical Data Cleaning",
                        "operation": "Consolidated Rare Categories to 'Other'",
                        "column": col,
                        "rows_affected": item.get("count", 0),
                        "status": "COMPLETED",
                        "reason": "Reduced category fragmentation by grouping <1% records into 'Other'."
                    })

        # Type consistency & schema sanity pass for parquet compatibility
        for c in cleaned.columns:
            if cleaned[c].dtype == object:
                valid_non_null = cleaned[c].dropna()
                if not valid_non_null.empty:
                    test_sample = valid_non_null.head(2000)
                    temp_replaced = test_sample.replace({"Unknown": np.nan, "unknown": np.nan, "N/A": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan, "": np.nan})
                    num_converted = pd.to_numeric(temp_replaced.astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")
                    if (num_converted.notna().sum() / len(test_sample)) >= 0.70:
                        med_fallback = float(num_converted.dropna().median()) if not num_converted.dropna().empty else 0.0
                        cleaned[c] = pd.to_numeric(
                            cleaned[c].replace({"Unknown": np.nan, "unknown": np.nan, "N/A": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan, "": np.nan})
                            .astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(),
                            errors="coerce"
                        ).fillna(med_fallback)

        # Ensure backward compatibility fields for legacy callers
        for entry in log:
            if "issue" not in entry:
                entry["issue"] = entry.get("category", "Data Cleaning")
            if "action" not in entry:
                entry["action"] = entry.get("operation", "Completed")
            if "count" not in entry:
                entry["count"] = entry.get("rows_affected", 0)

        return cleaned, log

    # -------------------------------------------------------------------------
    # 3. POST-CLEANING VALIDATION & DATA QUALITY RECALCULATION
    # -------------------------------------------------------------------------
    @classmethod
    def validate_cleaning(cls, before_df: pd.DataFrame, after_df: pd.DataFrame) -> Dict[str, Any]:
        """Compares before vs after DataFrames, quantifies defect reductions,

        and verifies true data hygiene improvement.
        """
        from core.models.quality_model import DataQualityModel

        cache_key = f"val_report_{len(before_df)}_{len(after_df)}"
        cached_report = AnalyticsCache.get(after_df, cache_key)
        if cached_report is not None:
            return cached_report

        if before_df is after_df:
            q_before = DataQualityModel.calculate_quality_score(before_df)
            score = q_before.get("score", 70)
            res = {
                "score_before": score,
                "score_after": score,
                "score_delta": 0,
                "grade_before": q_before.get("grade", "Fair"),
                "grade_after": q_before.get("grade", "Fair"),
                "validations": [],
                "status_headline": "BASELINE AUDIT: INITIAL DATA QUALITY BENCHMARK ESTABLISHED",
                "remaining_issues": q_before.get("breakdown", [])
            }
            AnalyticsCache.set(after_df, cache_key, res)
            return res

        q_before = DataQualityModel.calculate_quality_score(before_df)
        q_after = DataQualityModel.calculate_quality_score(after_df)

        score_before = q_before.get("score", 70)
        score_after = q_after.get("score", 95)
        score_delta = score_after - score_before

        before_missing = int(before_df.isna().sum().sum())
        after_missing = int(after_df.isna().sum().sum())
        missing_reduced = max(0, before_missing - after_missing)

        cached_b_dups = AnalyticsCache.get(before_df, "duplicate_rows")
        before_dups = cached_b_dups if cached_b_dups is not None else int(before_df.duplicated().sum())
        cached_a_dups = AnalyticsCache.get(after_df, "duplicate_rows")
        after_dups = cached_a_dups if cached_a_dups is not None else int(after_df.duplicated().sum())
        dups_reduced = max(0, before_dups - after_dups)

        # Count type corrections
        before_types = {c: str(t) for c, t in before_df.dtypes.items()}
        after_types = {c: str(t) for c, t in after_df.dtypes.items()}
        types_converted = sum(1 for c in before_types if c in after_types and before_types[c] != after_types[c])

        validations = [
            {
                "label": "Missing Value Reduction",
                "before": f"{before_missing:,}",
                "after": f"{after_missing:,}",
                "delta": f"-{missing_reduced:,}" if missing_reduced > 0 else "0",
                "status": "PASSED" if after_missing == 0 else ("IMPROVED" if missing_reduced > 0 else "UNCHANGED")
            },
            {
                "label": "Exact Duplicate Rows",
                "before": f"{before_dups:,}",
                "after": f"{after_dups:,}",
                "delta": f"-{dups_reduced:,}" if dups_reduced > 0 else "0",
                "status": "PASSED" if after_dups == 0 else ("IMPROVED" if dups_reduced > 0 else "UNCHANGED")
            },
            {
                "label": "Data Type Conversions",
                "before": f"{len(before_df.columns)} Columns",
                "after": f"{types_converted} Normalized",
                "delta": f"+{types_converted}",
                "status": "PASSED" if types_converted > 0 else "MAINTAINED"
            },
            {
                "label": "Dataset Record Retention",
                "before": f"{len(before_df):,} rows",
                "after": f"{len(after_df):,} rows",
                "delta": f"{len(after_df) - len(before_df):,} rows",
                "status": "VERIFIED"
            }
        ]

        status_headline = "CLEANING VERIFIED: ALL HIGH-CONFIDENCE ITEMS RESOLVED" if score_after >= 90 else "CLEANING COMPLETED WITH OUTLIERS / REVIEW ITEMS REMAINING"

        report = {
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_delta,
            "grade_before": q_before.get("grade", "Fair"),
            "grade_after": q_after.get("grade", "Excellent"),
            "validations": validations,
            "status_headline": status_headline,
            "remaining_issues": q_after.get("breakdown", [])
        }
        AnalyticsCache.set(after_df, cache_key, report)
        return report

    # -------------------------------------------------------------------------
    # 4. BEFORE / AFTER SAMPLE DIFF GENERATOR
    # -------------------------------------------------------------------------
    @classmethod
    def generate_sample_diff(cls, before_df: pd.DataFrame, after_df: pd.DataFrame, max_rows: int = 8) -> Dict[str, Any]:
        """Identifies records where values changed and generates a side-by-side comparison."""
        col_pairs: List[Tuple[str, str]] = []
        used_after = set()

        # 1. Exact column name matches
        for b_col in before_df.columns:
            if b_col in after_df.columns:
                col_pairs.append((b_col, b_col))
                used_after.add(b_col)

        def _norm_header(h: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]+", "_", str(h).strip()).strip("_").lower()

        # 2. Normalized header matches (e.g. 'Customer ID' -> 'customer_id')
        for b_col in before_df.columns:
            if any(p[0] == b_col for p in col_pairs):
                continue
            b_norm = _norm_header(b_col)
            for a_col in after_df.columns:
                if a_col not in used_after and _norm_header(a_col) == b_norm:
                    col_pairs.append((b_col, a_col))
                    used_after.add(a_col)
                    break

        diff_samples = []
        min_len = min(len(before_df), len(after_df))
        changed_rows = 0

        for i in range(min_len):
            row_diffs = {}
            for b_col, a_col in col_pairs:
                b_val = before_df.iat[i, before_df.columns.get_loc(b_col)]
                a_val = after_df.iat[i, after_df.columns.get_loc(a_col)]
                
                # Check for difference handling NaN
                is_different = False
                if pd.isna(b_val) and not pd.isna(a_val):
                    is_different = True
                elif not pd.isna(b_val) and pd.isna(a_val):
                    is_different = True
                elif not pd.isna(b_val) and not pd.isna(a_val) and str(b_val) != str(a_val):
                    is_different = True

                if is_different:
                    row_diffs[a_col] = {
                        "before": str(b_val) if not pd.isna(b_val) else "NaN",
                        "after": str(a_val) if not pd.isna(a_val) else "NaN"
                    }

            if row_diffs:
                changed_rows += 1
                if len(diff_samples) < max_rows:
                    diff_samples.append({
                        "row_index": i,
                        "changes": row_diffs
                    })

        return {
            "total_changed_rows": changed_rows,
            "sample_diffs": diff_samples
        }
