import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from core.cleaning.orchestrator import AICleaningOrchestrator

class DataCleanerModel:
    """Intelligent data cleaning engine with recommendations, outlier management,
    audit logging, non-destructive transformations, and AI Orchestrator integration.
    """

    @classmethod
    def scan_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Scans dataset using AICleaningOrchestrator across all 12 cleaning categories."""
        return AICleaningOrchestrator.scan_dataset(df)

    @classmethod
    def execute_plan(
        cls, df: pd.DataFrame, selected_items: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Executes selected AI cleaning plan items non-destructively."""
        return AICleaningOrchestrator.execute_plan(df, selected_items)

    @classmethod
    def validate_cleaning(
        cls, before_df: pd.DataFrame, after_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculates before-and-after quality scores and validates defect reduction."""
        return AICleaningOrchestrator.validate_cleaning(before_df, after_df)

    @classmethod
    def generate_sample_diff(
        cls, before_df: pd.DataFrame, after_df: pd.DataFrame, max_rows: int = 8
    ) -> Dict[str, Any]:
        """Generates side-by-side row differences for altered cells."""
        return AICleaningOrchestrator.generate_sample_diff(before_df, after_df, max_rows)


    @classmethod
    def recommend_cleaning_plan(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Scans dataset across all 12 cleaning categories and returns structured cleaning plan items.
        Maintains backward compatibility with legacy issue/recommended_action keys.
        """
        scan_res = AICleaningOrchestrator.scan_dataset(df)
        return scan_res.get("plan_items", [])


    @classmethod
    def apply_cleaning_actions(cls, df: pd.DataFrame, actions: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Applies chosen cleaning actions on a fresh copy of the DataFrame

        and returns (cleaned_df, cleaning_log).
        """
        cleaned = df.copy()
        log = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        orchestrator_items = [a for a in actions if "issue_type" in a or "default_option" in a or "chosen_option" in a]
        legacy_items = [a for a in actions if a not in orchestrator_items]

        if orchestrator_items:
            cleaned, orch_log = AICleaningOrchestrator.execute_plan(cleaned, orchestrator_items)
            log.extend(orch_log)

        for action_spec in legacy_items:
            action_type = action_spec.get("action")
            column = action_spec.get("column", "All")

            if action_type == "remove_duplicates":
                before_count = len(cleaned)
                cleaned = cleaned.drop_duplicates().reset_index(drop=True)
                removed = before_count - len(cleaned)
                if removed > 0:
                    log.append({
                        "issue": "Duplicates",
                        "column": "All",
                        "count": removed,
                        "action": "Removed"
                    })

            elif action_type == "impute_median" and column in cleaned.columns:
                null_count = int(cleaned[column].isna().sum())
                if null_count > 0:
                    median_val = cleaned[column].median()
                    cleaned[column] = cleaned[column].fillna(median_val)
                    log.append({
                        "issue": "Missing values",
                        "column": column,
                        "count": null_count,
                        "action": f"Imputed with Median ({median_val})"
                    })

            elif action_type == "impute_mean" and column in cleaned.columns:
                null_count = int(cleaned[column].isna().sum())
                if null_count > 0:
                    mean_val = round(float(cleaned[column].mean()), 2)
                    cleaned[column] = cleaned[column].fillna(mean_val)
                    log.append({
                        "issue": "Missing values",
                        "column": column,
                        "count": null_count,
                        "action": f"Imputed with Mean ({mean_val})"
                    })

            elif action_type == "impute_constant" and column in cleaned.columns:
                null_count = int(cleaned[column].isna().sum())
                const_val = action_spec.get("value", 0)
                if null_count > 0:
                    if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
                        if const_val not in cleaned[column].cat.categories:
                            cleaned[column] = cleaned[column].cat.add_categories([const_val])
                    cleaned[column] = cleaned[column].fillna(const_val)
                    log.append({
                        "issue": "Missing values",
                        "column": column,
                        "count": null_count,
                        "action": f"Imputed with Constant ({const_val})"
                    })

            elif action_type == "impute_mode" and column in cleaned.columns:
                null_count = int(cleaned[column].isna().sum())
                if null_count > 0 and not cleaned[column].mode().empty:
                    mode_val = cleaned[column].mode().iloc[0]
                    if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
                        if mode_val not in cleaned[column].cat.categories:
                            cleaned[column] = cleaned[column].cat.add_categories([mode_val])
                    cleaned[column] = cleaned[column].fillna(mode_val)
                    log.append({
                        "issue": "Missing values",
                        "column": column,
                        "count": null_count,
                        "action": f"Imputed with Mode ('{mode_val}')"
                    })

            elif action_type == "impute_unknown" and column in cleaned.columns:
                null_count = int(cleaned[column].isna().sum())
                if null_count > 0:
                    is_num_col = pd.api.types.is_numeric_dtype(cleaned[column])
                    if not is_num_col and (pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(cleaned[column])):
                        valid_s = cleaned[column].dropna()
                        if not valid_s.empty:
                            num_test = pd.to_numeric(valid_s.astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")
                            if (num_test.notna().sum() / len(valid_s)) >= 0.70:
                                is_num_col = True
                                cleaned[column] = pd.to_numeric(cleaned[column].astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")

                    if is_num_col:
                        med_val = float(cleaned[column].dropna().median()) if not cleaned[column].dropna().empty else 0.0
                        cleaned[column] = cleaned[column].fillna(med_val)
                        log.append({
                            "issue": "Missing values",
                            "column": column,
                            "count": null_count,
                            "action": f"Imputed with Median ({med_val:.2f})"
                        })
                    else:
                        if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
                            if "Unknown" not in cleaned[column].cat.categories:
                                cleaned[column] = cleaned[column].cat.add_categories(["Unknown"])
                        cleaned[column] = cleaned[column].fillna("Unknown")
                        log.append({
                            "issue": "Missing values",
                            "column": column,
                            "count": null_count,
                            "action": "Imputed with 'Unknown'"
                        })

            elif action_type == "forward_fill" and column in cleaned.columns:
                null_count = int(cleaned[column].isna().sum())
                if null_count > 0:
                    cleaned[column] = cleaned[column].ffill().bfill()
                    log.append({
                        "issue": "Missing dates",
                        "column": column,
                        "count": null_count,
                        "action": "Forward/Backward filled"
                    })

            elif action_type == "standardize_title" and column in cleaned.columns:
                cleaned[column] = cleaned[column].astype(str).str.strip().str.title()
                log.append({
                    "issue": "Category inconsistency",
                    "column": column,
                    "count": len(cleaned),
                    "action": "Standardized to Title Case"
                })

            elif action_type == "standardize_lower" and column in cleaned.columns:
                cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()
                log.append({
                    "issue": "Category inconsistency",
                    "column": column,
                    "count": len(cleaned),
                    "action": "Standardized to Lowercase"
                })

            elif action_type == "cap_outliers_iqr" and column in cleaned.columns:
                series = cleaned[column].dropna()
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_limit = q25 - 1.5 * iqr
                upper_limit = q75 + 1.5 * iqr
                capped_count = int(((cleaned[column] < lower_limit) | (cleaned[column] > upper_limit)).sum())
                if capped_count > 0:
                    cleaned[column] = cleaned[column].clip(lower=lower_limit, upper=upper_limit)
                    log.append({
                        "issue": "Outliers (IQR)",
                        "column": column,
                        "count": capped_count,
                        "action": f"Capped to [{lower_limit:.2f}, {upper_limit:.2f}]"
                    })

            elif action_type == "flag_outliers" and column in cleaned.columns:
                series = cleaned[column].dropna()
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_limit = q25 - 1.5 * iqr
                upper_limit = q75 + 1.5 * iqr
                flag_col = f"{column}_is_outlier"
                cleaned[flag_col] = ((cleaned[column] < lower_limit) | (cleaned[column] > upper_limit)).astype(int)
                flagged_count = int(cleaned[flag_col].sum())
                log.append({
                    "issue": "Outliers (IQR)",
                    "column": column,
                    "count": flagged_count,
                    "action": f"Flagged in '{flag_col}'"
                })

            elif action_type == "remove_outliers" and column in cleaned.columns:
                series = cleaned[column].dropna()
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_limit = q25 - 1.5 * iqr
                upper_limit = q75 + 1.5 * iqr
                mask = (cleaned[column] >= lower_limit) & (cleaned[column] <= upper_limit) | cleaned[column].isna()
                removed_count = len(cleaned) - int(mask.sum())
                if removed_count > 0:
                    cleaned = cleaned[mask].reset_index(drop=True)
                    log.append({
                        "issue": "Outliers (IQR)",
                        "column": column,
                        "count": removed_count,
                        "action": "Rows Removed"
                    })

        # Parquet type consistency pass
        for c in cleaned.columns:
            if cleaned[c].dtype == object:
                valid_non_null = cleaned[c].dropna()
                if not valid_non_null.empty:
                    temp_replaced = valid_non_null.replace({"Unknown": np.nan, "unknown": np.nan, "N/A": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan, "": np.nan})
                    num_converted = pd.to_numeric(temp_replaced.astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(), errors="coerce")
                    if (num_converted.notna().sum() / len(valid_non_null)) >= 0.70:
                        med_fallback = float(num_converted.dropna().median()) if not num_converted.dropna().empty else 0.0
                        cleaned[c] = pd.to_numeric(
                            cleaned[c].replace({"Unknown": np.nan, "unknown": np.nan, "N/A": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan, "": np.nan})
                            .astype(str).str.replace(r"[\$₹€£¥,]", "", regex=True).str.strip(),
                            errors="coerce"
                        ).fillna(med_fallback)

        return cleaned, log

    @classmethod
    def auto_clean(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Automatically applies high-confidence, non-destructive cleaning operations."""
        return AICleaningOrchestrator.apply_auto_clean(df)

