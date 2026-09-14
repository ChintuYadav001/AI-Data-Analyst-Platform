"""
Analytical Goal & Target Inference Engine for Quantum Analyst AI EDA Engine.
Infers business objectives from schema semantics, detects target candidates,
and prioritizes analytical variables.
"""
from __future__ import annotations

import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class EDAGoalEngine:
    """Infers analytical goals, resolves target variables, and prioritizes attributes."""

    @classmethod
    def infer_goal(
        cls,
        profile: Dict[str, Any],
        user_goal: Optional[str] = None,
        user_target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Infers or validates the analytical objective."""
        if user_goal and user_goal.strip() and user_goal.strip().lower() != "general business intelligence":
            return {
                "goal": user_goal.strip(),
                "is_inferred": False,
                "rationale": "Directly specified by analyst in session context."
            }

        target_candidates = profile.get("target_candidates", [])
        domain = profile.get("domain", "General Enterprise Tabular")

        target_col = user_target or (target_candidates[0]["column"] if target_candidates else None)
        task_type = target_candidates[0]["task_type"] if target_candidates else "regression"

        if target_col:
            clean_col = target_col.replace("_", " ").title()
            if task_type == "classification":
                inferred = f"Identify behavioral factors, segment variations, and key drivers associated with {clean_col}."
                rationale = f"Detected classification target '{target_col}' with strong semantic outcome role."
            else:
                inferred = f"Evaluate performance momentum, cohort variations, and factors impacting {clean_col}."
                rationale = f"Detected continuous business metric '{target_col}' as primary target."
        else:
            if "E-Commerce" in domain:
                inferred = "Analyze product category velocity, revenue concentration, and purchase volume."
                rationale = "E-Commerce domain detected without explicit target."
            elif "Financial" in domain:
                inferred = "Evaluate transaction volumes, risk boundaries, and balance anomalies."
                rationale = "Financial domain detected without explicit target."
            elif "Marketing" in domain:
                inferred = "Assess campaign conversion rates, channel ROI, and customer acquisition efficiency."
                rationale = "Marketing domain detected without explicit target."
            elif "HR" in domain:
                inferred = "Examine employee retention, compensation parity, and departmental performance."
                rationale = "HR domain detected without explicit target."
            elif "Healthcare" in domain:
                inferred = "Analyze clinical indicators, patient risk cohorts, and outcome correlations."
                rationale = "Healthcare domain detected without explicit target."
            else:
                inferred = "Multi-variable exploratory analysis of distributions, relationships, and key segment performance."
                rationale = "General tabular dataset with multi-dimensional exploratory focus."

        return {
            "goal": inferred,
            "is_inferred": True,
            "rationale": rationale
        }

    @classmethod
    def resolve_target(
        cls,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        user_target: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolves target variable and determines if it is continuous or discrete."""
        if user_target and user_target in df.columns:
            series = df[user_target].dropna()
            is_num = pd.api.types.is_numeric_dtype(series)
            n_unique = int(series.nunique())
            task_type = "classification" if (not is_num or n_unique <= 8) else "regression"
            return {
                "column": user_target,
                "task_type": task_type,
                "is_user_selected": True,
                "unique_values": n_unique,
                "is_numeric": bool(is_num)
            }

        target_candidates = profile.get("target_candidates", [])
        if target_candidates:
            top = target_candidates[0]
            col = top["column"]
            if col in df.columns:
                series = df[col].dropna()
                is_num = pd.api.types.is_numeric_dtype(series)
                return {
                    "column": col,
                    "task_type": top["task_type"],
                    "is_user_selected": False,
                    "unique_values": int(series.nunique()),
                    "is_numeric": bool(is_num)
                }

        return None

    @classmethod
    def select_priority_columns(
        cls,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        target_col: Optional[str] = None,
        max_cols: int = 12
    ) -> List[str]:
        """Selects priority columns excluding identifiers, constants, and high-missing attributes."""
        identifiers = set(profile.get("columns_by_type", {}).get("identifier", []))
        constants = set(profile.get("flags", {}).get("constant", []))
        near_constants = set(profile.get("flags", {}).get("near_constant", []))

        excluded = identifiers | constants | near_constants
        priority: List[str] = []

        if target_col and target_col in df.columns and target_col not in excluded:
            priority.append(target_col)

        # Datetime column first
        date_cols = profile.get("columns_by_type", {}).get("datetime", [])
        for dc in date_cols:
            if dc in df.columns and dc not in priority and dc not in excluded:
                priority.append(dc)
                break

        # Numerical columns with high variance/information
        num_cols = profile.get("columns_by_type", {}).get("numerical", [])
        for nc in num_cols:
            if nc in df.columns and nc not in priority and nc not in excluded:
                if len(priority) < max_cols:
                    priority.append(nc)

        # Informative categorical columns (cardinality 2 to 30)
        cat_cols = profile.get("columns_by_type", {}).get("categorical", [])
        for cc in cat_cols:
            if cc in df.columns and cc not in priority and cc not in excluded:
                n_uniq = df[cc].nunique()
                if 2 <= n_uniq <= 30 and len(priority) < max_cols:
                    priority.append(cc)

        # Boolean columns
        bool_cols = profile.get("columns_by_type", {}).get("boolean", [])
        for bc in bool_cols:
            if bc in df.columns and bc not in priority and bc not in excluded and len(priority) < max_cols:
                priority.append(bc)

        return priority
