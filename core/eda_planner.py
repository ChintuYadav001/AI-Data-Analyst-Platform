from typing import Dict, Any, List, Optional

class EDAPlanner:
    """Intelligent EDA planner that calibrates analysis modules to user goals,

    dataset scale, semantic types, and target variables—avoiding blind full-matrix scans.
    """

    @classmethod
    def create_eda_plan(
        cls,
        profile: Dict[str, Any],
        goal: str = "General Business Intelligence",
        target_col: Optional[str] = None,
        quality_score: float = 100.0
    ) -> Dict[str, Any]:
        """Classifies analyses into HIGH, MEDIUM, OPTIONAL, and NOT_APPLICABLE tiers."""
        total_rows = profile.get("rows", 0)
        num_cols = profile.get("numeric_columns", [])
        cat_cols = profile.get("categorical_columns", [])
        date_cols = profile.get("datetime_columns", [])
        id_cols = profile.get("identifier_columns", [])

        # Filter out IDs from statistical candidates
        clean_num_cols = [c for c in num_cols if c not in id_cols]
        clean_cat_cols = [c for c in cat_cols if c not in id_cols]

        # Auto-detect target column if not explicitly provided
        if not target_col and clean_num_cols:
            goal_lower = goal.lower()
            if any(k in goal_lower for k in ["revenue", "sales", "growth", "margin"]):
                target_col = next((c for c in clean_num_cols if any(k in c.lower() for k in ["revenue", "sales", "turnover", "total"])), None)
            elif any(k in goal_lower for k in ["profit", "loss"]):
                target_col = next((c for c in clean_num_cols if any(k in c.lower() for k in ["profit", "net", "margin"])), None)
            elif any(k in goal_lower for k in ["churn", "attrition", "retention"]):
                target_col = next((c for c in clean_num_cols if any(k in c.lower() for k in ["churn", "status", "retained"])), None)

            if not target_col and clean_num_cols:
                target_col = clean_num_cols[0]

        high_priority = []
        medium_priority = []
        optional = []
        not_applicable = []

        # 1. Target & Core Distributions
        if target_col:
            high_priority.append({
                "module": "target_distribution",
                "target_col": target_col,
                "title": f"Distribution & Moment Profile of Target '{target_col}'",
                "rationale": f"Primary KPI linked directly to '{goal}'."
            })

        # 2. Chronological Waves
        if date_cols and clean_num_cols:
            primary_date = date_cols[0]
            primary_metric = target_col or clean_num_cols[0]
            high_priority.append({
                "module": "time_series",
                "date_col": primary_date,
                "metric_col": primary_metric,
                "title": f"Chronological Momentum: {primary_metric} over {primary_date}",
                "rationale": "Uncovers temporal seasonality and trajectory."
            })

        # 3. Categorical Segmentation
        if clean_cat_cols and clean_num_cols:
            top_cat = clean_cat_cols[0]
            top_num = target_col or clean_num_cols[0]
            high_priority.append({
                "module": "segmentation",
                "cat_col": top_cat,
                "metric_col": top_num,
                "title": f"Cohort Breakdown: {top_num} by {top_cat}",
                "rationale": "Surfaces high-leverage business segments."
            })

        # 4. Correlation Lattice
        if len(clean_num_cols) >= 2:
            medium_priority.append({
                "module": "correlation",
                "columns": clean_num_cols[:8],
                "title": "Bivariate Covariance & Pearson Lattice",
                "rationale": "Quantifies linear collinearities and trade-offs."
            })

        # 5. Outlier & Anomaly Screening
        if clean_num_cols:
            medium_priority.append({
                "module": "outlier_iqr",
                "columns": clean_num_cols[:5],
                "title": "Tukey IQR Fast Outlier Detection",
                "rationale": "Identifies extreme tails exceeding 1.5x IQR."
            })

        # 6. Deep Machine Learning Anomalies & Driver Analysis
        if total_rows >= 1000 and clean_num_cols:
            optional.append({
                "module": "isolation_forest",
                "columns": clean_num_cols[:6],
                "title": "Multivariate Isolation Forest Anomaly Engine",
                "rationale": "High-dimensional isolation score for multivariate anomalies."
            })
            optional.append({
                "module": "statistical_drivers",
                "target_col": target_col,
                "candidates": [c for c in clean_num_cols if c != target_col][:5],
                "title": f"ANOVA & Spearman Driver Ranking for '{target_col}'",
                "rationale": "Identifies highest statistical influence on primary KPI."
            })

        # 7. Discard Irrelevant Elements
        if id_cols:
            not_applicable.append({
                "module": "identifier_scan",
                "columns": id_cols,
                "rationale": "Identifier tokens excluded from statistical aggregations."
            })

        return {
            "dataset_hash": profile.get("dataset_hash"),
            "version": profile.get("version"),
            "target_variable": target_col,
            "strategic_goal": goal,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "optional": optional,
            "not_applicable": not_applicable,
            "quick_eda_modules": [m["module"] for m in high_priority + medium_priority[:1]],
            "deep_eda_modules": [m["module"] for m in optional + medium_priority[1:]]
        }
