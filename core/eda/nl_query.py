"""
Controlled Natural Language EDA Query Assistant.
Translates analytical inquiries into DuckDB/Pandas aggregate computations
and delivers evidence-backed explanations with zero fabricated numbers.
"""
from __future__ import annotations

import re
import duckdb
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from core.eda.anomalies import EDAAnomalyEngine
from core.eda.statistical_engine import EDAStatisticalEngine
from core.eda.time_series import EDATimeSeriesEngine


class EDANLQueryEngine:
    """Answers natural language data inquiries using deterministic DuckDB/Pandas calculations."""

    @classmethod
    def answer_query(
        cls,
        df: pd.DataFrame,
        query: str,
        target_col: Optional[str] = None,
        goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translates user natural language prompt into executed analytics and structured answer."""
        if df is None or df.empty or not query or not query.strip():
            return {
                "query": query,
                "answer": "Please provide a valid dataset and query.",
                "evidence": "",
                "sql": "",
                "table": None,
                "chart_spec": None
            }

        q = query.lower().strip()
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not any(k in str(c).lower() for k in ["_id", "id_", "identifier", "row_number"])]
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "string", "category"]).columns
            if not any(k in str(c).lower() for k in ["_id", "id_", "identifier", "date", "time", "timestamp"])
        ]

        # 1. Match referenced columns with word boundary and synonyms
        synonyms = {"revenue": ["sales", "amount", "total"], "profit": ["margin", "income"], "churn": ["status", "attrition"]}
        matched_num = []
        for c in num_cols:
            c_low = str(c).lower()
            if re.search(r"\b" + re.escape(c_low) + r"\b", q):
                matched_num.append(c)
        if not matched_num:
            for syn_key, syn_list in synonyms.items():
                if syn_key in q:
                    for s in syn_list:
                        for c in num_cols:
                            if s in str(c).lower() and c not in matched_num:
                                matched_num.append(c)

        matched_cat = []
        for c in cat_cols:
            c_low = str(c).lower()
            plural = c_low[:-1] + "ies" if c_low.endswith("y") else c_low + "s"
            if re.search(r"\b" + re.escape(c_low) + r"\b", q) or re.search(r"\b" + re.escape(plural) + r"\b", q):
                matched_cat.append(c)

        primary_num = matched_num[0] if matched_num else (target_col if target_col in num_cols else (num_cols[0] if num_cols else None))
        primary_cat = matched_cat[0] if matched_cat else (cat_cols[0] if cat_cols else None)

        # 2. Intent Detection
        # A. Outliers / Anomalies
        if any(k in q for k in ["outlier", "anomal", "unusual", "extreme", "spike"]):
            anomalies = EDAAnomalyEngine.detect_anomalies(df, num_cols=[primary_num] if primary_num else None)
            if anomalies:
                top_anom = anomalies[0]
                ans = (
                    f"Anomaly audit on **`{top_anom['column']}`** detected **{top_anom['iqr_outliers_count']:,}** "
                    f"boundary outliers ({top_anom['iqr_outliers_pct']}% of records). "
                    f"These extreme values shift the sample mean by **{top_anom['mean_shift_pct']}%** "
                    f"(from {top_anom['mean_without_outliers']:,.2f} without outliers to {top_anom['original_mean']:,.2f} overall). "
                    f"Classification: *{top_anom['classification']}*."
                )
                return {
                    "query": query,
                    "answer": ans,
                    "evidence": f"IQR bounds: [{top_anom['lower_bound_mild']}, {top_anom['upper_bound_mild']}]. Extreme count: {top_anom['extreme_outliers_count']}.",
                    "sql": f"SELECT * FROM dataset WHERE {top_anom['column']} < {top_anom['lower_bound_mild']} OR {top_anom['column']} > {top_anom['upper_bound_mild']};",
                    "table": pd.DataFrame([top_anom]),
                    "chart_spec": {"type": "box", "y": top_anom["column"], "title": f"Anomaly Boundaries: {top_anom['column']}"}
                }

        # B. Correlation / Relationship
        if any(k in q for k in ["correlat", "relation", "association", "trade-off", "versus", " vs "]):
            cols_to_corr = matched_num if len(matched_num) >= 2 else num_cols[:2]
            if len(cols_to_corr) >= 2:
                c1, c2 = cols_to_corr[0], cols_to_corr[1]
                paired = df[[c1, c2]].dropna()
                r_val = float(paired.corr().iloc[0, 1])
                rel_type = "positive co-movement" if r_val > 0 else "inverse trade-off"
                strength = "strong" if abs(r_val) >= 0.7 else ("moderate" if abs(r_val) >= 0.4 else "weak")
                ans = (
                    f"Correlation between **`{c1}`** and **`{c2}`** is **r = {r_val:.3f}** ({strength} {rel_type}). "
                    f"This indicates that variations in {c1} statistically coincide with {rel_type} in {c2}. "
                    f"CRITICAL: This statistical association does NOT demonstrate that {c1} causes changes in {c2}."
                )
                return {
                    "query": query,
                    "answer": ans,
                    "evidence": f"Evaluated across {len(paired):,} paired observations with non-null values.",
                    "sql": f"SELECT CORR({c1}, {c2}) AS pearson_correlation FROM dataset;",
                    "table": pd.DataFrame([{"feature1": c1, "feature2": c2, "pearson_r": round(r_val, 3), "strength": strength}]),
                    "chart_spec": {"type": "scatter", "x": c1, "y": c2, "title": f"Bivariate Relationship: {c1} vs {c2}"}
                }

        # C. Trend / Temporal Velocity
        if any(k in q for k in ["trend", "time", "month", "year", "growth", "over time", "history"]):
            trends = EDATimeSeriesEngine.analyze_temporal_trends(df, metric_col=primary_num)
            if trends:
                ans = (
                    f"Chronological trajectory for **`{trends['metric_column']}`** across **`{trends['date_column']}`** "
                    f"exhibits **{trends['trend_direction']}** over {trends['span_days']} days ({trends['granularity']} aggregation). "
                    f"Latest period-over-period growth is **{trends['latest_growth_rate_pct']:+.1f}%**. "
                    f"Peak velocity reached **{trends['peak_period']['total']:,.2f}** in period {trends['peak_period']['period']}."
                )
                return {
                    "query": query,
                    "answer": ans,
                    "evidence": f"Peak: {trends['peak_period']['period']} ({trends['peak_period']['total']:,.2f}). Valley: {trends['valley_period']['period']} ({trends['valley_period']['total']:,.2f}).",
                    "sql": f"SELECT DATE_TRUNC('month', {trends['date_column']}) AS period, SUM({trends['metric_column']}) AS total FROM dataset GROUP BY 1 ORDER BY 1;",
                    "table": pd.DataFrame(trends["trend_points"]),
                    "chart_spec": {"type": "line", "x": "period", "y": "total", "title": f"Temporal Momentum: {trends['metric_column']}"}
                }

        # D. Category Ranking / Group Aggregations
        if primary_num and primary_cat:
            is_highest = any(k in q for k in ["top", "highest", "best", "most", "largest", "maximum", "leader"])
            is_lowest = any(k in q for k in ["bottom", "lowest", "worst", "least", "smallest", "minimum", "laggard"])
            order_dir = "ASC" if is_lowest else "DESC"

            try:
                con = duckdb.connect(database=":memory:")
                con.register("dataset", df)
                query_sql = f"""
                    SELECT "{primary_cat}" AS category,
                           COUNT(*) AS record_count,
                           ROUND(SUM("{primary_num}"), 2) AS total_metric,
                           ROUND(AVG("{primary_num}"), 2) AS avg_metric
                    FROM dataset
                    WHERE "{primary_cat}" IS NOT NULL AND "{primary_num}" IS NOT NULL
                    GROUP BY "{primary_cat}"
                    ORDER BY total_metric {order_dir}
                    LIMIT 8
                """
                res_df = con.execute(query_sql).df()
                con.close()

                if not res_df.empty:
                    leader_row = res_df.iloc[0]
                    ans = (
                        f"Ranked breakdown of **`{primary_num}`** across **`{primary_cat}`**: "
                        f"The {'leading' if not is_lowest else 'lowest'} cohort is **`{leader_row['category']}`** "
                        f"with a cumulative total of **{leader_row['total_metric']:,.2f}** "
                        f"(average of {leader_row['avg_metric']:,.2f} across {int(leader_row['record_count']):,} records)."
                    )
                    return {
                        "query": query,
                        "answer": ans,
                        "evidence": f"Computed across {len(df):,} records in active dataframe via analytical DuckDB kernel.",
                        "sql": query_sql.strip(),
                        "table": res_df,
                        "chart_spec": {"type": "bar", "x": "category", "y": "total_metric", "title": f"{primary_num} by {primary_cat}"}
                    }
            except Exception:
                pass

        # E. General Descriptive / Summary Fallback
        if primary_num:
            s = df[primary_num].dropna()
            ans = (
                f"Statistical summary for **`{primary_num}`**: Mean is **{s.mean():,.2f}**, "
                f"Median is **{s.median():,.2f}**, Standard Deviation is **{s.std():,.2f}**, "
                f"ranging from **{s.min():,.2f}** to **{s.max():,.2f}** across {len(s):,} non-null records."
            )
            return {
                "query": query,
                "answer": ans,
                "evidence": f"Skewness: {s.skew():.2f}, Kurtosis: {s.kurt():.2f}.",
                "sql": f"SELECT AVG({primary_num}), MEDIAN({primary_num}), MIN({primary_num}), MAX({primary_num}) FROM dataset;",
                "table": pd.DataFrame([{
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2)
                }]),
                "chart_spec": {"type": "histogram", "x": primary_num, "title": f"Distribution of {primary_num}"}
            }

        return {
            "query": query,
            "answer": f"Analysis complete on dataset ({len(df):,} records, {len(df.columns)} columns). No specific metric could be isolated for query '{query}'.",
            "evidence": "Please specify a column name or analytical question.",
            "sql": "SELECT COUNT(*) FROM dataset;",
            "table": None,
            "chart_spec": None
        }
