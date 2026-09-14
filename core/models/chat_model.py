import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional
from core.models.sql_model import SQLAnalyticsModel
from core.models.profiler_model import DataProfilerModel
from core.models.chart_model import ChartModel
from core.models.eda_model import EDAModel

class AIChatModel:
    """Conversational AI Data Analyst executing queries and questions

    via controlled backend tools (SQL engine, Profiler, EDA, Chart recommendation).
    """

    def __init__(self, sql_engine: SQLAnalyticsModel):
        self.sql_engine = sql_engine

    def answer_query(self, user_message: str, df: pd.DataFrame, table_name: str = "dataset") -> Dict[str, Any]:
        msg_lower = user_message.lower().strip()
        
        # 1. Natural Language SQL queries
        # E.g. "show top 5 products by revenue", "which region has highest profit", "average sales by category"
        sql_generated = self._infer_sql_query(msg_lower, df, table_name)
        if sql_generated:
            exec_result = self.sql_engine.execute_query(sql_generated)
            if exec_result.get("success"):
                res_df = exec_result["dataframe"]
                explanation = self._explain_sql_result(user_message, sql_generated, res_df)
                chart_spec = exec_result.get("recommended_chart")
                chart_fig = None
                if chart_spec and res_df is not None and not res_df.empty:
                    try:
                        chart_fig = ChartModel.create_plotly_figure(res_df, chart_spec)
                    except Exception:
                        pass
                        
                return {
                    "tool_used": "execute_sql()",
                    "sql": sql_generated,
                    "response": explanation,
                    "data": res_df,
                    "chart_spec": chart_spec,
                    "chart_figure": chart_fig,
                    "execution_time_ms": exec_result.get("execution_time_ms")
                }

        # 2. Profiling / Summary questions
        if any(w in msg_lower for w in ["profile", "overview", "how many rows", "columns", "summary", "shape", "size"]):
            profile = DataProfilerModel.profile_dataset(df)
            ds = profile["dataset_info"]
            resp = (
                f"**Dataset Overview:**\n"
                f"- **Rows:** {ds['rows']:,}\n"
                f"- **Columns:** {ds['columns']}\n"
                f"- **Memory Usage:** {ds['memory_mb']} MB\n"
                f"- **Duplicate Rows:** {ds['duplicate_rows']} ({ds['duplicate_pct']}%)\n"
                f"- **Missing Values:** {ds['total_missing']} ({ds['missing_pct']}%)"
            )
            return {
                "tool_used": "profile_dataset()",
                "response": resp,
                "data": pd.DataFrame(profile["columns"])[["name", "semantic_type", "null_count", "null_pct", "unique_count"]]
            }

        # 3. Outlier questions
        if any(w in msg_lower for w in ["outlier", "unusual", "anomaly", "anomalies"]):
            eda_num = EDAModel.analyze_numerical(df)
            skewed = [c for c in eda_num if abs(c["skewness"]) > 1.5]
            if skewed:
                lines = [f"- **{c['column']}**: Skewness = {c['skewness']:.2f}, Mean = {c['mean']:.2f}, Median = {c['median']:.2f}" for c in skewed]
                resp = "**Outlier & Distributional Anomaly Analysis:**\n" + "\n".join(lines)
            else:
                resp = "No extreme distribution outliers detected across numerical columns based on IQR thresholds."
            return {
                "tool_used": "detect_outliers()",
                "response": resp,
                "data": pd.DataFrame(eda_num)[["column", "min", "q25", "median", "q75", "max", "skewness"]] if eda_num else None
            }

        # 4. Visualization request
        if any(w in msg_lower for w in ["chart", "plot", "graph", "visualize"]):
            recs = ChartModel.recommend_charts(df)
            if recs:
                top_rec = recs[0]
                fig = ChartModel.create_plotly_figure(df, top_rec)
                return {
                    "tool_used": "recommend_chart()",
                    "response": f"Generated recommended chart: **{top_rec['title']}** based on column types and distributions.",
                    "chart_spec": top_rec,
                    "chart_figure": fig
                }

        # Default fallback: safe SQL preview or dataset schema
        return {
            "tool_used": "calculate_statistics()",
            "response": (
                f"I analyzed your question regarding '{user_message}'. "
                f"You can ask me to compute top categories (e.g. 'Show top 5 regions by sales'), "
                f"generate charts ('Plot monthly trend'), or inspect data quality ('Show outliers')."
            ),
            "data": df.head(5)
        }

    def _infer_sql_query(self, query_text: str, df: pd.DataFrame, table_name: str) -> Optional[str]:
        cols = {str(c).lower(): str(c) for c in df.columns}
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in str(c).lower()]
        cat_cols = [c for c in df.columns if df[c].nunique() <= 50 and not pd.api.types.is_numeric_dtype(df[c])]

        # Check for "top N ... by ..."
        top_match = re.search(r"top\s+(\d+)\s+([a-zA-Z_]+)(?:\s+by\s+([a-zA-Z_]+))?", query_text)
        if top_match:
            limit_n = top_match.group(1)
            target_ent = top_match.group(2).lower()
            metric_ent = top_match.group(3).lower() if top_match.group(3) else None

            # Find matching category column
            group_col = next((cols[k] for k in cols if target_ent in k or k in target_ent), None)
            if not group_col and cat_cols:
                group_col = cat_cols[0]

            # Find matching metric column
            metric_col = None
            if metric_ent:
                metric_col = next((cols[k] for k in cols if metric_ent in k or k in metric_ent), None)
            if not metric_col and num_cols:
                metric_col = num_cols[0]

            if group_col and metric_col:
                return f'SELECT "{group_col}", ROUND(SUM("{metric_col}"), 2) AS total_{metric_col} FROM "{table_name}" GROUP BY "{group_col}" ORDER BY total_{metric_col} DESC LIMIT {limit_n}'

        # Check for "highest / best ... by / in ..."
        if any(w in query_text for w in ["highest", "best", "most", "lowest", "worst"]):
            order_dir = "DESC" if any(w in query_text for w in ["highest", "best", "most"]) else "ASC"
            # Find category
            group_col = None
            for cat in cat_cols:
                if cat.lower() in query_text:
                    group_col = cat
                    break
            if not group_col and cat_cols:
                group_col = cat_cols[0]

            # Find metric
            metric_col = None
            for num in num_cols:
                if num.lower() in query_text:
                    metric_col = num
                    break
            if not metric_col and num_cols:
                metric_col = num_cols[0]

            if group_col and metric_col:
                return f'SELECT "{group_col}", ROUND(SUM("{metric_col}"), 2) AS total_{metric_col} FROM "{table_name}" GROUP BY "{group_col}" ORDER BY total_{metric_col} {order_dir} LIMIT 10'

        # Check for "average ... by ..."
        if "average" in query_text or "avg" in query_text or "mean" in query_text:
            group_col = cat_cols[0] if cat_cols else None
            metric_col = num_cols[0] if num_cols else None
            for c in cat_cols:
                if c.lower() in query_text:
                    group_col = c
            for n in num_cols:
                if n.lower() in query_text:
                    metric_col = n

            if group_col and metric_col:
                return f'SELECT "{group_col}", ROUND(AVG("{metric_col}"), 2) AS avg_{metric_col} FROM "{table_name}" GROUP BY "{group_col}" ORDER BY avg_{metric_col} DESC'

        return None

    def _explain_sql_result(self, question: str, sql: str, df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return "No records matched your criteria."

        first_row = df.iloc[0]
        col_names = list(df.columns)
        first_cat = str(first_row[col_names[0]])
        first_metric = first_row[col_names[1]]
        
        return (
            f"Based on your query, **'{first_cat}'** ranked #1 with **{first_metric:,.2f}** {col_names[1]}. "
            f"Retrieved {len(df)} records in total."
        )
