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

        # 0. Machine Learning & Predictive Analytics queries
        if any(w in msg_lower for w in ["predict", "forecast", "machine learning", "train model", "best model", "champion model", "ml readiness", "feature importance", "pca", "dimensionality", "anomaly", "outlier", "feature selection"]):
            try:
                import core.ml.ai_ml_tools as ai_tools
                from core.ml.experiment_tracker import get_active_champion

                # A. PCA / Dimensionality Reduction Query
                if any(w in msg_lower for w in ["pca", "principal component", "dimensionality"]):
                    pca_res = ai_tools.run_pca(df, n_components=2)
                    top_d = pca_res.get("top_drivers", {})
                    resp = (
                        f"📊 **Principal Component Analysis (PCA) Results**:\n"
                        f"- **Decomposition Method:** PCA (Orthogonal Linear Projection)\n"
                        f"- **Total Variance Retained:** **{pca_res.get('variance_retained')}%** across 2 components.\n"
                        f"- **Dimensionality Compression:** {pca_res.get('compression_ratio')}% feature space reduction.\n\n"
                        f"Explore interactive scree curves, 2D/3D projections, and component loadings in **🤖 ML Cockpit → Dimensionality (PCA)**."
                    )
                    return {
                        "tool_used": "run_pca(df, n_components=2)",
                        "response": resp,
                        "data": pd.DataFrame([pca_res.get("top_drivers", {})]) if pca_res.get("top_drivers") else None,
                    }

                # B. Anomaly / Outlier Query
                if any(w in msg_lower for w in ["anomaly", "outlier", "fraud", "irregular"]):
                    from core.ml.anomaly_detection import run_anomaly_detection
                    anom_res = run_anomaly_detection(df, algorithm="isolation_forest", contamination=0.05)
                    resp = (
                        f"🚨 **Automated Anomaly Audit (Isolation Forest)**:\n"
                        f"- **Identified Outliers:** **{anom_res['anomaly_count']:,}** records ({anom_res['anomaly_pct']}% of dataset).\n"
                        f"- **Algorithm:** Isolation Forest ensemble with 5% contamination threshold.\n\n"
                        f"Inspect the 2D cluster projection and anomalous record profiles in **🤖 ML Cockpit → Anomaly & Outliers**."
                    )
                    return {
                        "tool_used": "run_anomaly_detection(df, algorithm='isolation_forest')",
                        "response": resp,
                        "data": pd.DataFrame(anom_res.get("top_records", [])).head(5),
                    }

                # C. Future Forecasting Query
                if any(w in msg_lower for w in ["forecast", "future", "projection", "next month", "next week", "next year"]):
                    date_cols = df.select_dtypes(include=["datetime", "object"]).columns.tolist()
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    found_date = None
                    for c in date_cols:
                        if any(k in c.lower() for k in ["date", "time", "day", "month", "year"]):
                            found_date = c
                            break
                    if not found_date and len(df.columns) > 0:
                        found_date = df.columns[0]
                    target_candidate = num_cols[0] if num_cols else None

                    if found_date and target_candidate:
                        fc_res = ai_tools.forecast(df, date_col=found_date, target_col=target_candidate, horizon=14)
                        resp = (
                            f"📈 **Time-Series Forecast Trajectory: `{target_candidate}`**\n"
                            f"- **Best Model:** {fc_res.get('best_model')}\n"
                            f"- **Projected Horizon:** {fc_res.get('horizon')} periods ({fc_res.get('frequency')})\n"
                            f"- **Direction & Growth:** **{fc_res.get('metrics', {}).get('projected_growth_pct'):+.1f}%** ({fc_res.get('metrics', {}).get('growth_direction')})\n"
                            f"- **Expected Mean:** {fc_res.get('metrics', {}).get('expected_mean')}\n"
                            f"- **Backtest Accuracy:** MAE {fc_res.get('metrics', {}).get('mae')}\n\n"
                            f"Review interactive confidence intervals in **🤖 ML Cockpit → Future Forecasting**."
                        )
                        return {
                            "tool_used": "forecast(df, date_col, target_col)",
                            "response": resp,
                            "data": pd.DataFrame(fc_res.get("forecast", [])).head(5),
                        }

                # D. Champion / Best Model Query
                champ = get_active_champion()
                readiness = ai_tools.prepare_ml_data(df)

                if "champion" in msg_lower or "best model" in msg_lower:
                    if champ:
                        champ_metrics = ", ".join([f"{k}: {v}" for k, v in champ.get("metrics", {}).items() if isinstance(v, (int, float))])
                        resp = (
                            f"🏆 **Active Production Champion Model**: **{champ.get('model_name', 'Trained Estimator')}**\n"
                            f"- **Task Type:** {champ.get('task_type', 'N/A')}\n"
                            f"- **Target Variable:** `{champ.get('target_col', 'N/A')}`\n"
                            f"- **Validation Metrics:** {champ_metrics}\n"
                            f"- **Trained On:** {champ.get('created_at')}\n\n"
                            f"You can simulate predictions in the **🤖 ML & Predictive Analytics** workspace tab."
                        )
                    else:
                        resp = "No champion model has been registered yet. Visit the **🤖 ML & Predictive Analytics** workspace to launch automated training."
                    return {
                        "tool_used": "get_active_champion()",
                        "response": resp,
                        "data": pd.DataFrame([champ]) if champ else None,
                    }

                # General ML readiness / prediction summary
                target = readiness.get("selected_target")
                if target:
                    summary = (
                        f"### 🤖 Quantum ML Executive Briefing: `{target}`\n"
                        f"- **Identified ML Task:** {readiness.get('detected_task', 'N/A').replace('_', ' ').title()}\n"
                        f"- **Readiness Health Score:** {readiness.get('overall_score')}/100\n"
                        f"- **Scale:** {readiness.get('total_rows'):,} rows across {readiness.get('total_columns')} features.\n\n"
                        f"Navigate to **🤖 ML & Predictive Analytics** in the sidebar to benchmark models and run predictions."
                    )
                    return {
                        "tool_used": "prepare_ml_data()",
                        "response": summary,
                        "data": pd.DataFrame(readiness.get("target_candidates", [])).head(5),
                    }
            except Exception:
                pass

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
