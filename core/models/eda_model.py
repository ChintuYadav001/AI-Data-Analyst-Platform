import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
from core.cache import AnalyticsCache, compute_df_hash

class EDAModel:
    """Exploratory Data Analysis engine computing numerical distributions, skewness,
    categorical frequencies, correlation relationships, and temporal trends.
    """

    @classmethod
    def analyze_numerical(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "eda_numerical")
        if cached is not None:
            return cached
        results = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std())
            var_val = float(series.var())
            min_val = float(series.min())
            max_val = float(series.max())
            q25 = float(series.quantile(0.25))
            q50 = median_val
            q75 = float(series.quantile(0.75))
            iqr_val = q75 - q25
            skew_val = float(series.skew())
            
            mode_series = series.mode()
            mode_val = float(mode_series.iloc[0]) if not mode_series.empty else median_val
            
            # Distribution histogram bins
            counts, bin_edges = np.histogram(series, bins=min(15, max(5, series.nunique())))
            hist_data = [
                {"bin_start": round(float(bin_edges[i]), 2), "bin_end": round(float(bin_edges[i+1]), 2), "count": int(counts[i])}
                for i in range(len(counts))
            ]
            
            dist_type = "Normal-like" if abs(skew_val) < 0.5 else ("Right-skewed" if skew_val > 0.5 else "Left-skewed")
            
            results.append({
                "column": str(col),
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "mode": round(mode_val, 4),
                "std": round(std_val, 4),
                "variance": round(var_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "q25": round(q25, 4),
                "q50": round(q50, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr_val, 4),
                "skewness": round(skew_val, 4),
                "distribution_shape": dist_type,
                "histogram": hist_data
            })
            
        AnalyticsCache.set(df, "eda_numerical", results)
        return results

    @classmethod
    def analyze_categorical(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "eda_categorical")
        if cached is not None:
            return cached
        results = []
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        total_rows = len(df)
        
        for col in cat_cols:
            series = df[col].dropna().astype(str)
            if series.empty or series.nunique() > 200:
                continue
            
            val_counts = series.value_counts()
            top_cats = [
                {"category": str(k), "count": int(v), "percentage": round((v / total_rows) * 100, 2)}
                for k, v in val_counts.head(10).items()
            ]
            
            # Rare categories: < 2% of data
            rare_threshold = 0.02 * total_rows
            rare_cats = [
                {"category": str(k), "count": int(v), "percentage": round((v / total_rows) * 100, 2)}
                for k, v in val_counts.items() if v <= rare_threshold
            ]
            
            results.append({
                "column": str(col),
                "unique_count": series.nunique(),
                "top_categories": top_cats,
                "rare_categories": rare_cats[:10],
                "rare_count_total": len(rare_cats)
            })
            
        AnalyticsCache.set(df, "eda_categorical", results)
        return results

    @classmethod
    def analyze_relationships(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cached = AnalyticsCache.get(df, "eda_relationships")
        if cached is not None:
            return cached
        num_df = df.select_dtypes(include=[np.number]).dropna()
        if num_df.shape[1] < 2 or len(num_df) < 5:
            res = {"correlation_matrix": {}, "strong_positive": [], "strong_negative": []}
            AnalyticsCache.set(df, "eda_relationships", res)
            return res
            
        corr = num_df.corr(method="pearson").round(3)
        corr_dict = corr.to_dict()
        
        strong_pos = []
        strong_neg = []
        cols = list(corr.columns)
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = float(corr.iloc[i, j])
                if np.isnan(r):
                    continue
                pair = {"col1": str(cols[i]), "col2": str(cols[j]), "correlation": round(r, 3)}
                if r >= 0.6:
                    strong_pos.append(pair)
                elif r <= -0.6:
                    strong_neg.append(pair)
                    
        strong_pos = sorted(strong_pos, key=lambda x: x["correlation"], reverse=True)
        strong_neg = sorted(strong_neg, key=lambda x: x["correlation"])
        
        res = {
            "correlation_matrix": corr_dict,
            "columns": cols,
            "strong_positive": strong_pos,
            "strong_negative": strong_neg
        }
        AnalyticsCache.set(df, "eda_relationships", res)
        return res

    @classmethod
    def analyze_time_trends(cls, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "eda_time_trends")
        if cached is not None:
            return cached
        # Find potential date column
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
            if any(k in str(col).lower() for k in ["date", "time", "timestamp"]):
                try:
                    pd.to_datetime(df[col].dropna().iloc[:10])
                    date_col = col
                    break
                except Exception:
                    pass
                    
        if not date_col:
            return None
            
        temp_df = df.copy()
        temp_df["_parsed_date"] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna(subset=["_parsed_date"]).sort_values("_parsed_date")
        
        num_cols = temp_df.select_dtypes(include=[np.number]).columns
        primary_num = None
        for col in num_cols:
            if col != "_parsed_date" and "id" not in str(col).lower():
                primary_num = col
                break
                
        if not primary_num:
            return None
            
        trends = {}
        # Daily
        daily = temp_df.set_index("_parsed_date").resample("D")[primary_num].sum().reset_index()
        trends["daily"] = [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in zip(daily["_parsed_date"], daily[primary_num])]
        # Monthly
        monthly = temp_df.set_index("_parsed_date").resample("ME")[primary_num].sum().reset_index()
        trends["monthly"] = [{"date": d.strftime("%Y-%m"), "value": round(float(v), 2)} for d, v in zip(monthly["_parsed_date"], monthly[primary_num])]
        
        res = {
            "date_column": str(date_col),
            "target_metric": str(primary_num),
            "daily_sample": trends["daily"][:30],
            "monthly": trends["monthly"]
        }
        AnalyticsCache.set(df, "eda_time_trends", res)
        return res

    # -------------------------------------------------------------------------
    # HIGH-LEVEL DATASET-AWARE & GOAL-AWARE AI EDA ENGINE (14-STAGE PIPELINE)
    # -------------------------------------------------------------------------
    @classmethod
    def run_ai_eda(
        cls,
        df: pd.DataFrame,
        user_goal: Optional[str] = None,
        user_target: Optional[str] = None,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """Executes the adaptive 14-stage AI EDA pipeline via the central AIEDAOrchestrator."""
        from core.eda.orchestrator import AIEDAOrchestrator
        return AIEDAOrchestrator.run_full_eda(df, user_goal=user_goal, user_target=user_target, mode=mode)

    @classmethod
    def ask_eda_assistant(
        cls,
        df: pd.DataFrame,
        query: str,
        target_col: Optional[str] = None,
        goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translates natural language questions into executed DuckDB/Pandas analytics."""
        from core.eda.nl_query import EDANLQueryEngine
        return EDANLQueryEngine.answer_query(df, query=query, target_col=target_col, goal=goal)

    @classmethod
    def explain_chart_insight(cls, chart_spec: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generates a strictly evidence-backed explanation of the specified chart."""
        explanation = chart_spec.get("explanation", "")
        if explanation:
            return explanation

        c_type = chart_spec.get("type", "chart")
        title = chart_spec.get("title", "Visualization")
        x_col = chart_spec.get("x")
        y_col = chart_spec.get("y")

        if c_type == "histogram" and x_col and x_col in df.columns:
            s = df[x_col].dropna()
            return f"Histogram reveals distribution of `{x_col}`: mean = {s.mean():.2f}, median = {s.median():.2f}, skewness = {s.skew():.2f} across {len(s):,} records."
        elif c_type == "scatter" and x_col and y_col and x_col in df.columns and y_col in df.columns:
            r = df[[x_col, y_col]].dropna().corr().iloc[0, 1]
            return f"Scatter diagram displays bivariate co-movement between `{x_col}` and `{y_col}` (Pearson r = {r:.3f}). Association does not imply causation."
        elif c_type == "bar" and x_col and x_col in df.columns:
            top_v = df[x_col].value_counts().head(1)
            return f"Bar breakdown highlights frequency concentration across `{x_col}`; top category `{top_v.index[0]}` accounts for {int(top_v.iloc[0]):,} records."
        return f"Analytical chart '{title}' computed from active dataset telemetry ({len(df):,} rows)."

    @classmethod
    def generate_chart_sql(cls, chart_spec: Dict[str, Any], df: pd.DataFrame) -> str:
        """Returns validated read-only SQL corresponding to the chart aggregation."""
        sql = chart_spec.get("sql", "")
        if sql:
            return sql

        c_type = chart_spec.get("type", "chart")
        x_col = chart_spec.get("x", "dimension")
        y_col = chart_spec.get("y", "metric")

        if c_type in ["bar", "line"]:
            return f"SELECT {x_col}, SUM({y_col}) AS total_{y_col} FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 10;"
        elif c_type == "histogram":
            return f"SELECT {x_col} FROM dataset WHERE {x_col} IS NOT NULL;"
        elif c_type == "scatter":
            return f"SELECT {x_col}, {y_col} FROM dataset WHERE {x_col} IS NOT NULL AND {y_col} IS NOT NULL;"
        return "SELECT * FROM dataset LIMIT 100;"

