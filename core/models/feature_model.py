import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from core.cache import AnalyticsCache

class FeatureEngineeringModel:
    """Dataset-aware feature engineering engine that identifies meaningful date expansions,

    business formulas (e.g. profit margin), customer analytics (RFM), and numerical transformations.
    Explains the formula, rationale, and analytical value for user approval.
    """

    @classmethod
    def recommend_features(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "feat_recs")
        if cached is not None:
            return cached

        candidates = []
        cols_lower = {str(c).lower(): c for c in df.columns}

        # 1. Date Feature Decompositions
        for lower_name, orig_col in cols_lower.items():
            if any(k in lower_name for k in ["date", "time", "timestamp"]):
                try:
                    pd.to_datetime(df[orig_col].dropna().iloc[:10])
                    candidates.append({
                        "id": f"date_decomp_{orig_col}",
                        "type": "date_expansion",
                        "feature_name": f"{orig_col}_year, {orig_col}_month, {orig_col}_dayofweek, {orig_col}_is_weekend",
                        "target_col": str(orig_col),
                        "formula": f"Extract Year, Month, Day of Week, and Weekend Flag from {orig_col}",
                        "reason": f"Decomposing '{orig_col}' enables seasonality analysis and temporal grouping.",
                        "expected_value": "High - unlocks periodic pattern discovery."
                    })
                except Exception:
                    pass

        # 2. Business Features: Profit Margin
        rev_col = next((cols_lower[k] for k in ["revenue", "sales", "turnover", "total_amount"] if k in cols_lower), None)
        prof_col = next((cols_lower[k] for k in ["profit", "net_profit", "earnings"] if k in cols_lower), None)
        cost_col = next((cols_lower[k] for k in ["cost", "cogs", "expenses"] if k in cols_lower), None)

        if prof_col and rev_col:
            candidates.append({
                "id": "business_profit_margin",
                "type": "ratio",
                "feature_name": "profit_margin",
                "col1": prof_col,
                "col2": rev_col,
                "formula": f"{prof_col} / ({rev_col} + 1e-6)",
                "reason": f"Measures percentage profitability per unit of {rev_col}.",
                "expected_value": "Very High - standardized metric for profitability efficiency."
            })
        elif rev_col and cost_col:
            candidates.append({
                "id": "business_profit_calc",
                "type": "subtract",
                "feature_name": "computed_profit",
                "col1": rev_col,
                "col2": cost_col,
                "formula": f"{rev_col} - {cost_col}",
                "reason": f"Derives net bottom-line profit from gross {rev_col} and {cost_col}.",
                "expected_value": "High - provides essential financial baseline."
            })

        # 3. Customer RFM / Aggregate Features
        cust_col = next((cols_lower[k] for k in ["customer_id", "user_id", "client_id", "account_id"] if k in cols_lower), None)
        if cust_col and rev_col:
            candidates.append({
                "id": "customer_monetary_spend",
                "type": "rfm_monetary",
                "feature_name": f"customer_total_spend",
                "customer_col": cust_col,
                "metric_col": rev_col,
                "formula": f"SUM({rev_col}) grouped by {cust_col}",
                "reason": f"Computes cumulative monetary value per {cust_col} for high-value customer segmentation.",
                "expected_value": "Very High - foundational for customer lifetime value (CLV)."
            })

        # 4. Numerical Transformations (Log Transform for skewed variables)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if "id" in str(col).lower():
                continue
            series = df[col].dropna()
            if len(series) > 10:
                s_sample = series.sample(5000, random_state=42) if len(series) > 5000 else series
                skew = s_sample.skew()
                min_val = s_sample.min()
                if skew > 1.5 and min_val >= 0:
                    candidates.append({
                        "id": f"log_transform_{col}",
                        "type": "log_transform",
                        "feature_name": f"log_{col}",
                        "target_col": str(col),
                        "formula": f"np.log1p({col})",
                        "reason": f"Column '{col}' exhibits heavy right skewness ({skew:.2f}).",
                        "expected_value": "Medium - normalizes heavy tails for linear modeling."
                    })
                    break

        # 5. Numerical Binning / Discretization
        for col in num_cols:
            if any(k in str(col).lower() for k in ["age", "income", "salary", "sales"]):
                candidates.append({
                    "id": f"binning_{col}",
                    "type": "binning",
                    "feature_name": f"{col}_binned",
                    "target_col": str(col),
                    "formula": f"pd.qcut({col}, q=4, labels=['Low', 'Medium', 'High', 'Very High'])",
                    "reason": f"Discretizes continuous '{col}' into 4 intuitive categorical quartiles.",
                    "expected_value": "Medium - useful for stratified comparisons in dashboards."
                })
                break

        AnalyticsCache.set(df, "feat_recs", candidates)
        return candidates

    @classmethod
    def apply_feature(cls, df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        df_out = df.copy()
        feat_type = spec.get("type")

        if feat_type == "date_expansion":
            target_col = spec.get("target_col")
            dt_series = pd.to_datetime(df_out[target_col], errors="coerce")
            df_out[f"{target_col}_year"] = dt_series.dt.year
            df_out[f"{target_col}_month"] = dt_series.dt.month
            df_out[f"{target_col}_dayofweek"] = dt_series.dt.day_name()
            df_out[f"{target_col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            return df_out, f"Added 4 temporal features: {target_col}_year, {target_col}_month, {target_col}_dayofweek, {target_col}_is_weekend"

        elif feat_type == "ratio":
            col1 = spec.get("col1")
            col2 = spec.get("col2")
            out_name = spec.get("feature_name", f"{col1}_per_{col2}")
            df_out[out_name] = (df_out[col1] / (df_out[col2].replace(0, np.nan))).round(4)
            return df_out, f"Created ratio feature '{out_name}' ({col1} / {col2})"

        elif feat_type == "subtract":
            col1 = spec.get("col1")
            col2 = spec.get("col2")
            out_name = spec.get("feature_name", f"{col1}_minus_{col2}")
            df_out[out_name] = (df_out[col1] - df_out[col2]).round(2)
            return df_out, f"Created difference feature '{out_name}' ({col1} - {col2})"

        elif feat_type == "log_transform":
            target_col = spec.get("target_col")
            out_name = spec.get("feature_name", f"log_{target_col}")
            df_out[out_name] = np.log1p(df_out[target_col].clip(lower=0)).round(4)
            return df_out, f"Created log-transformed feature '{out_name}'"

        elif feat_type == "binning":
            target_col = spec.get("target_col")
            out_name = spec.get("feature_name", f"{target_col}_binned")
            try:
                df_out[out_name] = pd.qcut(df_out[target_col], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop")
            except Exception:
                df_out[out_name] = pd.cut(df_out[target_col], bins=4, labels=["Low", "Medium", "High", "Very High"])
            return df_out, f"Created binned quantile feature '{out_name}'"

        elif feat_type == "rfm_monetary":
            cust_col = spec.get("customer_col")
            metric_col = spec.get("metric_col")
            out_name = spec.get("feature_name", f"{cust_col}_total_spend")
            cust_totals = df_out.groupby(cust_col)[metric_col].transform("sum").round(2)
            df_out[out_name] = cust_totals
            return df_out, f"Created customer aggregate feature '{out_name}'"

        return df_out, "No transformation applied"
