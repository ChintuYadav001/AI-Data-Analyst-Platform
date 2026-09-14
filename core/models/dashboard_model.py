import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from core.models.chart_model import ChartModel
from core.cache import AnalyticsCache

class DashboardModel:
    """Enterprise dashboard engine generating prioritized KPIs, dynamic multi-dimension filters,

    analytical requirement specifications, and coherent data stories with sub-millisecond caching.
    """

    @classmethod
    def compute_kpis(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "kpis")
        if cached is not None:
            return cached

        kpis = []
        n_rows = len(df)
        if n_rows == 0:
            return kpis

        cols_lower = {str(c).lower(): c for c in df.columns}

        # 1. Total Volume / Records
        kpis.append({
            "id": "kpi_volume",
            "label": "Total Records",
            "value": f"{n_rows:,}",
            "subtext": "Active observations",
            "icon": "📊",
            "badge": "Volume",
            "status": "neutral",
            "rationale": "Fundamental baseline metric representing overall transactional activity."
        })

        # 2. Revenue / Sales
        sales_col = next((cols_lower[k] for k in ["sales", "revenue", "turnover", "total_amount", "balance", "price"] if k in cols_lower), None)
        total_sales = 0.0
        if sales_col and pd.api.types.is_numeric_dtype(df[sales_col]):
            total_sales = float(df[sales_col].sum())
            avg_sales = float(df[sales_col].mean())
            kpis.append({
                "id": "kpi_revenue",
                "label": f"Total {sales_col.replace('_', ' ').title()}",
                "value": f"${total_sales:,.2f}" if total_sales > 1000 else f"{total_sales:,.2f}",
                "subtext": f"Avg/row: ${avg_sales:,.2f}" if avg_sales > 100 else f"Avg: {avg_sales:,.2f}",
                "icon": "💰",
                "badge": "Top-Line",
                "status": "positive" if total_sales > 0 else "neutral",
                "rationale": f"Primary top-line financial measure extracted from column '{sales_col}'."
            })

        # 3. Profit / Bottom-Line Margin
        profit_col = next((cols_lower[k] for k in ["profit", "net_profit", "earnings", "income"] if k in cols_lower), None)
        if profit_col and pd.api.types.is_numeric_dtype(df[profit_col]):
            total_profit = float(df[profit_col].sum())
            profit_status = "positive" if total_profit > 0 else "negative"
            kpis.append({
                "id": "kpi_profit",
                "label": f"Total {profit_col.replace('_', ' ').title()}",
                "value": f"${total_profit:,.2f}" if abs(total_profit) > 1000 else f"{total_profit:,.2f}",
                "subtext": "Net bottom line",
                "icon": "📈",
                "badge": "Bottom-Line",
                "status": profit_status,
                "rationale": f"Bottom-line profitability driver extracted from column '{profit_col}'."
            })

            # Profit Margin Efficiency
            if sales_col and total_sales != 0:
                margin = (total_profit / total_sales) * 100
                margin_status = "positive" if margin >= 15 else ("neutral" if margin >= 0 else "negative")
                kpis.append({
                    "id": "kpi_margin",
                    "label": "Profit Margin",
                    "value": f"{margin:.1f}%",
                    "subtext": "Net revenue conversion",
                    "icon": "🎯",
                    "badge": "Efficiency",
                    "status": margin_status,
                    "rationale": "Standardized efficiency ratio comparing net profit against gross sales."
                })

        # 4. Distinct Customer / Entity Count
        cust_col = next((cols_lower[k] for k in ["customer_id", "user_id", "client_id", "employee_id", "account_id"] if k in cols_lower), None)
        if cust_col:
            unique_custs = int(df[cust_col].nunique())
            avg_per_cust = round(total_sales / unique_custs, 2) if total_sales > 0 and unique_custs > 0 else 0
            kpis.append({
                "id": "kpi_customers",
                "label": f"Unique {cust_col.replace('_id', '').title()}s",
                "value": f"{unique_custs:,}",
                "subtext": f"Avg spend: ${avg_per_cust:,.2f}" if avg_per_cust > 0 else "Distinct entities",
                "icon": "👥",
                "badge": "Audience",
                "status": "neutral",
                "rationale": f"Monitors customer reach and market penetration based on '{cust_col}'."
            })

        # Fill with secondary numeric aggregates if needed
        if len(kpis) < 4:
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col not in [sales_col, profit_col] and "id" not in str(col).lower():
                    mean_val = float(df[col].mean())
                    kpis.append({
                        "id": f"kpi_{col}",
                        "label": f"Average {str(col).replace('_', ' ').title()}",
                        "value": f"{mean_val:,.2f}",
                        "subtext": f"Median: {float(df[col].median()):,.2f}",
                        "icon": "⚡",
                        "badge": "Metric",
                        "status": "neutral",
                        "rationale": f"Significant numerical attribute '{col}' representing operational scale."
                    })
                if len(kpis) >= 5:
                    break

        AnalyticsCache.set(df, "kpis", kpis[:5])
        return kpis[:5]

    @classmethod
    def detect_filter_dimensions(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cached = AnalyticsCache.get(df, "dimensions")
        if cached is not None:
            return cached

        filters = []
        candidate_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        n_rows = len(df)
        for col in candidate_cols:
            if any(k in str(col).lower() for k in ["_id", "id_", "identifier", "uuid", "guid", "hash"]):
                continue
            if n_rows > 10000 and df[col].iloc[:500].nunique() > 50:
                continue
            unique_count = int(df[col].nunique())
            if 2 <= unique_count <= 50:
                vals = [str(x) for x in df[col].dropna().unique().tolist()]
                filters.append({
                    "column": str(col),
                    "options": sorted(vals),
                    "cardinality": unique_count,
                    "rationale": f"Dimension '{col}' possesses {unique_count} distinct categories ideal for cross-filtering."
                })
        
        AnalyticsCache.set(df, "dimensions", filters[:8])
        return filters[:8]

    @classmethod
    def apply_filters(cls, df: pd.DataFrame, active_filters: Dict[str, List[str]]) -> pd.DataFrame:
        if not active_filters:
            return df
        # Create a stable filter key for caching
        filter_items = tuple(sorted((k, tuple(sorted(v))) for k, v in active_filters.items() if v))
        if not filter_items:
            return df
            
        filter_str = str(filter_items)
        cached = AnalyticsCache.get(df, "filtered_df", extra=filter_str)
        if cached is not None:
            return cached

        mask = pd.Series(True, index=df.index)
        for col, selected_vals in filter_items:
            if col in df.columns:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    num_vals = []
                    for v in selected_vals:
                        try:
                            num_vals.append(float(v))
                        except ValueError:
                            pass
                    mask &= series.isin(num_vals)
                else:
                    if pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
                        mask &= series.isin(selected_vals)
                    else:
                        mask &= series.astype(str).isin(selected_vals)
                        
        filtered = df[mask]
        AnalyticsCache.set(df, "filtered_df", filtered, extra=filter_str)
        return filtered

    @classmethod
    def generate_dashboard_specification(cls, df: pd.DataFrame) -> Dict[str, Any]:
        cached = AnalyticsCache.get(df, "dash_spec")
        if cached is not None:
            return cached

        kpis = cls.compute_kpis(df)
        recs = ChartModel.recommend_charts(df)
        dimensions = cls.detect_filter_dimensions(df)

        spec = {
            "title": "Executive Data Analytics Architecture & Specification",
            "kpis": kpis,
            "charts": recs[:6],
            "filter_dimensions": dimensions,
            "architecture_principles": [
                "1. Pareto Priority: Visualizations emphasize top revenue and profit drivers first.",
                "2. Non-Destructive Integrity: All aggregations are computed on dynamic views without altering underlying data.",
                "3. Dual-Perspective Breakdown: Balances macro temporal growth trends with micro category segmentation.",
                "4. High Data-to-Ink Ratio: Minimalist visual layouts with high information density and responsive hover mechanics."
            ]
        }
        AnalyticsCache.set(df, "dash_spec", spec)
        return spec
