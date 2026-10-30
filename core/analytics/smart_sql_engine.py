"""
core.analytics.smart_sql_engine
================================
High-accuracy Natural Language to SQL Analytics Engine for DuckDB.
Features:
  - Rich semantic schema analysis & synonym resolution
  - Categorical entity inverted index (recognizes "California", "West", "Technology", etc.)
  - Accurate intent classification: Top N ranking, multi-metric grouping, time-series,
    filtered aggregations, scalar summaries, threshold audits
  - DuckDB-optimized SQL synthesis with double-quoted identifiers and robust date handling
  - Self-healing query execution with automated syntax/column correction
  - Executive Natural Language Answer Synthesizer with KPI metrics and evidence
  - Interactive multi-chart visualization generator (Bar, Line, Donut, Treemap, KPI Cards)
"""

from __future__ import annotations

import re
import difflib
import time
from typing import Dict, Any, List, Optional, Tuple
import duckdb
import pandas as pd
import numpy as np

from core.models.sql_model import SQLAnalyticsModel
from core.ai.llm_client import LLMClient


# ── Known Business Synonyms ──────────────────────────────────────────
COLUMN_SYNONYMS = {
    "sales": ["sales", "revenue", "turnover", "income", "amount", "gross_sales", "total_sales", "sold"],
    "profit": ["profit", "margin", "earnings", "net_profit", "income", "gain", "loss", "bottom_line"],
    "quantity": ["quantity", "volume", "units", "items", "count", "pieces", "qty", "number_of_items"],
    "discount": ["discount", "rebate", "markdown", "deduction", "promo", "concession", "discount_rate"],
    "customer": ["customer", "client", "buyer", "shopper", "consumer", "account", "user", "member"],
    "customer_name": ["customer_name", "client_name", "customer", "name", "full_name"],
    "product": ["product", "item", "article", "merchandise", "sku", "goods", "product_name"],
    "category": ["category", "department", "segment", "division", "item_type", "product_category"],
    "sub_category": ["sub_category", "subcategory", "sub-category", "item_type", "sub_class"],
    "region": ["region", "area", "zone", "territory", "location", "geo_region"],
    "state": ["state", "province", "territory", "region_state"],
    "city": ["city", "town", "metro", "municipality"],
    "country": ["country", "nation"],
    "ship_mode": ["ship_mode", "shipping_mode", "shipping_method", "delivery_mode", "carrier", "shipment"],
    "segment": ["segment", "customer_segment", "market_segment", "tier"],
    "date": ["date", "day", "month", "year", "quarter", "time", "period", "timestamp", "order_date", "ship_date"],
    "cost": ["cost", "expense", "cogs", "spending", "purchase_cost"],
    "price": ["price", "unit_price", "rate", "cost_per_unit", "listing_price"],
}


class SmartSQLEngine:
    """Enterprise NL-to-SQL Analytics Engine powered by DuckDB."""

    @classmethod
    def extract_metadata(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Deeply inspects the DataFrame schema, columns, types, and categorical entities."""
        numeric_cols = []
        cat_cols = []
        date_cols = []
        id_cols = []

        for col in df.columns:
            c_str = str(col)
            c_low = c_str.lower().strip()
            is_id = (
                any(k in c_low for k in ["_id", "id_", "identifier", "row_id", "row id", "index", "ticket"])
                or c_low.endswith(" id")
                or c_low.endswith(" code")
                or c_low == "id"
            )
            if is_id:
                id_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(k in c_low for k in ["date", "time", "timestamp"]):
                date_cols.append(col)
            else:
                cat_cols.append(col)

        # Build Inverted Categorical Value Index
        # Maps lowercase string value -> (column_name, original_value)
        value_index: Dict[str, Tuple[str, str]] = {}
        candidate_cols = [c for c in cat_cols if df[c].nunique() <= 300]
        for c in candidate_cols:
            try:
                unique_vals = df[c].dropna().unique()
                for uv in unique_vals:
                    uv_str = str(uv).strip()
                    if 1 < len(uv_str) < 40:
                        uv_low = uv_str.lower()
                        # Do not overwrite if already indexed with higher priority
                        if uv_low not in value_index:
                            value_index[uv_low] = (c, uv_str)
            except Exception:
                pass

        return {
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "date_cols": date_cols,
            "id_cols": id_cols,
            "value_index": value_index,
            "total_rows": len(df),
            "columns": list(df.columns),
        }

    @classmethod
    def get_informative_columns(
        cls,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        primary_metric: str,
        secondary_metric: Optional[str] = None
    ) -> List[str]:
        """Picks the most informative, descriptive columns for record drill-downs on any dataset."""
        selected: List[str] = []
        all_cols = list(df.columns)

        # 1. Primary ID / Code if available (prefer Order ID, Transaction ID, clean IDs)
        id_cols = metadata.get("id_cols", [])
        def id_priority(c: str) -> int:
            c_low = str(c).lower()
            if "order" in c_low or "transaction" in c_low:
                return 0
            if c_low == "id":
                return 1
            if "customer" in c_low:
                return 2
            if "product" in c_low:
                return 3
            return 4

        id_cols_sorted = sorted(id_cols, key=id_priority)
        for id_c in id_cols_sorted:
            if id_c in all_cols and id_c not in selected and "+" not in str(id_c):
                selected.append(id_c)
                break

        # 2. Key descriptive columns (Name, Title, Product, Customer, Category, City, etc.)
        cat_cols = [c for c in metadata.get("cat_cols", []) if c in all_cols and c not in selected]

        def desc_score(c: str) -> int:
            c_low = str(c).lower()
            if "product name" in c_low or "item name" in c_low:
                return 0
            if "customer name" in c_low or "client name" in c_low:
                return 1
            if any(w in c_low for w in ["name", "title"]):
                return 2
            if any(w in c_low for w in ["product", "item", "article"]):
                return 3
            if any(w in c_low for w in ["customer", "client", "user"]):
                return 4
            if any(w in c_low for w in ["category", "type", "segment", "department"]):
                return 5
            if any(w in c_low for w in ["city", "state", "region", "country"]):
                return 6
            return 7

        cat_cols.sort(key=desc_score)
        for c in cat_cols[:3]:
            if c not in selected:
                selected.append(c)

        # 3. Primary Metric
        if primary_metric and primary_metric in all_cols and primary_metric not in selected:
            selected.append(primary_metric)

        # 4. Secondary Metric
        if secondary_metric and secondary_metric in all_cols and secondary_metric not in selected:
            selected.append(secondary_metric)

        # Fallback if too few columns
        if len(selected) < 3:
            for c in all_cols:
                if c not in selected and len(selected) < 5:
                    selected.append(c)

        return selected

    @classmethod
    def resolve_column(
        cls,
        text: str,
        available_cols: List[str],
        synonym_family: Optional[str] = None
    ) -> Optional[str]:
        """Resolves a referenced name or concept to the most suitable column."""
        t_low = text.lower().strip()
        t_clean = re.sub(r"[\s_\-]+", "", t_low)

        # 1. Exact or cleaned match
        for c in available_cols:
            c_clean = re.sub(r"[\s_\-]+", "", str(c).lower())
            if t_clean == c_clean:
                return c

        # 2. Substring match
        for c in available_cols:
            c_low = str(c).lower()
            if t_low in c_low or c_low in t_low:
                return c

        # 3. Synonym dictionary matching
        if synonym_family and synonym_family in COLUMN_SYNONYMS:
            syn_words = COLUMN_SYNONYMS[synonym_family]
            for c in available_cols:
                c_clean = str(c).lower().replace(" ", "_")
                if any(syn in c_clean for syn in syn_words):
                    return c

        # 4. Check all synonym families
        for fam, words in COLUMN_SYNONYMS.items():
            if any(w in t_low for w in words):
                for c in available_cols:
                    c_clean = str(c).lower().replace(" ", "_")
                    if any(w in c_clean for w in words):
                        return c

        # 5. Fuzzy match
        col_names = [str(c) for c in available_cols]
        matches = difflib.get_close_matches(text, col_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    @classmethod
    def find_all_matching_columns(cls, question: str, columns: List[str]) -> List[str]:
        """Finds all columns from the dataset explicitly or implicitly mentioned in the question."""
        q_low = question.lower()
        matched = []

        # 1. Check multi-word and single-word column mentions
        for c in columns:
            c_str = str(c)
            c_low = c_str.lower()
            c_words = [w for w in re.split(r"[\s_\-]+", c_low) if len(w) > 2]

            # Exact phrase match in question (e.g. "product name", "sub category")
            c_phrase = " ".join(c_words)
            if c_phrase and re.search(r"\b" + re.escape(c_phrase) + r"\b", q_low):
                if c not in matched:
                    matched.append(c)
                continue

            # Check significant individual words (e.g. "products" in question matches "Product Name")
            for w in c_words:
                singular = w.rstrip("s")
                plural = w + "s" if not w.endswith("s") else w
                if re.search(r"\b" + re.escape(w) + r"\b", q_low) or \
                   re.search(r"\b" + re.escape(singular) + r"\b", q_low) or \
                   re.search(r"\b" + re.escape(plural) + r"\b", q_low):
                    if c not in matched:
                        matched.append(c)
                    break

        # 2. Check synonym families
        for fam, words in COLUMN_SYNONYMS.items():
            if any(re.search(r"\b" + re.escape(w) + r"\b", q_low) for w in words):
                for c in columns:
                    c_clean = str(c).lower().replace(" ", "_").replace("-", "_")
                    if any(w in c_clean for w in words) and c not in matched:
                        matched.append(c)

        return matched

    @classmethod
    def extract_filters(
        cls,
        question: str,
        metadata: Dict[str, Any]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extracts WHERE conditions based on entity values, thresholds, and dates."""
        q_low = question.lower()
        conditions: List[str] = []
        filter_explanations: List[Dict[str, Any]] = []

        # 1. Categorical value entity extraction from inverted index
        value_index = metadata.get("value_index", {})
        # Sort keys by length descending to match longest phrases first (e.g. "North Carolina" before "Carolina")
        sorted_vals = sorted(value_index.keys(), key=lambda x: len(x), reverse=True)

        matched_entities = set()
        for val_low in sorted_vals:
            if len(val_low) < 3:
                continue
            # Word boundary search
            pattern = r"\b" + re.escape(val_low) + r"\b"
            if re.search(pattern, q_low):
                col, orig_val = value_index[val_low]
                if col not in matched_entities:
                    matched_entities.add(col)
                    escaped_val = orig_val.replace("'", "''")
                    conditions.append(f'"{col}" ILIKE \'%{escaped_val}%\'')
                    filter_explanations.append({
                        "column": col,
                        "value": orig_val,
                        "type": "categorical_match"
                    })

        # 2. Numeric threshold filters
        num_cols = metadata.get("numeric_cols", [])
        profit_col = cls.resolve_column("profit", num_cols, "profit")
        sales_col = cls.resolve_column("sales", num_cols, "sales")
        discount_col = cls.resolve_column("discount", num_cols, "discount")

        # Negative profit / loss detection
        if any(k in q_low for k in ["negative profit", "loss making", "in loss", "losses", "unprofitable", "loss"]) and profit_col:
            conditions.append(f'"{profit_col}" < 0')
            filter_explanations.append({"column": profit_col, "operator": "<", "value": 0, "type": "loss_filter"})
        elif any(k in q_low for k in ["profitable", "positive profit"]) and profit_col:
            conditions.append(f'"{profit_col}" > 0')
            filter_explanations.append({"column": profit_col, "operator": ">", "value": 0, "type": "profit_filter"})

        # Greater than / Less than threshold extraction (e.g. "discount > 0.2", "sales > 500")
        ineq_matches = re.findall(r'([a-zA-Z\s_]+)\s*(>|<|>=|<=|=|greater than|more than|less than|over|under)\s*([0-9\.]+)\s*(%?)', q_low)
        for col_cand, op, num_val, pct in ineq_matches:
            col_target = cls.resolve_column(col_cand, num_cols)
            if col_target:
                val = float(num_val)
                if pct == "%" or "pct" in col_cand:
                    val = val / 100.0 if val > 1.0 else val
                sql_op = ">" if any(w in op for w in [">", "greater", "more", "over"]) else ("<" if any(w in op for w in ["<", "less", "under"]) else "=")
                conditions.append(f'"{col_target}" {sql_op} {val}')
                filter_explanations.append({"column": col_target, "operator": sql_op, "value": val, "type": "threshold"})

        # 3. Year / Temporal filter (e.g. "in 2017", "year 2018")
        date_cols = metadata.get("date_cols", [])
        if date_cols:
            d_col = date_cols[0]
            year_match = re.search(r'\b(in|for|year|during)\s*(20\d\d|19\d\d)\b', q_low) or re.search(r'\b(20\d\d|19\d\d)\b', q_low)
            if year_match:
                year_val = year_match.group(2) if year_match.lastindex and year_match.lastindex >= 2 else year_match.group(1)
                if year_val.isdigit() and len(year_val) == 4:
                    conditions.append(f'EXTRACT(YEAR FROM TRY_CAST("{d_col}" AS TIMESTAMP)) = {year_val}')
                    filter_explanations.append({"column": d_col, "year": int(year_val), "type": "year_filter"})

        return conditions, filter_explanations

    @classmethod
    def generate_smart_sql(
        cls,
        question: str,
        df: pd.DataFrame,
        table: str = "dataset"
    ) -> Dict[str, Any]:
        """
        Translates a natural language question into an accurate DuckDB SQL query.
        Returns query, intent, target columns, filters, and explanatory context.
        """
        metadata = cls.extract_metadata(df)
        num_cols = metadata["numeric_cols"]
        cat_cols = metadata["cat_cols"]
        date_cols = metadata["date_cols"]
        all_cols = metadata["columns"]

        q_clean = question.strip()
        q_low = q_clean.lower()

        # Step 1: Detect explicit and implicit column references
        matched_nums = cls.find_all_matching_columns(q_clean, num_cols)
        matched_cats = cls.find_all_matching_columns(q_clean, cat_cols)

        # Primary metric & dimension defaults
        primary_metric = matched_nums[0] if matched_nums else (cls.resolve_column("sales", num_cols, "sales") or (num_cols[0] if num_cols else None))
        secondary_metric = matched_nums[1] if len(matched_nums) > 1 else (cls.resolve_column("profit", num_cols, "profit") if primary_metric != cls.resolve_column("profit", num_cols, "profit") else None)
        
        # Step 2: Extract WHERE filters
        where_conds, filter_explanations = cls.extract_filters(q_clean, metadata)
        where_clause = f"\nWHERE {' AND '.join(where_conds)}" if where_conds else ""

        # Step 3: Classify Intent
        # Advanced & Extreme Intent Detections
        is_pareto = any(k in q_low for k in ["pareto", "80/20", "80-20", "80 20", "cumulative revenue share", "80% of", "cumulative share"])
        is_growth = any(k in q_low for k in ["mom", "yoy", "month over month", "year over year", "growth rate", "period over period", "sales growth", "growth by month", "growth by year"]) and bool(date_cols)
        is_running_total = any(k in q_low for k in ["running total", "cumulative total", "cumulative sum", "cumulative sales", "running sum", "cumulative profit"]) and bool(date_cols)
        is_moving_avg = any(k in q_low for k in ["moving average", "rolling average", "moving avg", "rolling avg", "rolling sales", "smoothed"]) and bool(date_cols)
        is_pct_of_total = any(k in q_low for k in ["percent of total", "percentage of total", "share of total", "pct of total", "contribution of each", "revenue share"])
        is_rfm = any(k in q_low for k in ["rfm", "customer tier", "customer segments", "champion", "customer segmentation", "monetary frequency"])
        is_outlier = any(k in q_low for k in ["outlier", "outliers", "anomaly", "anomalies", "z-score", "z score", "anomalous", "unusual"])
        is_above_avg = any(k in q_low for k in ["above average", "greater than average", "higher than average", "exceeding average"])
        is_below_avg = any(k in q_low for k in ["below average", "less than average", "lower than average"])
        is_duration = any(k in q_low for k in ["duration", "days to ship", "shipping delay", "delivery time", "time to ship", "shipping time"]) and len(date_cols) >= 2
        is_having = any(k in q_low for k in ["having", "with more than", "with greater than", "who ordered more than", "with sales greater than", "with orders over"])

        # Ranking / Top N / Bottom N
        is_top = any(k in q_low for k in ["top", "highest", "best", "most", "largest", "maximum", "max", "leader", "greatest", "peak"])
        is_bottom = any(k in q_low for k in ["bottom", "lowest", "worst", "least", "smallest", "minimum", "min", "laggard"])
        limit_match = re.search(r'\b(top|bottom|first|last|highest|lowest)\s*(\d+)\b', q_low)
        limit_n = int(limit_match.group(2)) if limit_match else (1 if ("which" in q_low or "what is the highest" in q_low or "what is the best" in q_low) else 10)
        if limit_n > 50:
            limit_n = 50

        is_partitioned_top = (is_top or is_bottom) and any(k in q_low for k in ["in each", "for each", "per each", "within each", "in every", "for every", "per category", "per region"])

        is_ranking_intent = is_top or is_bottom or any(k in q_low for k in ["which", "leader", "worst"])
        is_group_intent = any(k in q_low for k in ["by", "per", "each", "across", "for each", "breakdown", "distribution"])

        # Assign primary dimension only if matched or explicitly requested by ranking/grouping intent
        filter_cols = {f["column"] for f in filter_explanations if "column" in f}
        candidate_dims = [c for c in matched_cats if c not in filter_cols]

        if candidate_dims:
            primary_dim = candidate_dims[0]
            secondary_dim = candidate_dims[1] if len(candidate_dims) > 1 else None
        elif is_group_intent or is_pct_of_total:
            primary_dim = cls.resolve_column("category", cat_cols, "category") or (cat_cols[0] if cat_cols else None)
            secondary_dim = None
        else:
            primary_dim = None
            secondary_dim = None

        # B. Time-series / Chronological
        is_timeseries = any(k in q_low for k in ["trend", "monthly", "by month", "over time", "history", "trajectory", "by year", "yearly", "quarterly", "daily"]) and bool(date_cols)

        # C. Counts / Row count / Distinct count
        is_count_only = any(k in q_low for k in ["how many rows", "total rows", "number of records", "row count", "how many records"])
        is_distinct_count = any(k in q_low for k in ["how many unique", "how many distinct", "distinct customers", "unique customers", "how many different"])

        # D. Summary / Scalar KPI (Total, Average, Overall)
        is_scalar_total = any(k in q_low for k in ["what is the total", "what is total", "overall", "entire", "grand total", "sum of all"]) and not is_group_intent

        # E. Multi-Metric (e.g. Sales AND Profit)
        has_multi_metric = len(matched_nums) >= 2 or (primary_metric and secondary_metric and any(k in q_low for k in ["and", "both", "versus", "vs", "compare", "with"]))

        # ── SYNTHESIS LOGIC (ADVANCED TO BASIC PRIORITY) ──

        # Case 0A: Partitioned Window Ranking (Top N within each group)
        if is_partitioned_top and primary_metric:
            group_dim = None
            item_dim = None
            for c in matched_cats:
                c_low_w = str(c).lower()
                if any(w in c_low_w for w in ["region", "category", "segment", "department", "state"]):
                    if not group_dim:
                        group_dim = c
                elif any(w in c_low_w for w in ["product", "item", "customer", "sub"]):
                    if not item_dim:
                        item_dim = c
            if not group_dim and cat_cols:
                group_dim = cat_cols[0]
            if not item_dim:
                item_dim = cat_cols[1] if len(cat_cols) > 1 and cat_cols[1] != group_dim else (cat_cols[2] if len(cat_cols) > 2 else "item")
            if not item_dim or item_dim == group_dim:
                item_dim = all_cols[0]

            order_dir = "ASC" if is_bottom else "DESC"
            clean_m = primary_metric.replace(" ", "_")
            sql = (
                f'WITH ranked_partition AS (\n'
                f'  SELECT\n'
                f'    "{group_dim}",\n'
                f'    "{item_dim}",\n'
                f'    ROUND(SUM("{primary_metric}"), 2) AS total_{clean_m},\n'
                f'    ROW_NUMBER() OVER (PARTITION BY "{group_dim}" ORDER BY SUM("{primary_metric}") {order_dir}) AS rank\n'
                f'  FROM "{table}"{where_clause}\n'
                f'  GROUP BY "{group_dim}", "{item_dim}"\n'
                f')\n'
                f'SELECT "{group_dim}", "{item_dim}", total_{clean_m}, rank\n'
                f'FROM ranked_partition\n'
                f'WHERE rank <= {limit_n}\n'
                f'ORDER BY "{group_dim}" ASC, rank ASC;'
            )
            return {
                "sql": sql,
                "intent": "window_ranking_per_group",
                "primary_metric": f"total_{clean_m}",
                "secondary_metric": "rank",
                "primary_dim": group_dim,
                "secondary_dim": item_dim,
                "limit": limit_n,
                "filters": filter_explanations,
                "explanation": f"Top {limit_n} {item_dim} ranked within each {group_dim} by {primary_metric} using window function partitioning."
            }

        # Case 0B: Period-over-Period Growth (MoM / YoY Growth Rate)
        if is_growth and primary_metric and date_cols:
            d_col = date_cols[0]
            is_yearly = any(k in q_low for k in ["year over year", "yoy", "annual", "by year"])
            if is_yearly:
                p_expr = f'EXTRACT(YEAR FROM TRY_CAST("{d_col}" AS TIMESTAMP))'
                p_alias = "period_year"
                growth_label = "YoY"
            else:
                p_expr = f'DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP))'
                p_alias = "period_month"
                growth_label = "MoM"
            clean_m = primary_metric.replace(" ", "_")
            sql = (
                f'WITH period_summary AS (\n'
                f'  SELECT\n'
                f'    {p_expr} AS {p_alias},\n'
                f'    ROUND(SUM("{primary_metric}"), 2) AS current_{clean_m}\n'
                f'  FROM "{table}"{where_clause}\n'
                f'  WHERE {p_expr} IS NOT NULL\n'
                f'  GROUP BY 1\n'
                f'),\n'
                f'growth_calc AS (\n'
                f'  SELECT\n'
                f'    {p_alias},\n'
                f'    current_{clean_m},\n'
                f'    LAG(current_{clean_m}, 1) OVER (ORDER BY {p_alias} ASC) AS prior_{clean_m},\n'
                f'    ROUND(((current_{clean_m} - LAG(current_{clean_m}, 1) OVER (ORDER BY {p_alias} ASC)) / NULLIF(LAG(current_{clean_m}, 1) OVER (ORDER BY {p_alias} ASC), 0)) * 100.0, 2) AS {growth_label.lower()}_growth_pct\n'
                f'  FROM period_summary\n'
                f')\n'
                f'SELECT {p_alias}, current_{clean_m}, prior_{clean_m}, {growth_label.lower()}_growth_pct\n'
                f'FROM growth_calc\n'
                f'ORDER BY {p_alias} ASC;'
            )
            return {
                "sql": sql,
                "intent": "period_over_period",
                "primary_metric": f"current_{clean_m}",
                "secondary_metric": f"{growth_label.lower()}_growth_pct",
                "primary_dim": p_alias,
                "filters": filter_explanations,
                "growth_type": growth_label,
                "explanation": f"{growth_label} growth rate analysis of {primary_metric} over time using LAG() window functions."
            }

        # Case 0C: Cumulative Running Total
        if is_running_total and primary_metric and date_cols:
            d_col = date_cols[0]
            p_expr = f'DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP))'
            clean_m = primary_metric.replace(" ", "_")
            sql = (
                f'WITH monthly_totals AS (\n'
                f'  SELECT\n'
                f'    {p_expr} AS month,\n'
                f'    ROUND(SUM("{primary_metric}"), 2) AS monthly_{clean_m}\n'
                f'  FROM "{table}"{where_clause}\n'
                f'  WHERE {p_expr} IS NOT NULL\n'
                f'  GROUP BY 1\n'
                f')\n'
                f'SELECT\n'
                f'  month,\n'
                f'  monthly_{clean_m},\n'
                f'  ROUND(SUM(monthly_{clean_m}) OVER (ORDER BY month ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_{clean_m}\n'
                f'FROM monthly_totals\n'
                f'ORDER BY month ASC;'
            )
            return {
                "sql": sql,
                "intent": "running_total",
                "primary_metric": f"cumulative_{clean_m}",
                "secondary_metric": f"monthly_{clean_m}",
                "primary_dim": "month",
                "filters": filter_explanations,
                "explanation": f"Cumulative running total progression of {primary_metric} using unbounded preceding window frames."
            }

        # Case 0D: Moving / Rolling Average
        if is_moving_avg and primary_metric and date_cols:
            d_col = date_cols[0]
            p_expr = f'DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP))'
            clean_m = primary_metric.replace(" ", "_")
            sql = (
                f'WITH period_data AS (\n'
                f'  SELECT\n'
                f'    {p_expr} AS period,\n'
                f'    ROUND(SUM("{primary_metric}"), 2) AS period_{clean_m}\n'
                f'  FROM "{table}"{where_clause}\n'
                f'  WHERE {p_expr} IS NOT NULL\n'
                f'  GROUP BY 1\n'
                f')\n'
                f'SELECT\n'
                f'  period,\n'
                f'  period_{clean_m},\n'
                f'  ROUND(AVG(period_{clean_m}) OVER (ORDER BY period ASC ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS rolling_3_period_avg\n'
                f'FROM period_data\n'
                f'ORDER BY period ASC;'
            )
            return {
                "sql": sql,
                "intent": "moving_average",
                "primary_metric": f"period_{clean_m}",
                "secondary_metric": "rolling_3_period_avg",
                "primary_dim": "period",
                "filters": filter_explanations,
                "explanation": f"Smoothed 3-period moving average of {primary_metric} across time."
            }

        # Case 0E: Pareto 80/20 Cumulative Share Analysis
        if is_pareto and primary_metric:
            target_dim = primary_dim or (cat_cols[0] if cat_cols else all_cols[0])
            clean_m = primary_metric.replace(" ", "_")
            sql = (
                f'WITH item_aggregation AS (\n'
                f'  SELECT "{target_dim}", ROUND(SUM("{primary_metric}"), 2) AS total_{clean_m}\n'
                f'  FROM "{table}"{where_clause}\n'
                f'  GROUP BY "{target_dim}"\n'
                f'),\n'
                f'pareto_calc AS (\n'
                f'  SELECT\n'
                f'    "{target_dim}",\n'
                f'    total_{clean_m},\n'
                f'    ROUND(100.0 * SUM(total_{clean_m}) OVER (ORDER BY total_{clean_m} DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / NULLIF(SUM(total_{clean_m}) OVER (), 0), 2) AS cumulative_pct\n'
                f'  FROM item_aggregation\n'
                f')\n'
                f'SELECT "{target_dim}", total_{clean_m}, cumulative_pct\n'
                f'FROM pareto_calc\n'
                f'ORDER BY total_{clean_m} DESC\n'
                f'LIMIT 25;'
            )
            return {
                "sql": sql,
                "intent": "pareto_80_20",
                "primary_metric": f"total_{clean_m}",
                "secondary_metric": "cumulative_pct",
                "primary_dim": target_dim,
                "filters": filter_explanations,
                "explanation": f"Pareto 80/20 cumulative revenue share of {primary_metric} across {target_dim}."
            }

        # Case 0F: RFM Customer Segmentation
        if is_rfm and primary_metric:
            cust_col = None
            for c in cat_cols + all_cols:
                if any(w in str(c).lower() for w in ["customer", "client", "user", "account", "buyer"]):
                    cust_col = c
                    break
            if not cust_col:
                cust_col = cat_cols[0] if cat_cols else all_cols[0]

            id_col = None
            for c in metadata.get("id_cols", []) + all_cols:
                if any(w in str(c).lower() for w in ["order", "trans", "invoice", "ticket"]):
                    id_col = c
                    break
            if not id_col:
                id_col = all_cols[0]

            sql = (
                f'WITH customer_summary AS (\n'
                f'  SELECT\n'
                f'    "{cust_col}",\n'
                f'    COUNT(DISTINCT "{id_col}") AS frequency,\n'
                f'    ROUND(SUM("{primary_metric}"), 2) AS monetary\n'
                f'  FROM "{table}"{where_clause}\n'
                f'  WHERE "{cust_col}" IS NOT NULL\n'
                f'  GROUP BY "{cust_col}"\n'
                f'),\n'
                f'rfm_scored AS (\n'
                f'  SELECT\n'
                f'    "{cust_col}",\n'
                f'    frequency,\n'
                f'    monetary,\n'
                f'    NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,\n'
                f'    NTILE(5) OVER (ORDER BY monetary ASC) AS m_score\n'
                f'  FROM customer_summary\n'
                f')\n'
                f'SELECT\n'
                f'  "{cust_col}",\n'
                f'  frequency,\n'
                f'  monetary,\n'
                f'  f_score,\n'
                f'  m_score,\n'
                f'  CASE\n'
                f'    WHEN m_score >= 4 AND f_score >= 4 THEN \'Champions / VIP\'\n'
                f'    WHEN m_score >= 3 AND f_score >= 3 THEN \'Loyal Customers\'\n'
                f'    WHEN m_score >= 3 AND f_score <= 2 THEN \'High-Value / At Risk\'\n'
                f'    ELSE \'Standard Customers\'\n'
                f'  END AS customer_tier\n'
                f'FROM rfm_scored\n'
                f'ORDER BY monetary DESC\n'
                f'LIMIT 30;'
            )
            return {
                "sql": sql,
                "intent": "rfm_segmentation",
                "primary_metric": "monetary",
                "secondary_metric": "frequency",
                "primary_dim": cust_col,
                "filters": filter_explanations,
                "explanation": f"RFM customer segmentation modeling using NTILE(5) scoring."
            }

        # Case 0G: Outlier Detection via Statistical Z-Score
        if is_outlier and primary_metric:
            info_cols = cls.get_informative_columns(df, metadata, primary_metric)
            sel_cols = ", ".join([f'd."{c}"' for c in info_cols[:3]])
            sql = (
                f'WITH stats AS (\n'
                f'  SELECT\n'
                f'    AVG("{primary_metric}") AS avg_val,\n'
                f'    STDDEV_SAMP("{primary_metric}") AS std_val\n'
                f'  FROM "{table}"\n'
                f')\n'
                f'SELECT\n'
                f'  {sel_cols},\n'
                f'  d."{primary_metric}",\n'
                f'  ROUND(ABS(d."{primary_metric}" - s.avg_val) / NULLIF(s.std_val, 0), 2) AS z_score\n'
                f'FROM "{table}" d, stats s\n'
                f'WHERE ABS(d."{primary_metric}" - s.avg_val) / NULLIF(s.std_val, 0) > 2.0\n'
                f'ORDER BY z_score DESC\n'
                f'LIMIT 25;'
            )
            return {
                "sql": sql,
                "intent": "outlier_zscore",
                "primary_metric": primary_metric,
                "secondary_metric": "z_score",
                "primary_dim": info_cols[0],
                "filters": filter_explanations,
                "explanation": f"Statistical outlier detection on {primary_metric} with Z-Score > 2.0."
            }

        # Case 0H: Above / Below Benchmark (Subquery Filtering)
        if (is_above_avg or is_below_avg) and primary_metric:
            op = ">" if is_above_avg else "<"
            op_name = "Above" if is_above_avg else "Below"
            info_cols = cls.get_informative_columns(df, metadata, primary_metric)
            sel_cols = ", ".join([f'"{c}"' for c in info_cols[:3]])
            sub_where = f'{where_clause} AND ' if where_clause else 'WHERE '
            sql = (
                f'SELECT\n'
                f'  {sel_cols},\n'
                f'  "{primary_metric}",\n'
                f'  ROUND((SELECT AVG("{primary_metric}") FROM "{table}"), 2) AS benchmark_average\n'
                f'FROM "{table}"\n'
                f'{sub_where}"{primary_metric}" {op} (SELECT AVG("{primary_metric}") FROM "{table}")\n'
                f'ORDER BY "{primary_metric}" {"DESC" if is_above_avg else "ASC"}\n'
                f'LIMIT 25;'
            )
            return {
                "sql": sql,
                "intent": "above_below_average",
                "primary_metric": primary_metric,
                "secondary_metric": "benchmark_average",
                "primary_dim": info_cols[0],
                "filters": filter_explanations,
                "explanation": f"Records with {primary_metric} {op_name.lower()} overall dataset benchmark average."
            }

        # Case 0I: Percentage of Total / Contribution Share
        if is_pct_of_total and primary_metric:
            target_dim = primary_dim or (cat_cols[0] if cat_cols else all_cols[0])
            clean_m = primary_metric.replace(" ", "_")
            sql = (
                f'SELECT\n'
                f'  "{target_dim}",\n'
                f'  ROUND(SUM("{primary_metric}"), 2) AS total_{clean_m},\n'
                f'  ROUND(100.0 * SUM("{primary_metric}") / NULLIF(SUM(SUM("{primary_metric}")) OVER (), 0), 2) AS pct_of_total\n'
                f'FROM "{table}"{where_clause}\n'
                f'GROUP BY "{target_dim}"\n'
                f'ORDER BY total_{clean_m} DESC\n'
                f'LIMIT 20;'
            )
            return {
                "sql": sql,
                "intent": "percent_of_total",
                "primary_metric": f"total_{clean_m}",
                "secondary_metric": "pct_of_total",
                "primary_dim": target_dim,
                "filters": filter_explanations,
                "explanation": f"Percentage contribution of {target_dim} to total {primary_metric}."
            }

        # Case 0J: Shipping Duration / Date-Difference Analysis
        if is_duration and len(date_cols) >= 2:
            d1, d2 = date_cols[0], date_cols[1]
            target_dim = primary_dim or (cat_cols[0] if cat_cols else all_cols[0])
            sql = (
                f'SELECT\n'
                f'  "{target_dim}",\n'
                f'  ROUND(AVG(DATEDIFF(\'day\', TRY_CAST("{d1}" AS TIMESTAMP), TRY_CAST("{d2}" AS TIMESTAMP))), 1) AS avg_duration_days,\n'
                f'  COUNT(*) AS total_records\n'
                f'FROM "{table}"{where_clause}\n'
                f'WHERE "{d1}" IS NOT NULL AND "{d2}" IS NOT NULL\n'
                f'GROUP BY "{target_dim}"\n'
                f'ORDER BY avg_duration_days ASC;'
            )
            return {
                "sql": sql,
                "intent": "shipping_duration",
                "primary_metric": "avg_duration_days",
                "secondary_metric": "total_records",
                "primary_dim": target_dim,
                "filters": filter_explanations,
                "explanation": f"Average duration in days between {d1} and {d2} across {target_dim}."
            }

        # Case 0K: Grouping with HAVING threshold
        if is_having and primary_dim and primary_metric:
            clean_m = primary_metric.replace(" ", "_")
            thresh_match = re.search(r'\b(greater than|more than|over|>|>=)\s*([0-9\.]+)\b', q_low)
            thresh_val = float(thresh_match.group(2)) if thresh_match else 1000.0
            sql = (
                f'SELECT\n'
                f'  "{primary_dim}",\n'
                f'  ROUND(SUM("{primary_metric}"), 2) AS total_{clean_m},\n'
                f'  COUNT(*) AS total_records\n'
                f'FROM "{table}"{where_clause}\n'
                f'GROUP BY "{primary_dim}"\n'
                f'HAVING SUM("{primary_metric}") > {thresh_val}\n'
                f'ORDER BY total_{clean_m} DESC;'
            )
            return {
                "sql": sql,
                "intent": "having_threshold",
                "primary_metric": f"total_{clean_m}",
                "secondary_metric": "total_records",
                "primary_dim": primary_dim,
                "filters": filter_explanations,
                "explanation": f"{primary_dim} performance filtered with HAVING threshold > {thresh_val}."
            }

        # Case 1: Simple Row Count
        if is_count_only:
            sql = f'SELECT COUNT(*) AS total_rows FROM "{table}"{where_clause};'
            return {
                "sql": sql,
                "intent": "scalar_count",
                "primary_metric": "total_rows",
                "primary_dim": None,
                "filters": filter_explanations,
                "explanation": "Calculates total matching records."
            }

        # Case 2: Distinct Entity Count
        if is_distinct_count:
            target_entity = primary_dim or (metadata["id_cols"][0] if metadata["id_cols"] else all_cols[0])
            for id_c in metadata["id_cols"] + cat_cols:
                if any(w in str(id_c).lower() for w in ["customer", "client", "user", "order", "product"]):
                    target_entity = id_c
                    break
            sql = f'SELECT COUNT(DISTINCT "{target_entity}") AS unique_{str(target_entity).replace(" ", "_")}\nFROM "{table}"{where_clause};'
            return {
                "sql": sql,
                "intent": "distinct_count",
                "primary_metric": f"unique_{target_entity}",
                "primary_dim": target_entity,
                "filters": filter_explanations,
                "explanation": f"Counts distinct occurrences of {target_entity}."
            }

        # Case 3: Time Series / Trend
        if is_timeseries and primary_metric:
            d_col = date_cols[0]
            is_yearly = any(k in q_low for k in ["by year", "yearly", "annual", "per year"])
            is_quarterly = any(k in q_low for k in ["quarter", "quarterly"])
            
            if is_yearly:
                period_expr = f'EXTRACT(YEAR FROM TRY_CAST("{d_col}" AS TIMESTAMP))'
                period_alias = "order_year"
            elif is_quarterly:
                period_expr = f'DATE_TRUNC(\'quarter\', TRY_CAST("{d_col}" AS TIMESTAMP))'
                period_alias = "quarter"
            else:
                period_expr = f'DATE_TRUNC(\'month\', TRY_CAST("{d_col}" AS TIMESTAMP))'
                period_alias = "month"

            metric_aggregates = [f'ROUND(SUM("{primary_metric}"), 2) AS total_{primary_metric.replace(" ", "_")}']
            if secondary_metric and has_multi_metric:
                metric_aggregates.append(f'ROUND(SUM("{secondary_metric}"), 2) AS total_{secondary_metric.replace(" ", "_")}')
            metric_aggregates.append('COUNT(*) AS orders_count')

            sql = f'SELECT {period_expr} AS {period_alias},\n  ' + ",\n  ".join(metric_aggregates) + f'\nFROM "{table}"{where_clause}\nWHERE {period_alias} IS NOT NULL\nGROUP BY {period_alias}\nORDER BY {period_alias} ASC;'
            # If where_clause already exists, join with AND
            if where_clause:
                sql = f'SELECT {period_expr} AS {period_alias},\n  ' + ",\n  ".join(metric_aggregates) + f'\nFROM "{table}"{where_clause} AND {period_expr} IS NOT NULL\nGROUP BY {period_alias}\nORDER BY {period_alias} ASC;'

            return {
                "sql": sql,
                "intent": "time_series",
                "primary_metric": primary_metric,
                "secondary_metric": secondary_metric if has_multi_metric else None,
                "primary_dim": period_alias,
                "filters": filter_explanations,
                "explanation": f"Chronological trend of {primary_metric} grouped by {period_alias}."
            }

        # Case 4: Top N or Bottom N Ranking
        if is_top or is_bottom:
            order_dir = "ASC" if is_bottom else "DESC"

            # Subcase 4A: Grouped Top N (e.g. "top 3 categories in west region", "top 5 products by profit")
            if primary_dim and primary_metric:
                agg_func = "AVG" if any(k in q_low for k in ["average", "avg", "mean"]) else "SUM"
                metric_alias = f'{"avg" if agg_func == "AVG" else "total"}_{primary_metric.replace(" ", "_")}'

                metric_exprs = [f'ROUND({agg_func}("{primary_metric}"), 2) AS {metric_alias}']
                if secondary_metric and has_multi_metric:
                    sec_alias = f'total_{secondary_metric.replace(" ", "_")}'
                    metric_exprs.append(f'ROUND(SUM("{secondary_metric}"), 2) AS {sec_alias}')
                metric_exprs.append('COUNT(*) AS order_count')

                sql = f'SELECT "{primary_dim}",\n  ' + ",\n  ".join(metric_exprs) + f'\nFROM "{table}"{where_clause}\nGROUP BY "{primary_dim}"\nORDER BY {metric_alias} {order_dir}\nLIMIT {limit_n};'

                return {
                    "sql": sql,
                    "intent": "ranking",
                    "primary_metric": metric_alias,
                    "secondary_metric": f"total_{secondary_metric.replace(' ', '_')}" if (secondary_metric and has_multi_metric) else None,
                    "primary_dim": primary_dim,
                    "limit": limit_n,
                    "order": order_dir,
                    "filters": filter_explanations,
                    "explanation": f"{'Top' if order_dir == 'DESC' else 'Bottom'} {limit_n} {primary_dim} ranked by {agg_func} of {primary_metric}."
                }

            # Subcase 4B: Individual Record Top N (e.g. "top 3 highest sales in west region", "top 5 sales")
            elif primary_metric:
                info_cols = cls.get_informative_columns(df, metadata, primary_metric, secondary_metric)
                select_clause = ",\n  ".join([f'"{c}"' for c in info_cols])
                sql = f'SELECT\n  {select_clause}\nFROM "{table}"{where_clause}\nORDER BY "{primary_metric}" {order_dir}\nLIMIT {limit_n};'

                desc_col = None
                for c in info_cols:
                    if not pd.api.types.is_numeric_dtype(df[c]) and any(w in str(c).lower() for w in ["name", "title", "product", "customer", "item"]):
                        desc_col = c
                        break
                if not desc_col:
                    desc_col = info_cols[1] if len(info_cols) > 1 else info_cols[0]

                return {
                    "sql": sql,
                    "intent": "individual_top_records",
                    "primary_metric": primary_metric,
                    "secondary_metric": secondary_metric,
                    "primary_dim": desc_col,
                    "limit": limit_n,
                    "order": order_dir,
                    "filters": filter_explanations,
                    "info_cols": info_cols,
                    "explanation": f"{'Top' if order_dir == 'DESC' else 'Bottom'} {limit_n} highest {primary_metric} records."
                }

        # Case 5: Multi-Dimension Breakdown (e.g. Sales by Category and Region)
        if primary_dim and secondary_dim and primary_metric and any(k in q_low for k in ["by", "per", "each", "across"]):
            metric_alias = f'total_{primary_metric.replace(" ", "_")}'
            sql = f'SELECT "{primary_dim}", "{secondary_dim}",\n  ROUND(SUM("{primary_metric}"), 2) AS {metric_alias},\n  COUNT(*) AS records\nFROM "{table}"{where_clause}\nGROUP BY "{primary_dim}", "{secondary_dim}"\nORDER BY "{primary_dim}" ASC, {metric_alias} DESC\nLIMIT 30;'
            return {
                "sql": sql,
                "intent": "multi_dimension",
                "primary_metric": metric_alias,
                "primary_dim": primary_dim,
                "secondary_dim": secondary_dim,
                "filters": filter_explanations,
                "explanation": f"Cross-dimensional breakdown of {primary_metric} across {primary_dim} and {secondary_dim}."
            }

        # Case 6: Group Aggregation (by Dimension)
        if primary_dim and primary_metric and (any(k in q_low for k in ["by", "per", "each", "across", "for each", "breakdown", "distribution"]) or not where_clause):
            agg_func = "AVG" if any(k in q_low for k in ["average", "avg", "mean"]) else "SUM"
            metric_alias = f'{"avg" if agg_func == "AVG" else "total"}_{primary_metric.replace(" ", "_")}'

            metric_exprs = [f'ROUND({agg_func}("{primary_metric}"), 2) AS {metric_alias}']
            if secondary_metric and has_multi_metric:
                sec_alias = f'total_{secondary_metric.replace(" ", "_")}'
                metric_exprs.append(f'ROUND(SUM("{secondary_metric}"), 2) AS {sec_alias}')
            metric_exprs.append('COUNT(*) AS record_count')

            sql = f'SELECT "{primary_dim}",\n  ' + ",\n  ".join(metric_exprs) + f'\nFROM "{table}"{where_clause}\nGROUP BY "{primary_dim}"\nORDER BY {metric_alias} DESC\nLIMIT 20;'

            return {
                "sql": sql,
                "intent": "group_aggregation",
                "primary_metric": metric_alias,
                "secondary_metric": f"total_{secondary_metric.replace(' ', '_')}" if (secondary_metric and has_multi_metric) else None,
                "primary_dim": primary_dim,
                "filters": filter_explanations,
                "explanation": f"Aggregation of {primary_metric} grouped by {primary_dim}."
            }

        # Case 7: Filtered Summary / Scalar Totals (e.g. "What is total sales in California?")
        if primary_metric:
            select_parts = [
                f'ROUND(SUM("{primary_metric}"), 2) AS total_{primary_metric.replace(" ", "_")}',
                f'ROUND(AVG("{primary_metric}"), 2) AS avg_{primary_metric.replace(" ", "_")}'
            ]
            if secondary_metric:
                select_parts.append(f'ROUND(SUM("{secondary_metric}"), 2) AS total_{secondary_metric.replace(" ", "_")}')
            select_parts.append('COUNT(*) AS total_records')

            sql = f'SELECT\n  ' + ",\n  ".join(select_parts) + f'\nFROM "{table}"{where_clause};'

            return {
                "sql": sql,
                "intent": "scalar_summary",
                "primary_metric": f'total_{primary_metric.replace(" ", "_")}',
                "secondary_metric": f'total_{secondary_metric.replace(" ", "_")}' if secondary_metric else None,
                "primary_dim": None,
                "filters": filter_explanations,
                "explanation": f"Executive summary of {primary_metric}" + (f" with applied filters" if where_clause else ".")
            }

        # Default Fallback: Safe preview
        sql = f'SELECT * FROM "{table}"{where_clause} LIMIT 20;'
        return {
            "sql": sql,
            "intent": "preview",
            "primary_metric": None,
            "primary_dim": None,
            "filters": filter_explanations,
            "explanation": "Data preview query."
        }

    @classmethod
    def execute_with_self_healing(
        cls,
        con: duckdb.DuckDBPyConnection,
        sql: str,
        df: pd.DataFrame,
        table: str = "dataset",
        max_repairs: int = 2
    ) -> Dict[str, Any]:
        """
        Executes query on DuckDB with self-healing repairs for column mismatch or date type casts.
        """
        current_sql = sql
        for attempt in range(max_repairs + 1):
            is_safe, error_msg = SQLAnalyticsModel.validate_sql_security(current_sql)
            if not is_safe:
                return {
                    "success": False,
                    "error": error_msg,
                    "dataframe": None,
                    "sql": current_sql,
                    "execution_time_ms": 0.0,
                    "row_count": 0
                }

            start_t = time.perf_counter()
            try:
                result_df = con.execute(current_sql).df()
                elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
                return {
                    "success": True,
                    "dataframe": result_df,
                    "sql": current_sql,
                    "execution_time_ms": elapsed_ms,
                    "row_count": len(result_df),
                    "columns": list(result_df.columns),
                    "error": None
                }
            except Exception as e:
                err_str = str(e)
                if attempt == max_repairs:
                    return {
                        "success": False,
                        "error": err_str,
                        "dataframe": None,
                        "sql": current_sql,
                        "execution_time_ms": round((time.perf_counter() - start_t) * 1000, 2),
                        "row_count": 0
                    }

                # Attempt Query Repair:
                # 1. Missing Column Error: "Binder Error: Referenced column "xxx" not found"
                col_match = re.search(r'Referenced column ["\']([^"\']+)["\'] not found', err_str)
                if col_match:
                    bad_col = col_match.group(1)
                    closest = difflib.get_close_matches(bad_col, df.columns, n=1, cutoff=0.5)
                    if closest:
                        current_sql = current_sql.replace(f'"{bad_col}"', f'"{closest[0]}"')
                        continue

                # 2. Date conversion error: Replace CAST with TRY_CAST
                if "Conversion Error" in err_str or "date" in err_str.lower():
                    if "CAST(" in current_sql and "TRY_CAST(" not in current_sql:
                        current_sql = current_sql.replace("CAST(", "TRY_CAST(")
                        continue

                # If no specific repair identified, break and report
                return {
                    "success": False,
                    "error": err_str,
                    "dataframe": None,
                    "sql": current_sql,
                    "execution_time_ms": round((time.perf_counter() - start_t) * 1000, 2),
                    "row_count": 0
                }

    @classmethod
    def synthesize_answer(
        cls,
        question: str,
        result_df: pd.DataFrame,
        intent_info: Dict[str, Any],
        source_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Transforms the query execution result into a clear, direct, human-readable executive answer,
        complete with highlighted metrics and recommended chart configuration.
        """
        if result_df is None or result_df.empty:
            return {
                "answer_markdown": "⚠️ No records matched your query criteria.",
                "kpi_cards": [],
                "chart_spec": None
            }

        intent = intent_info.get("intent", "generic")
        cols = list(result_df.columns)
        num_rows = len(result_df)
        kpi_cards = []
        answer_parts = []
        chart_spec = None

        # ── Formatter Helpers ──
        def fmt_val(v: Any, col_name: str = "") -> str:
            if pd.isna(v):
                return "N/A"
            try:
                num = float(v)
                c_low = col_name.lower()
                is_curr = any(k in c_low for k in ["sales", "profit", "revenue", "cost", "price", "amount"])
                is_pct = any(k in c_low for k in ["pct", "percent", "margin", "discount", "rate"]) and num <= 100.0

                if is_curr:
                    return f"${num:,.2f}" if abs(num) >= 1 else f"${num:.4f}"
                if is_pct:
                    return f"{num:.1f}%"
                if num.is_integer() and abs(num) >= 1000:
                    return f"{int(num):,}"
                return f"{num:,.2f}" if abs(num) >= 10 else f"{num:.2f}"
            except Exception:
                return str(v)

        # Case 0: Individual Record Drill-down (e.g. "top 3 highest sales in west region")
        if intent == "individual_top_records" and num_rows >= 1:
            pm = intent_info.get("primary_metric")
            if not pm or pm not in result_df.columns:
                num_cs = [c for c in cols if pd.api.types.is_numeric_dtype(result_df[c])]
                pm = num_cs[0] if num_cs else cols[-1]

            sec_m = intent_info.get("secondary_metric")
            if sec_m and sec_m not in result_df.columns:
                sec_m = None

            is_asc = intent_info.get("order") == "ASC"
            rank_title = "Lowest" if is_asc else "Highest"

            filters = intent_info.get("filters", [])
            filter_desc = ""
            if filters:
                parts = [f"**{f.get('column')}** = '{f.get('value', f.get('year', ''))}'" for f in filters if 'column' in f]
                filter_desc = f" in {', '.join(parts)}"

            # Find best descriptive column for labeling (e.g. Product Name, Customer Name)
            desc_col = intent_info.get("primary_dim")
            if not desc_col or desc_col not in result_df.columns or pd.api.types.is_numeric_dtype(result_df[desc_col]):
                for c in result_df.columns:
                    if not pd.api.types.is_numeric_dtype(result_df[c]) and any(w in str(c).lower() for w in ["name", "title", "product", "customer", "item"]):
                        desc_col = c
                        break
            if not desc_col:
                str_cs = [c for c in cols if not pd.api.types.is_numeric_dtype(result_df[c])]
                desc_col = str_cs[0] if str_cs else cols[0]

            top_row = result_df.iloc[0]
            top_val = top_row[pm] if pm in top_row else 0
            top_desc = str(top_row[desc_col]) if desc_col in top_row else "Top Record"

            # KPI Cards
            kpi_cards.append({
                "label": f"#1 {pm.replace('_', ' ').title()}",
                "value": fmt_val(top_val, pm),
                "subtext": top_desc[:30]
            })

            combined_sum = result_df[pm].sum() if pm in result_df and pd.api.types.is_numeric_dtype(result_df[pm]) else 0
            kpi_cards.append({
                "label": f"Combined Top {num_rows} {pm.replace('_', ' ').title()}",
                "value": fmt_val(combined_sum, pm),
                "subtext": f"Sum of top {num_rows} records"
            })

            if sec_m and sec_m in result_df and pd.api.types.is_numeric_dtype(result_df[sec_m]):
                kpi_cards.append({
                    "label": f"Combined {sec_m.replace('_', ' ').title()}",
                    "value": fmt_val(result_df[sec_m].sum(), sec_m),
                    "subtext": f"#1: {fmt_val(top_row[sec_m], sec_m)}"
                })

            kpi_cards.append({
                "label": "Records Displayed",
                "value": f"{num_rows:,}",
                "subtext": f"Ranked observations"
            })

            # Narrative list
            answer_parts.append(f"🏆 **Top {num_rows} {rank_title} {pm.replace('_', ' ').title()} records{filter_desc}:**<br>")
            for i in range(num_rows):
                r = result_df.iloc[i]
                r_desc = str(r[desc_col]) if desc_col in r else f"Record #{i+1}"
                r_val = fmt_val(r[pm], pm)
                
                context_tags = []
                for ec in result_df.columns:
                    if ec != desc_col and ec != pm and ec in r and not pd.isna(r[ec]):
                        ec_low = str(ec).lower()
                        if any(w in ec_low for w in ["product", "item", "customer", "category", "city", "state", "type"]):
                            context_tags.append(f"{ec}: {r[ec]}")
                tag_str = f" ({', '.join(context_tags[:2])})" if context_tags else ""

                answer_parts.append(f"• **#{i+1}: {r_desc}** — **{r_val}**{tag_str}<br>")

            answer_parts.append(f"<br>💡 Combined {pm.replace('_', ' ')} for these {num_rows} records is **{fmt_val(combined_sum, pm)}**.")

            clean_f_desc = filter_desc.replace("**", "").strip()
            chart_spec = {
                "type": "bar",
                "x": desc_col,
                "y": pm,
                "metrics": [pm],
                "title": f"Top {num_rows} {pm.replace('_', ' ').title()} Records" + (f" ({clean_f_desc})" if clean_f_desc else "")
            }

        # Case: Period-over-Period Growth (MoM / YoY)
        elif intent == "period_over_period" and num_rows >= 2:
            p_col = cols[0]
            cur_col = cols[1]
            growth_col = cols[-1]
            last_row = result_df.iloc[-1]
            prev_row = result_df.iloc[-2]

            latest_p = str(last_row[p_col])[:10]
            latest_val = last_row[cur_col]
            latest_growth = last_row[growth_col]

            kpi_cards.append({
                "label": f"Latest Period ({latest_p})",
                "value": fmt_val(latest_val, cur_col),
                "subtext": "Current Period Volume"
            })
            kpi_cards.append({
                "label": "Latest Growth Rate",
                "value": f"{'+' if pd.notna(latest_growth) and latest_growth > 0 else ''}{latest_growth:.1f}%" if pd.notna(latest_growth) else "N/A",
                "subtext": f"vs prior period ({fmt_val(prev_row[cur_col], cur_col)})"
            })
            valid_growths = result_df[growth_col].dropna()
            avg_growth = valid_growths.mean() if not valid_growths.empty else 0.0
            kpi_cards.append({
                "label": "Average Growth Rate",
                "value": f"{'+' if avg_growth > 0 else ''}{avg_growth:.1f}%",
                "subtext": f"Across {num_rows} historical periods"
            })
            kpi_cards.append({
                "label": "Total Tracked Volume",
                "value": fmt_val(result_df[cur_col].sum(), cur_col),
                "subtext": "Cumulative period sum"
            })

            growth_sign = "expanded" if pd.notna(latest_growth) and latest_growth > 0 else "contracted"
            abs_g = abs(latest_growth) if pd.notna(latest_growth) else 0.0
            answer_parts.append(
                f"📈 **Period-over-Period Growth Trajectory:** In the latest period (**{latest_p}**), "
                f"volume reached **{fmt_val(latest_val, cur_col)}**, which **{growth_sign} by {abs_g:.1f}%** "
                f"compared to **{fmt_val(prev_row[cur_col], cur_col)}** in the preceding period."
            )
            chart_spec = {
                "type": "line",
                "x": p_col,
                "y": cur_col,
                "metrics": [cur_col],
                "title": f"Growth & Trajectory of {cur_col.replace('_', ' ').title()}"
            }

        # Case: Cumulative Running Total
        elif intent == "running_total" and num_rows >= 1:
            p_col = cols[0]
            period_col = cols[1]
            cum_col = cols[2] if len(cols) > 2 else cols[-1]
            final_cum = result_df.iloc[-1][cum_col]

            kpi_cards.append({
                "label": "Final Cumulative Total",
                "value": fmt_val(final_cum, cum_col),
                "subtext": "Total accumulated volume"
            })
            kpi_cards.append({
                "label": "Peak Single Period",
                "value": fmt_val(result_df[period_col].max(), period_col),
                "subtext": "Maximum periodic contribution"
            })
            kpi_cards.append({
                "label": "Total Periods Tracked",
                "value": f"{num_rows:,}",
                "subtext": "Historical time increments"
            })

            answer_parts.append(
                f"📈 **Cumulative Progression:** Across **{num_rows}** recorded periods, "
                f"cumulative volume steadily advanced to a grand total of **{fmt_val(final_cum, cum_col)}**."
            )
            chart_spec = {
                "type": "line",
                "x": p_col,
                "y": cum_col,
                "metrics": [cum_col, period_col],
                "title": "Cumulative Progression vs Periodic Volume"
            }

        # Case: Pareto 80/20 Cumulative Distribution
        elif intent == "pareto_80_20" and num_rows >= 1:
            dim_c = cols[0]
            val_c = cols[1]
            cum_c = cols[2] if len(cols) > 2 else cols[-1]

            top_row = result_df.iloc[0]
            under_80_cnt = int((result_df[cum_c] <= 80.0).sum())

            kpi_cards.append({
                "label": f"#1 {dim_c.replace('_', ' ').title()}",
                "value": str(top_row[dim_c])[:25],
                "subtext": f"{fmt_val(top_row[val_c], val_c)} ({top_row[cum_c]:.1f}% cumulative)"
            })
            kpi_cards.append({
                "label": "80% Revenue Concentration",
                "value": f"{max(1, under_80_cnt)} Entities",
                "subtext": "Drive ~80% of entire volume"
            })
            kpi_cards.append({
                "label": "Total Ranked Entities",
                "value": f"{num_rows:,}",
                "subtext": "In top Pareto sample"
            })

            answer_parts.append(
                f"⚖️ **Pareto 80/20 Distribution:** **{str(top_row[dim_c])}** ranks #1 with "
                f"**{fmt_val(top_row[val_c], val_c)}**, accounting for **{top_row[cum_c]:.1f}%** of total volume. "
                f"The top **{max(1, under_80_cnt)}** entities drive the majority (~80%) of entire aggregate revenue."
            )
            chart_spec = {
                "type": "bar",
                "x": dim_c,
                "y": val_c,
                "metrics": [val_c],
                "title": f"Pareto 80/20 Analysis by {dim_c.replace('_', ' ').title()}"
            }

        # Case: RFM Customer Segmentation
        elif intent == "rfm_segmentation" and num_rows >= 1:
            cust_c = cols[0]
            freq_c = cols[1]
            mon_c = cols[2]
            tier_c = cols[-1]

            tier_counts = result_df[tier_c].value_counts()
            top_tier = tier_counts.index[0] if not tier_counts.empty else "Standard"

            kpi_cards.append({
                "label": "Dominant Segment",
                "value": str(top_tier),
                "subtext": f"{tier_counts.iloc[0]} customers in cohort"
            })
            kpi_cards.append({
                "label": "Top Customer",
                "value": str(result_df.iloc[0][cust_c])[:25],
                "subtext": f"{fmt_val(result_df.iloc[0][mon_c], mon_c)} ({result_df.iloc[0][freq_c]} orders)"
            })
            kpi_cards.append({
                "label": "Cohort Total Spend",
                "value": fmt_val(result_df[mon_c].sum(), mon_c),
                "subtext": f"Across top {num_rows} customers"
            })

            answer_parts.append(
                f"💎 **RFM Customer Segmentation:** Segmented customers based on order frequency and monetary value. "
                f"The leading customer is **{result_df.iloc[0][cust_c]}** with **{fmt_val(result_df.iloc[0][mon_c], mon_c)}** across **{result_df.iloc[0][freq_c]} orders**. "
                f"The dominant tier is **{top_tier}** ({tier_counts.iloc[0]} accounts)."
            )
            chart_spec = {
                "type": "bar",
                "x": cust_c,
                "y": mon_c,
                "metrics": [mon_c],
                "title": "Top Customers by Monetary Value & Segment Tier"
            }

        # Case: Statistical Z-Score Outliers
        elif intent == "outlier_zscore" and num_rows >= 1:
            z_c = "z_score"
            max_z = result_df[z_c].max() if z_c in result_df else 0.0
            top_outlier = result_df.iloc[0]
            target_metric = intent_info.get("primary_metric", cols[-2])

            kpi_cards.append({
                "label": "Detected Anomalies",
                "value": f"{num_rows:,} Records",
                "subtext": "Z-Score > 2.0 standard deviations"
            })
            kpi_cards.append({
                "label": "Peak Anomaly Z-Score",
                "value": f"{max_z:.2f} σ",
                "subtext": "Standard deviations from mean"
            })
            kpi_cards.append({
                "label": "Peak Outlier Value",
                "value": fmt_val(top_outlier[target_metric], target_metric) if target_metric in top_outlier else "N/A",
                "subtext": "Extreme observation"
            })

            answer_parts.append(
                f"🚨 **Statistical Outlier Radar:** Identified **{num_rows:,} anomalous transactions** exceeding 2.0 standard deviations. "
                f"The most extreme observation has a Z-Score of **{max_z:.2f} σ** with **{target_metric} = {fmt_val(top_outlier[target_metric], target_metric) if target_metric in top_outlier else 'N/A'}**."
            )
            chart_spec = {
                "type": "bar",
                "x": cols[0],
                "y": z_c,
                "metrics": [z_c],
                "title": "Statistical Outliers by Z-Score Deviation"
            }

        # Case: Partitioned Window Ranking (Top N within each group)
        elif intent == "window_ranking_per_group" and num_rows >= 1:
            grp_c = cols[0]
            item_c = cols[1]
            val_c = cols[2]

            unique_grps = result_df[grp_c].nunique()
            top_1 = result_df.iloc[0]

            kpi_cards.append({
                "label": "Categories / Groups",
                "value": f"{unique_grps:,}",
                "subtext": "Partitioned dimensions"
            })
            kpi_cards.append({
                "label": f"Overall Top {item_c.replace('_', ' ').title()}",
                "value": str(top_1[item_c])[:25],
                "subtext": f"{fmt_val(top_1[val_c], val_c)} in {top_1[grp_c]}"
            })
            kpi_cards.append({
                "label": "Total Ranked Volume",
                "value": fmt_val(result_df[val_c].sum(), val_c),
                "subtext": f"Across {num_rows} ranked items"
            })

            answer_parts.append(
                f"🏆 **Partitioned Ranking by Group:** Extracted the top-ranking **{item_c.replace('_', ' ')}** observations within each **{grp_c.replace('_', ' ')}**. "
                f"Overall leader is **{top_1[item_c]}** in **{top_1[grp_c]}** generating **{fmt_val(top_1[val_c], val_c)}**."
            )
            chart_spec = {
                "type": "bar",
                "x": item_c,
                "y": val_c,
                "metrics": [val_c],
                "title": f"Top {item_c.replace('_', ' ').title()} across {grp_c.replace('_', ' ').title()}"
            }

        # Case: Percentage of Total / Contribution Share
        elif intent == "percent_of_total" and num_rows >= 1:
            dim_c = cols[0]
            val_c = cols[1]
            pct_c = cols[2] if len(cols) > 2 else cols[-1]
            top_1 = result_df.iloc[0]

            kpi_cards.append({
                "label": f"#1 {dim_c.replace('_', ' ').title()}",
                "value": str(top_1[dim_c])[:25],
                "subtext": f"{top_1[pct_c]:.1f}% contribution share"
            })
            kpi_cards.append({
                "label": f"Top {val_c.replace('_', ' ').title()}",
                "value": fmt_val(top_1[val_c], val_c),
                "subtext": "Absolute volume"
            })
            kpi_cards.append({
                "label": "Total Volume",
                "value": fmt_val(result_df[val_c].sum(), val_c),
                "subtext": f"Across all {num_rows} categories"
            })

            answer_parts.append(
                f"🥧 **Contribution Share Analysis:** **{top_1[dim_c]}** leads with **{fmt_val(top_1[val_c], val_c)}**, "
                f"representing **{top_1[pct_c]:.1f}% of total volume**."
            )
            chart_spec = {
                "type": "donut",
                "names": dim_c,
                "values": val_c,
                "title": f"Share of {val_c.replace('_', ' ').title()} by {dim_c.replace('_', ' ').title()}"
            }

        # Case 1: Scalar Count or Single Row Summary
        elif num_rows == 1:
            row = result_df.iloc[0]
            metric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(result_df[c])]

            for c in metric_cols:
                kpi_cards.append({
                    "label": c.replace("_", " ").title(),
                    "value": fmt_val(row[c], c),
                    "subtext": "Calculated Metric"
                })

            # Check if there are active filters in question
            filters = intent_info.get("filters", [])
            filter_desc = ""
            if filters:
                parts = [f"**{f.get('column')}** = '{f.get('value', f.get('year', ''))}'" for f in filters if 'column' in f]
                filter_desc = f" for {', '.join(parts)}"

            if len(metric_cols) >= 2:
                m1, m2 = metric_cols[0], metric_cols[1]
                answer_parts.append(
                    f"📊 **Result Summary{filter_desc}:** The total **{m1.replace('_', ' ')}** is "
                    f"**{fmt_val(row[m1], m1)}**, with **{m2.replace('_', ' ')}** of "
                    f"**{fmt_val(row[m2], m2)}**."
                )
            elif len(metric_cols) == 1:
                m = metric_cols[0]
                answer_parts.append(
                    f"🎯 **Result{filter_desc}:** The **{m.replace('_', ' ')}** is "
                    f"**{fmt_val(row[m], m)}**."
                )
            else:
                answer_parts.append(f"Query returned 1 record: {row.to_dict()}")

            chart_spec = {"type": "kpi_cards", "columns": metric_cols}

        # Case 2: Ranking / Group Aggregation (Dimension vs Metric)
        elif num_rows > 1 and len(cols) >= 2:
            dim_col = cols[0]
            num_metric_cols = [c for c in cols[1:] if pd.api.types.is_numeric_dtype(result_df[c])]

            if num_metric_cols:
                top_m = num_metric_cols[0]
                total_sum = result_df[top_m].sum() if pd.api.types.is_numeric_dtype(result_df[top_m]) else 0

                top_1 = result_df.iloc[0]
                top_val = top_1[top_m]
                top_name = str(top_1[dim_col])
                share_pct = (top_val / total_sum * 100) if total_sum > 0 else 0

                # Top Entity KPI Card
                kpi_cards.append({
                    "label": f"Top {dim_col.replace('_', ' ').title()}",
                    "value": top_name,
                    "subtext": f"{fmt_val(top_val, top_m)} ({share_pct:.1f}% of total)"
                })

                # Overall Metric Total KPI Card
                kpi_cards.append({
                    "label": f"Total {top_m.replace('_', ' ').title()}",
                    "value": fmt_val(total_sum, top_m),
                    "subtext": f"Across {num_rows} {dim_col.replace('_', ' ')} categories"
                })

                # If secondary metric exists (e.g. Profit or Count)
                if len(num_metric_cols) >= 2:
                    sec_m = num_metric_cols[1]
                    kpi_cards.append({
                        "label": f"Total {sec_m.replace('_', ' ').title()}",
                        "value": fmt_val(result_df[sec_m].sum(), sec_m),
                        "subtext": f"Top: {fmt_val(top_1[sec_m], sec_m)}"
                    })

                # Narrative Synthesis
                is_asc = intent_info.get("order") == "ASC"
                rank_word = "lowest" if is_asc else "highest"
                
                lead_text = f"🏆 **{top_name}** leads with the {rank_word} **{top_m.replace('_', ' ')}** of **{fmt_val(top_val, top_m)}**"
                if share_pct > 0:
                    lead_text += f" (accounting for **{share_pct:.1f}%** of the total {top_m.replace('_', ' ')})"
                lead_text += "."
                answer_parts.append(lead_text)

                # Next runners up
                runners = []
                for idx in range(1, min(4, num_rows)):
                    r = result_df.iloc[idx]
                    r_val = r[top_m]
                    r_share = (r_val / total_sum * 100) if total_sum > 0 else 0
                    runners.append(f"**{r[dim_col]}** ({fmt_val(r_val, top_m)}, {r_share:.1f}%)")

                if runners:
                    answer_parts.append(f"Following close behind are: {', '.join(runners)}.")

                # Recommended chart type
                is_time = any(k in str(dim_col).lower() for k in ["month", "year", "date", "quarter", "day"])
                if is_time:
                    chart_spec = {
                        "type": "line",
                        "x": dim_col,
                        "y": top_m,
                        "metrics": num_metric_cols,
                        "title": f"{top_m.replace('_', ' ').title()} Trend by {dim_col.replace('_', ' ').title()}"
                    }
                elif len(num_metric_cols) >= 2:
                    chart_spec = {
                        "type": "grouped_bar",
                        "x": dim_col,
                        "metrics": num_metric_cols[:3],
                        "title": f"Comparison across {dim_col.replace('_', ' ').title()}"
                    }
                else:
                    chart_spec = {
                        "type": "bar",
                        "x": dim_col,
                        "y": top_m,
                        "metrics": [top_m],
                        "title": f"{top_m.replace('_', ' ').title()} by {dim_col.replace('_', ' ').title()}"
                    }

        if not answer_parts:
            answer_parts.append(f"✅ Executed query successfully. Returned **{num_rows:,}** analytical rows.")

        return {
            "answer_markdown": " ".join(answer_parts),
            "kpi_cards": kpi_cards[:4],
            "chart_spec": chart_spec
        }

    @classmethod
    def answer_question(
        cls,
        question: str,
        df: pd.DataFrame,
        con: Optional[duckdb.DuckDBPyConnection] = None,
        table: str = "dataset"
    ) -> Dict[str, Any]:
        """
        Complete end-to-end pipeline:
        1. Translates natural language question into safe, accurate DuckDB SQL.
        2. Executes query on DuckDB with self-healing recovery.
        3. Synthesizes direct executive answer and KPI metric stat cards.
        4. Recommends tailored Plotly interactive visualizations.
        """
        if con is None:
            con = duckdb.connect(database=":memory:")
            con.register(table, df)

        # Step 1: Generate high-accuracy DuckDB SQL using SmartSQLEngine (instantaneous & deterministic)
        intent_info = cls.generate_smart_sql(question, df, table=table)
        generated_sql = intent_info["sql"]

        # Step 2: If intent was an unclassified preview, attempt external LLM if configured
        if intent_info.get("intent") == "preview":
            provider = LLMClient.get_configured_provider()
            if provider != "offline":
                try:
                    metadata = cls.extract_metadata(df)
                    col_types = [f'"{c}" ({df[c].dtype})' for c in df.columns[:25]]
                    schema_str = ", ".join(col_types)
                    prompt = (
                        f"You are a DuckDB SQL expert. Write a single analytical DuckDB SQL query to answer this question.\n"
                        f"TABLE: \"{table}\"\n"
                        f"COLUMNS: {schema_str}\n"
                        f"QUESTION: {question}\n"
                        f"RULES:\n"
                        f"1. Output ONLY the SQL string. No markdown, no backticks, no comments.\n"
                        f"2. Quote all column names with double quotes.\n"
                        f"3. Use standard DuckDB functions: DATE_TRUNC, EXTRACT, ROUND, SUM, AVG, COUNT.\n"
                        f"4. Limit large queries to 20 rows unless aggregated.\n"
                        f"SQL:"
                    )
                    res_llm = LLMClient.call_llm(prompt=prompt, max_tokens=300, temperature=0.0)
                    if res_llm.get("success") and res_llm.get("text"):
                        txt = res_llm["text"].strip()
                        if txt.startswith("```"):
                            lines = [l for l in txt.split("\n") if not l.strip().startswith("```")]
                            txt = "\n".join(lines).strip()
                        if not txt.endswith(";"):
                            txt += ";"
                        is_safe, _ = SQLAnalyticsModel.validate_sql_security(txt)
                        if is_safe:
                            generated_sql = txt
                except Exception:
                    pass

        # Step 3: Execute on DuckDB with self-healing
        exec_result = cls.execute_with_self_healing(con, generated_sql, df, table=table)

        # If LLM generated SQL failed execution, fall back immediately to SmartSQLEngine
        if not exec_result["success"] and intent_info == {}:
            intent_info = cls.generate_smart_sql(question, df, table=table)
            generated_sql = intent_info["sql"]
            exec_result = cls.execute_with_self_healing(con, generated_sql, df, table=table)

        # Step 4: Synthesize Executive Answer & Visualizations
        if exec_result["success"]:
            synth = cls.synthesize_answer(
                question=question,
                result_df=exec_result["dataframe"],
                intent_info=intent_info,
                source_df=df
            )
            exec_result["answer_markdown"] = synth["answer_markdown"]
            exec_result["kpi_cards"] = synth["kpi_cards"]
            exec_result["chart_spec"] = synth["chart_spec"]
            exec_result["intent_info"] = intent_info
        else:
            exec_result["answer_markdown"] = f"❌ Unable to execute SQL query: {exec_result.get('error')}"
            exec_result["kpi_cards"] = []
            exec_result["chart_spec"] = None
            exec_result["intent_info"] = intent_info

        return exec_result
