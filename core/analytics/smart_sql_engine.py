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
            c_low = c_str.lower()
            if any(k in c_low for k in ["_id", "id_", "identifier", "row_id", "row id", "index", "ticket"]):
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
        
        primary_dim = matched_cats[0] if matched_cats else (cls.resolve_column("category", cat_cols, "category") or (cat_cols[0] if cat_cols else None))
        secondary_dim = matched_cats[1] if len(matched_cats) > 1 else None

        # Step 2: Extract WHERE filters
        where_conds, filter_explanations = cls.extract_filters(q_clean, metadata)
        where_clause = f"\nWHERE {' AND '.join(where_conds)}" if where_conds else ""

        # Step 3: Classify Intent
        # A. Ranking / Top N / Bottom N
        is_top = any(k in q_low for k in ["top", "highest", "best", "most", "largest", "maximum", "max", "leader", "greatest", "peak"])
        is_bottom = any(k in q_low for k in ["bottom", "lowest", "worst", "least", "smallest", "minimum", "min", "laggard"])
        limit_match = re.search(r'\b(top|bottom|first|last|highest|lowest)\s*(\d+)\b', q_low)
        limit_n = int(limit_match.group(2)) if limit_match else (1 if ("which" in q_low or "what is the highest" in q_low or "what is the best" in q_low) else 10)
        if limit_n > 50:
            limit_n = 50

        # B. Time-series / Chronological
        is_timeseries = any(k in q_low for k in ["trend", "monthly", "by month", "over time", "history", "trajectory", "by year", "yearly", "quarterly", "daily"]) and bool(date_cols)

        # C. Counts / Row count / Distinct count
        is_count_only = any(k in q_low for k in ["how many rows", "total rows", "number of records", "row count", "how many records"])
        is_distinct_count = any(k in q_low for k in ["how many unique", "how many distinct", "distinct customers", "unique customers", "how many different"])

        # D. Summary / Scalar KPI (Total, Average, Overall)
        is_scalar_total = any(k in q_low for k in ["what is the total", "what is total", "overall", "entire", "grand total", "sum of all"]) and not any(k in q_low for k in ["by", "per", "each", "across"])

        # E. Multi-Metric (e.g. Sales AND Profit)
        has_multi_metric = len(matched_nums) >= 2 or (primary_metric and secondary_metric and any(k in q_low for k in ["and", "both", "versus", "vs", "compare", "with"]))

        # ── SYNTHESIS LOGIC ──

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

        # Case 4: Top N or Bottom N Ranking with Group By
        if (is_top or is_bottom) and primary_dim and primary_metric:
            order_dir = "ASC" if is_bottom else "DESC"
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

        # Case 1: Scalar Count or Single Row Summary
        if num_rows == 1:
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

        # Step 1: Check if external LLM can provide high-fidelity SQL
        # (with strict fallback to our deterministic engine)
        generated_sql = None
        intent_info = {}

        provider = LLMClient.get_configured_provider()
        if provider != "offline":
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
            try:
                res_llm = LLMClient.call_llm(prompt=prompt, max_tokens=300, temperature=0.0)
                if res_llm.get("success") and res_llm.get("text"):
                    txt = res_llm["text"].strip()
                    if txt.startswith("```"):
                        lines = [l for l in txt.split("\n") if not l.strip().startswith("```")]
                        txt = "\n".join(lines).strip()
                    if not txt.endswith(";"):
                        txt += ";"
                    # Validate basic security
                    is_safe, _ = SQLAnalyticsModel.validate_sql_security(txt)
                    if is_safe:
                        generated_sql = txt
            except Exception:
                pass

        # Step 2: If LLM didn't produce or is offline, use SmartSQLEngine
        if not generated_sql:
            intent_info = cls.generate_smart_sql(question, df, table=table)
            generated_sql = intent_info["sql"]

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
