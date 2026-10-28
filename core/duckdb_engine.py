import duckdb
import pandas as pd
import numpy as np
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from core.config import DUCKDB_PATH
from core.models.sql_model import SQLAnalyticsModel

class DuckDBEngine:
    """High-performance analytical engine operating directly on Parquet files,

    eliminating redundant Pandas DataFrame copies and powering exact aggregations,
    quantiles, correlations, and timeseries at scale.
    """

    _con: Optional[duckdb.DuckDBPyConnection] = None

    @classmethod
    def get_connection(cls) -> duckdb.DuckDBPyConnection:
        if cls._con is None:
            # Shared in-memory engine connection optimized for multithreaded analytical queries
            cls._con = duckdb.connect(database=":memory:")
            # Configure for high-performance on large datasets
            try:
                import os
                cpu_count = os.cpu_count() or 4
                threads = min(cpu_count, 8)
                cls._con.execute(f"PRAGMA threads={threads};")
                cls._con.execute("PRAGMA memory_limit='2GB';")
                cls._con.execute("PRAGMA enable_object_cache;")
            except Exception:
                pass
        return cls._con

    @classmethod
    def _clean_path(cls, path: Union[str, Path]) -> str:
        return str(path).replace("\\", "/")

    @classmethod
    def execute_query(
        cls,
        sql: str,
        parquet_path: Optional[Union[str, Path]] = None,
        table_alias: str = "dataset"
    ) -> Dict[str, Any]:
        """Executes analytical SQL query with security validation and optional parquet binding."""
        is_safe, err = SQLAnalyticsModel.validate_sql_security(sql)
        if not is_safe:
            return {
                "success": False,
                "error": err,
                "data": None,
                "execution_time_ms": 0.0,
                "row_count": 0
            }

        con = cls.get_connection()
        t0 = time.perf_counter()
        try:
            if parquet_path:
                clean_p = cls._clean_path(parquet_path)
                con.execute(f"CREATE OR REPLACE VIEW {table_alias} AS SELECT * FROM read_parquet('{clean_p}');")

            res_df = con.execute(sql).df()
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "success": True,
                "error": None,
                "data": res_df.to_dict(orient="records"),
                "dataframe": res_df,
                "execution_time_ms": elapsed_ms,
                "row_count": len(res_df),
                "columns": list(res_df.columns)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "row_count": 0
            }

    @classmethod
    def get_aggregates(
        cls,
        parquet_path: Union[str, Path],
        numeric_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculates single-pass exact statistical aggregates across millions of rows."""
        if not numeric_cols:
            return {}

        clean_p = cls._clean_path(parquet_path)
        select_parts = []
        for c in numeric_cols:
            safe_c = f'"{c}"'
            select_parts.extend([
                f'COUNT({safe_c}) AS "{c}__count"',
                f'AVG({safe_c}) AS "{c}__avg"',
                f'SUM({safe_c}) AS "{c}__sum"',
                f'MIN({safe_c}) AS "{c}__min"',
                f'MAX({safe_c}) AS "{c}__max"',
                f'STDDEV_SAMP({safe_c}) AS "{c}__std"'
            ])

        sql = f"SELECT {', '.join(select_parts)} FROM read_parquet('{clean_p}');"
        con = cls.get_connection()
        res = con.execute(sql).df().iloc[0]

        stats = {}
        for c in numeric_cols:
            stats[c] = {
                "count": int(res.get(f"{c}__count", 0)),
                "mean": float(res.get(f"{c}__avg", 0.0)) if pd.notnull(res.get(f"{c}__avg")) else 0.0,
                "sum": float(res.get(f"{c}__sum", 0.0)) if pd.notnull(res.get(f"{c}__sum")) else 0.0,
                "min": float(res.get(f"{c}__min", 0.0)) if pd.notnull(res.get(f"{c}__min")) else 0.0,
                "max": float(res.get(f"{c}__max", 0.0)) if pd.notnull(res.get(f"{c}__max")) else 0.0,
                "std": float(res.get(f"{c}__std", 0.0)) if pd.notnull(res.get(f"{c}__std")) else 0.0,
            }
        return stats

    @classmethod
    def get_categorical_frequency(
        cls,
        parquet_path: Union[str, Path],
        col: str,
        top_n: int = 30
    ) -> List[Dict[str, Any]]:
        """Calculates exact categorical value counts and proportions."""
        clean_p = cls._clean_path(parquet_path)
        safe_c = f'"{col}"'
        sql = f"""
        SELECT 
            COALESCE(CAST({safe_c} AS VARCHAR), 'NULL') AS category,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
        FROM read_parquet('{clean_p}')
        GROUP BY 1
        ORDER BY count DESC
        LIMIT {top_n};
        """
        con = cls.get_connection()
        return con.execute(sql).df().to_dict(orient="records")

    @classmethod
    def get_date_timeseries(
        cls,
        parquet_path: Union[str, Path],
        date_col: str,
        metric_col: str,
        granularity: str = "month"
    ) -> List[Dict[str, Any]]:
        """Calculates chronological trends rolled up to day, month, quarter, or year."""
        clean_p = cls._clean_path(parquet_path)
        safe_date = f'"{date_col}"'
        safe_metric = f'"{metric_col}"'

        sql = f"""
        SELECT 
            strftime(DATE_TRUNC('{granularity}', TRY_CAST({safe_date} AS TIMESTAMP)), '%Y-%m-%d') AS period,
            COUNT(*) AS transactions,
            SUM({safe_metric}) AS total,
            ROUND(AVG({safe_metric}), 2) AS average
        FROM read_parquet('{clean_p}')
        WHERE TRY_CAST({safe_date} AS TIMESTAMP) IS NOT NULL AND {safe_metric} IS NOT NULL
        GROUP BY 1
        ORDER BY 1 ASC;
        """
        con = cls.get_connection()
        return con.execute(sql).df().to_dict(orient="records")

    @classmethod
    def get_quantiles(
        cls,
        parquet_path: Union[str, Path],
        col: str,
        quantiles: List[float] = [0.25, 0.50, 0.75]
    ) -> Dict[float, float]:
        """Calculates exact or approximate quantiles without full in-memory Pandas sorting."""
        clean_p = cls._clean_path(parquet_path)
        safe_c = f'"{col}"'
        parts = [f"QUANTILE_CONT({safe_c}, {q}) AS q_{int(q*100)}" for q in quantiles]
        sql = f"SELECT {', '.join(parts)} FROM read_parquet('{clean_p}') WHERE {safe_c} IS NOT NULL;"
        con = cls.get_connection()
        row = con.execute(sql).df().iloc[0]
        return {q: float(row.get(f"q_{int(q*100)}", 0.0)) for q in quantiles}

    @classmethod
    def get_bivariate_aggregation(
        cls,
        parquet_path: Union[str, Path],
        group_col: str,
        metric_col: str,
        top_n: int = 25
    ) -> List[Dict[str, Any]]:
        """Calculates exact bivariate totals and averages for dashboard & EDA views."""
        clean_p = cls._clean_path(parquet_path)
        safe_g = f'"{group_col}"'
        safe_m = f'"{metric_col}"'

        sql = f"""
        SELECT 
            COALESCE(CAST({safe_g} AS VARCHAR), 'Unknown') AS label,
            SUM({safe_m}) AS total,
            ROUND(AVG({safe_m}), 2) AS average,
            COUNT(*) AS count
        FROM read_parquet('{clean_p}')
        WHERE {safe_m} IS NOT NULL
        GROUP BY 1
        ORDER BY total DESC
        LIMIT {top_n};
        """
        con = cls.get_connection()
        return con.execute(sql).df().to_dict(orient="records")

    @classmethod
    def get_correlation_matrix(
        cls,
        parquet_path: Union[str, Path],
        cols: List[str],
        sample_size: int = 30000
    ) -> pd.DataFrame:
        """Computes Pearson correlation matrix directly in DuckDB over full or sample data."""
        if len(cols) < 2:
            return pd.DataFrame()

        clean_p = cls._clean_path(parquet_path)
        con = cls.get_connection()

        # If columns <= 8, compute exact matrix using DuckDB corr() in a single query
        if len(cols) <= 8:
            select_parts = []
            for i, c1 in enumerate(cols):
                for j, c2 in enumerate(cols):
                    if i <= j:
                        select_parts.append(f'ROUND(CORR("{c1}", "{c2}"), 4) AS "corr__{c1}__{c2}"')
            sql = f"SELECT {', '.join(select_parts)} FROM read_parquet('{clean_p}');"
            try:
                row = con.execute(sql).df().iloc[0]
                matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
                for i, c1 in enumerate(cols):
                    for j, c2 in enumerate(cols):
                        if i <= j:
                            val = float(row.get(f"corr__{c1}__{c2}", 1.0 if c1 == c2 else 0.0))
                            matrix.loc[c1, c2] = val
                            matrix.loc[c2, c1] = val
                return matrix.fillna(0.0)
            except Exception:
                pass

        # Fallback to smart sample for wide column sets
        from core.parquet_manager import ParquetManager
        sample_df = ParquetManager.read_sample(parquet_path, n=sample_size)[cols]
        return sample_df.corr(method="pearson").round(4)
