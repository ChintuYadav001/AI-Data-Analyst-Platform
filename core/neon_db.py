"""
core.neon_db
============
Institutional-grade Neon Serverless PostgreSQL Cloud Connector.
Provides high-performance data storage (upload) and retrieval (fetch) operations
for datasets and platform entities using psycopg2 and SQLAlchemy.

Key capabilities:
- URI normalization (handles postgres:// and postgresql:// with automatic sslmode=require)
- Bulk DataFrame storage into Neon with batched copy/insert
- High-speed DataFrame fetching from Neon tables
- Table discovery, metadata inspection, and schema reflection
- Live connection latency and health telemetry
"""

import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_neon_url(conn_str: Optional[str] = None) -> str:
    """Resolve and normalize Neon PostgreSQL connection URI."""
    url = conn_str or os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
    url = url.strip()
    if not url:
        return ""
    
    # Normalize postgres:// to postgresql:// for SQLAlchemy compatibility
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    # Ensure sslmode=require for Neon serverless
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    return url


def sanitize_table_name(name: str) -> str:
    """Sanitize string into valid PostgreSQL table identifier."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or clean[0].isdigit():
        clean = f"ds_{clean}"
    return clean[:63]


def test_neon_connection(conn_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Test connection to Neon PostgreSQL and measure round-trip ping latency.
    Returns dictionary with status, latency_ms, server_version, and database_name.
    """
    url = get_neon_url(conn_str)
    if not url:
        return {
            "success": False,
            "error": "No Neon connection URL provided. Set NEON_DATABASE_URL or paste connection string.",
            "latency_ms": None,
        }

    start = time.perf_counter()
    try:
        import psycopg2

        conn = psycopg2.connect(url, connect_timeout=8)
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_database(), current_user;")
            row = cur.fetchone()
            cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")
            table_count = cur.fetchone()[0]
        conn.close()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "success": True,
            "latency_ms": latency_ms,
            "server_version": row[0] if row else "PostgreSQL",
            "database_name": row[1] if row else "neondb",
            "current_user": row[2] if row else "unknown",
            "public_tables_count": table_count,
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "success": False,
            "error": str(e),
            "latency_ms": latency_ms,
        }


def list_neon_tables(conn_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all user tables in the public schema of Neon database with row counts and sizes."""
    url = get_neon_url(conn_str)
    if not url:
        return []

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(url, connect_timeout=8)
        query = """
            SELECT 
                t.table_name,
                COALESCE(s.n_live_tup, 0) AS estimated_rows,
                pg_size_pretty(pg_total_relation_size('"' || t.table_schema || '"."' || t.table_name || '"')) AS size_formatted,
                pg_total_relation_size('"' || t.table_schema || '"."' || t.table_name || '"') AS size_bytes
            FROM information_schema.tables t
            LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
            WHERE t.table_schema = 'public' 
              AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name ASC;
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            tables = cur.fetchall()
        conn.close()
        return [dict(t) for t in tables]
    except Exception as e:
        logger.error(f"Failed to list tables in Neon: {e}")
        return []


def fetch_table_from_neon(
    table_name: str,
    conn_str: Optional[str] = None,
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch a table from Neon PostgreSQL into a Pandas DataFrame.
    Returns (DataFrame, telemetry_dict).
    """
    url = get_neon_url(conn_str)
    if not url:
        raise ValueError("No Neon connection URL provided.")

    clean_table = sanitize_table_name(table_name)
    start = time.perf_counter()

    try:
        from sqlalchemy import create_engine

        engine = create_engine(url)
        cols_clause = "*"
        if columns:
            cols_clause = ", ".join([f'"{re.sub(r"[^a-zA-Z0-9_]", "", c)}"' for c in columns])

        query = f'SELECT {cols_clause} FROM "{clean_table}"'
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql(query, engine)
        fetch_time = round(time.perf_counter() - start, 3)

        meta = {
            "table_name": clean_table,
            "rows": len(df),
            "columns": len(df.columns),
            "fetch_time_seconds": fetch_time,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }
        return df, meta
    except Exception as e:
        logger.error(f"Error fetching table '{clean_table}' from Neon: {e}")
        raise


def store_dataframe_to_neon(
    df: pd.DataFrame,
    table_name: str,
    conn_str: Optional[str] = None,
    if_exists: str = "replace",  # 'replace', 'append', or 'fail'
    chunksize: int = 2000,
) -> Dict[str, Any]:
    """
    Store/upload any Pandas DataFrame to a Neon PostgreSQL table.
    Uses SQLAlchemy and fast batched inserts.
    """
    url = get_neon_url(conn_str)
    if not url:
        raise ValueError("No Neon connection URL provided. Set NEON_DATABASE_URL or paste connection string.")

    if df.empty:
        raise ValueError("Cannot store empty DataFrame to Neon.")

    clean_table = sanitize_table_name(table_name)
    start = time.perf_counter()

    # Clean column names for PostgreSQL identifiers
    df_clean = df.copy()
    col_mapping = {}
    seen_cols = set()
    for col in df_clean.columns:
        s_col = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip().lower()).strip("_")
        if not s_col or s_col[0].isdigit():
            s_col = f"col_{s_col}"
        # Ensure uniqueness
        base_s = s_col
        counter = 1
        while s_col in seen_cols:
            s_col = f"{base_s}_{counter}"
            counter += 1
        seen_cols.add(s_col)
        col_mapping[col] = s_col

    df_clean.rename(columns=col_mapping, inplace=True)

    try:
        from sqlalchemy import create_engine

        engine = create_engine(url, pool_pre_ping=True)
        df_clean.to_sql(
            clean_table,
            engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
        )
        duration = round(time.perf_counter() - start, 3)

        return {
            "success": True,
            "table_name": clean_table,
            "rows_stored": len(df_clean),
            "columns_stored": len(df_clean.columns),
            "execution_time_seconds": duration,
            "if_exists": if_exists,
        }
    except Exception as e:
        logger.error(f"Error storing table '{clean_table}' to Neon: {e}")
        raise


def execute_neon_query(
    query: str,
    conn_str: Optional[str] = None,
    params: Optional[Tuple] = None,
) -> pd.DataFrame:
    """Execute arbitrary SQL query against Neon and return result as DataFrame."""
    url = get_neon_url(conn_str)
    if not url:
        raise ValueError("No Neon connection URL provided.")

    from sqlalchemy import create_engine

    engine = create_engine(url)
    return pd.read_sql(query, engine, params=params)
