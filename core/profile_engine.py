import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from core.parquet_manager import ParquetManager
from core.duckdb_engine import DuckDBEngine
from core.cache_manager import CacheManager

class ProfileEngine:
    """DuckDB-accelerated Fast Profiler computing comprehensive schema intelligence,

    null topologies, cardinality spectrums, and statistical moments in < 100ms.
    """

    @classmethod
    def profile_dataset_fast(
        cls,
        parquet_path: Union[str, Path],
        dataset_hash: str,
        version: str = "dataset_v1"
    ) -> Dict[str, Any]:
        """Calculates or retrieves cached DatasetProfile for a Parquet dataset."""
        # 1. Check Profile Cache
        cached = CacheManager.get(dataset_hash, version, "fast_profile")
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        meta = ParquetManager.get_parquet_metadata(parquet_path)
        total_rows = meta["num_rows"]
        total_cols = meta["num_columns"]
        col_names = meta["columns"]
        col_types = meta["column_types"]

        clean_p = DuckDBEngine._clean_path(parquet_path)
        con = DuckDBEngine.get_connection()

        # Classify column types based on Arrow schema
        num_cols = []
        cat_cols = []
        date_cols = []
        id_candidates = []

        for col in col_names:
            t_str = col_types.get(col, "").lower()
            name_lower = col.lower()
            if any(k in t_str for k in ["int", "float", "double", "decimal", "numeric"]):
                num_cols.append(col)
            elif any(k in t_str for k in ["date", "time", "timestamp"]):
                date_cols.append(col)
            else:
                if any(k in name_lower for k in ["date", "time", "timestamp", "dt"]):
                    date_cols.append(col)
                else:
                    cat_cols.append(col)

        # Single-pass DuckDB null & distinct query across all columns
        null_summary = {}
        unique_counts = {}

        if total_rows > 0:
            select_parts = []
            for c in col_names:
                safe_c = f'"{c}"'
                select_parts.append(f'COUNT({safe_c}) AS "{c}__nn"')
                if total_rows > 50000:
                    select_parts.append(f'APPROX_COUNT_DISTINCT({safe_c}) AS "{c}__uniq"')
                else:
                    select_parts.append(f'COUNT(DISTINCT {safe_c}) AS "{c}__uniq"')

            try:
                agg_sql = f"SELECT {', '.join(select_parts)} FROM read_parquet('{clean_p}');"
                row_agg = con.execute(agg_sql).df().iloc[0]
                for c in col_names:
                    non_null = int(row_agg.get(f"{c}__nn", total_rows))
                    null_count = max(0, total_rows - non_null)
                    null_pct = round((null_count / total_rows) * 100, 2)
                    null_summary[c] = {"null_count": null_count, "null_pct": null_pct}
                    uniq = int(row_agg.get(f"{c}__uniq", 0))
                    unique_counts[c] = uniq

                    # Check for identifier
                    c_lower = c.lower()
                    if (uniq >= total_rows * 0.98 and total_rows > 50) or any(k in c_lower for k in ["_id", "id_", "identifier", "uuid", "guid"]):
                        if c not in id_candidates:
                            id_candidates.append(c)
            except Exception:
                # Fallback on sample if complex nested types
                sample_df = ParquetManager.read_sample(parquet_path, n=5000)
                for c in col_names:
                    if c in sample_df.columns:
                        s_nn = int(sample_df[c].dropna().count())
                        s_null = len(sample_df) - s_nn
                        null_summary[c] = {"null_count": int(s_null * (total_rows / len(sample_df))), "null_pct": round((s_null / len(sample_df)) * 100, 2)}
                        unique_counts[c] = int(sample_df[c].nunique() * (total_rows / len(sample_df)))

        # Fast numerical moments for numeric columns (excluding high-cardinality IDs)
        active_num_cols = [c for c in num_cols if c not in id_candidates]
        statistics = {}
        if active_num_cols and total_rows > 0:
            statistics = DuckDBEngine.get_aggregates(parquet_path, active_num_cols)

        # Date range for datetime columns
        date_range = {}
        for d_col in date_cols:
            safe_d = f'"{d_col}"'
            try:
                d_sql = f"SELECT MIN({safe_d}) AS min_d, MAX({safe_d}) AS max_d FROM read_parquet('{clean_p}') WHERE {safe_d} IS NOT NULL;"
                d_row = con.execute(d_sql).df().iloc[0]
                date_range[d_col] = {
                    "min": str(d_row["min_d"]) if pd.notnull(d_row["min_d"]) else None,
                    "max": str(d_row["max_d"]) if pd.notnull(d_row["max_d"]) else None
                }
            except Exception:
                pass

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        profile = {
            "dataset_hash": dataset_hash,
            "version": version,
            "rows": total_rows,
            "columns": total_cols,
            "column_names": col_names,
            "column_types": col_types,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": date_cols,
            "identifier_columns": id_candidates,
            "null_summary": null_summary,
            "unique_counts": unique_counts,
            "statistics": statistics,
            "date_range": date_range,
            "memory_info": {
                "serialized_bytes": meta["serialized_size_bytes"],
                "serialized_mb": round(meta["serialized_size_bytes"] / (1024 * 1024), 2)
            },
            "profiling_time_ms": elapsed_ms
        }

        # Cache persistently in L1, L2, and L3
        CacheManager.set(dataset_hash, version, "fast_profile", profile, persist_l3=True)
        return profile
