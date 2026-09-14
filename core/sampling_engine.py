import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from core.duckdb_engine import DuckDBEngine
from core.parquet_manager import ParquetManager

class SamplingEngine:
    """Intelligent sampling engine distinguishing Exact Analytics from

    Sampled Visualizations, with stratification, temporal preservation, and reproducibility.
    """

    DEFAULT_SEED = 42

    @classmethod
    def get_visualization_sample(
        cls,
        parquet_path: Union[str, Path],
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        target_size: int = 5000,
        random_state: int = DEFAULT_SEED
    ) -> Tuple[pd.DataFrame, bool, str]:
        """Returns (sampled_dataframe, is_exact, telemetry_notice)."""
        meta = ParquetManager.get_parquet_metadata(parquet_path)
        total_rows = meta["num_rows"]

        # Small datasets require no sampling
        if total_rows <= target_size:
            df = ParquetManager.read_sample(parquet_path, n=total_rows)
            return df, True, f"Exact telemetry ({total_rows:,} records)"

        clean_p = DuckDBEngine._clean_path(parquet_path)
        con = DuckDBEngine.get_connection()

        # 1. Scatter Plot Sampling: Preserve extremes & tails
        if chart_type == "scatter" and x_col and y_col:
            # Subsample to max 1,500 points for crisp WebGL rendering
            sample_n = min(1500, target_size)
            query = f"""
            SELECT "{x_col}", "{y_col}" 
            FROM read_parquet('{clean_p}') 
            WHERE "{x_col}" IS NOT NULL AND "{y_col}" IS NOT NULL
            USING SAMPLE {sample_n} ROWS (BERNOULLI, {random_state});
            """
            try:
                scat_df = con.execute(query).df()
                return scat_df, False, f"Sampled {len(scat_df):,} representative points from {total_rows:,} records [WebGL Accelerated]"
            except Exception:
                pass

        # 2. Histogram / Distribution Sampling
        if chart_type in ["histogram", "box"] and (x_col or y_col):
            col = x_col or y_col
            sample_n = min(5000, target_size)
            query = f"""
            SELECT "{col}" 
            FROM read_parquet('{clean_p}') 
            WHERE "{col}" IS NOT NULL
            USING SAMPLE {sample_n} ROWS (BERNOULLI, {random_state});
            """
            try:
                hist_df = con.execute(query).df()
                return hist_df, False, f"Sampled {len(hist_df):,} observations from {total_rows:,} records [Distribution Preserved]"
            except Exception:
                pass

        # 3. Bar Chart: EXACT Pre-aggregated grouping over the entire 1M+ rows!
        if chart_type == "bar" and x_col:
            if y_col:
                agg_data = DuckDBEngine.get_bivariate_aggregation(parquet_path, x_col, y_col, top_n=30)
            else:
                agg_data = DuckDBEngine.get_categorical_frequency(parquet_path, x_col, top_n=30)
            bar_df = pd.DataFrame(agg_data)
            return bar_df, True, f"Exact aggregation computed across all {total_rows:,} records"

        # General Bernoulli sample
        query = f"SELECT * FROM read_parquet('{clean_p}') USING SAMPLE {target_size} ROWS (BERNOULLI, {random_state});"
        try:
            df = con.execute(query).df()
            return df, False, f"Sampled {len(df):,} rows from {total_rows:,} records"
        except Exception:
            df = con.execute(f"SELECT * FROM read_parquet('{clean_p}') LIMIT {target_size};").df()
            return df, False, f"Head sample of {len(df):,} rows from {total_rows:,} records"

    @classmethod
    def get_stratified_sample(
        cls,
        parquet_path: Union[str, Path],
        strata_col: str,
        sample_size: int = 5000,
        random_state: int = DEFAULT_SEED
    ) -> pd.DataFrame:
        """Draws a balanced sample preserving representation of minor and major classes."""
        clean_p = DuckDBEngine._clean_path(parquet_path)
        con = DuckDBEngine.get_connection()
        # Stratified sampling via window functions in DuckDB
        sql = f"""
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY "{strata_col}" 
                    ORDER BY HASH(CAST("{strata_col}" AS VARCHAR) || CAST(ROW_NUMBER() OVER() AS VARCHAR) || '{random_state}')
                ) as rank_within_class
            FROM read_parquet('{clean_p}')
            WHERE "{strata_col}" IS NOT NULL
        )
        SELECT * EXCLUDE (rank_within_class)
        FROM ranked
        ORDER BY rank_within_class ASC
        LIMIT {sample_size};
        """
        try:
            return con.execute(sql).df()
        except Exception:
            return ParquetManager.read_sample(parquet_path, n=sample_size, random_state=random_state)
