import pandas as pd
from typing import Dict, Any, Optional

def compute_df_hash(df: Optional[pd.DataFrame], extra: str = "") -> str:
    if df is None:
        return f"empty_{extra}"
    n_rows, n_cols = df.shape
    if n_rows == 0:
        return f"0x{n_cols}_{extra}"
    
    first_idx = df.index[0]
    last_idx = df.index[-1]
    cols_hash = hash(tuple(df.columns))
    sample_val1 = str(df.iat[0, 0]) if n_cols > 0 else ""
    sample_val2 = str(df.iat[-1, min(n_cols - 1, 1)]) if n_cols > 0 else ""
    
    return f"{n_rows}x{n_cols}_{cols_hash}_{first_idx}_{last_idx}_{sample_val1}_{sample_val2}_{extra}"

class AnalyticsCache:
    """High-speed in-memory cache for analytical models to prevent redundant recalculations
    on every Streamlit page rerun.
    """
    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, df: pd.DataFrame, key: str, extra: str = "") -> Optional[Any]:
        df_hash = compute_df_hash(df, extra)
        cache_key = f"{df_hash}:{key}"
        return cls._cache.get(cache_key)

    @classmethod
    def set(cls, df: pd.DataFrame, key: str, value: Any, extra: str = "") -> None:
        df_hash = compute_df_hash(df, extra)
        cache_key = f"{df_hash}:{key}"
        # Keep cache bounded to prevent memory leakage
        if len(cls._cache) > 1000:
            cls._cache.clear()
        cls._cache[cache_key] = value

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()
