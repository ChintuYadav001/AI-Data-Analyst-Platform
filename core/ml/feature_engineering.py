"""
core.ml.feature_engineering
===========================
Automated Feature Engineering Suite for Quantum Analyst.
Supports Yeo-Johnson Power Transformations, Quantile Normalization,
Polynomial & Interaction Crosses, Binning Discretization, and Time Series Lags.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
    KBinsDiscretizer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)


def apply_power_transform(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = "yeo-johnson",
) -> pd.DataFrame:
    """Stabilizes variance and normalizes skewed numeric distributions via Yeo-Johnson transformation."""
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return df.copy()

    res_df = df.copy()
    pt = PowerTransformer(method=method, standardize=True)
    transformed = pt.fit_transform(res_df[cols].fillna(res_df[cols].median()))

    for i, col in enumerate(cols):
        res_df[f"{col}_yeojohnson"] = transformed[:, i]

    return res_df


def apply_quantile_transform(
    df: pd.DataFrame,
    numeric_cols: List[str],
    output_distribution: str = "normal",
    n_quantiles: int = 100,
) -> pd.DataFrame:
    """Maps feature distributions to uniform or Gaussian bell curves, robust to extreme outliers."""
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return df.copy()

    res_df = df.copy()
    n_q = min(n_quantiles, len(df))
    qt = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_q, random_state=42)
    transformed = qt.fit_transform(res_df[cols].fillna(res_df[cols].median()))

    for i, col in enumerate(cols):
        res_df[f"{col}_quantile_{output_distribution}"] = transformed[:, i]

    return res_df


def generate_polynomial_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    degree: int = 2,
    interaction_only: bool = True,
    max_features: int = 20,
) -> pd.DataFrame:
    """Generates non-linear cross-product interaction terms between numeric columns."""
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return df.copy()

    # Cap to top 6 columns to prevent feature explosion
    eval_cols = cols[:6]
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_imputed = df[eval_cols].fillna(df[eval_cols].median())
    poly_mat = poly.fit_transform(X_imputed)
    poly_names = poly.get_feature_names_out(eval_cols)

    # Exclude original single features
    new_names = [name for name in poly_names if " " in name or "^" in name][:max_features]
    res_df = df.copy()

    for name in new_names:
        idx = list(poly_names).index(name)
        safe_col_name = name.replace(" ", "_x_")
        res_df[f"poly_{safe_col_name}"] = poly_mat[:, idx]

    return res_df


def generate_binned_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    n_bins: int = 5,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Discretizes continuous numeric features into categorical ordinal quantile bins."""
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return df.copy()

    res_df = df.copy()
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None)
    X_imputed = res_df[cols].fillna(res_df[cols].median())
    binned = disc.fit_transform(X_imputed)

    for i, col in enumerate(cols):
        res_df[f"{col}_bin_{n_bins}"] = binned[:, i].astype(int)

    return res_df


def generate_time_series_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = [1, 2, 3, 7],
    rolling_windows: List[int] = [3, 7, 14],
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Builds temporal autoregressive lag features, rolling window aggregates (mean, std),
    and calendar datetime indicators.
    """
    res_df = df.copy()

    # Sort if date_col provided
    if date_col and date_col in res_df.columns:
        res_df[date_col] = pd.to_datetime(res_df[date_col], errors="coerce")
        res_df = res_df.sort_values(by=date_col).reset_index(drop=True)

        # Date features
        dt = res_df[date_col]
        res_df[f"{date_col}_year"] = dt.dt.year
        res_df[f"{date_col}_month"] = dt.dt.month
        res_df[f"{date_col}_day"] = dt.dt.day
        res_df[f"{date_col}_dayofweek"] = dt.dt.dayofweek
        res_df[f"{date_col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Lags
    if target_col in res_df.columns:
        s = pd.to_numeric(res_df[target_col], errors="coerce")
        for lag in lags:
            res_df[f"{target_col}_lag_{lag}"] = s.shift(lag)

        # Rolling windows (shifted by 1 to prevent leakage)
        for w in rolling_windows:
            res_df[f"{target_col}_roll_mean_{w}"] = s.shift(1).rolling(w).mean()
            res_df[f"{target_col}_roll_std_{w}"] = s.shift(1).rolling(w).std()

    return res_df
