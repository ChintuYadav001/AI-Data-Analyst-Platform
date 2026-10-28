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


def engineer_sales_and_business_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Automatically detects and constructs high-impact domain features for predictive sales,
    revenue, and business models:
    1. Product Category & Region domain groupings
    2. Pricing, Discount, and Quantity volume cross-product interactions
    3. Calendar & Seasonality indicators (month, quarter, day of week, cyclical sine/cosine)
    """
    res_df = df.copy()
    new_features = []
    cols_lower = {c: c.lower().strip() for c in res_df.columns}

    # 1. Detect Pricing, Discount, and Volume/Quantity columns
    qty_col = next((c for c, l in cols_lower.items() if l in ["quantity", "qty", "units", "units_sold", "order_quantity", "volume"] and c != target_col), None)
    price_col = next((c for c, l in cols_lower.items() if l in ["price", "unit_price", "unitprice", "item_price", "rate"] and c != target_col), None)
    discount_col = next((c for c, l in cols_lower.items() if l in ["discount", "discount_rate", "discount_pct", "discount_percent"] and c != target_col), None)
    profit_col = next((c for c, l in cols_lower.items() if l in ["profit", "margin", "net_profit"] and c != target_col), None)

    # Derive interaction features
    if qty_col and price_col and pd.api.types.is_numeric_dtype(res_df[qty_col]) and pd.api.types.is_numeric_dtype(res_df[price_col]):
        feat_name = f"{qty_col}_x_{price_col}_gross_val"
        res_df[feat_name] = res_df[qty_col] * res_df[price_col]
        new_features.append(feat_name)

    if price_col and discount_col and pd.api.types.is_numeric_dtype(res_df[price_col]) and pd.api.types.is_numeric_dtype(res_df[discount_col]):
        # Normalize discount if represented as percentage 0-100
        disc_series = res_df[discount_col].astype(float)
        if disc_series.max() > 1.0:
            disc_series = disc_series / 100.0

        feat_disc_price = f"{price_col}_net_after_discount"
        res_df[feat_disc_price] = res_df[price_col] * (1.0 - disc_series)
        new_features.append(feat_disc_price)

        feat_disc_amt = f"{price_col}_discount_amount"
        res_df[feat_disc_amt] = res_df[price_col] * disc_series
        new_features.append(feat_disc_amt)

    if profit_col and target_col and target_col in res_df.columns and pd.api.types.is_numeric_dtype(res_df[profit_col]) and pd.api.types.is_numeric_dtype(res_df[target_col]):
        feat_margin = f"{profit_col}_ratio_to_target"
        res_df[feat_margin] = res_df[profit_col] / (res_df[target_col].abs() + 1e-5)
        new_features.append(feat_margin)

    # 2. Seasonality & Calendar features from Datetime columns
    for c, l in cols_lower.items():
        if c == target_col:
            continue
        # Check if column contains date/time or is datetime dtype
        is_dt = pd.api.types.is_datetime64_any_dtype(res_df[c])
        if not is_dt and any(k in l for k in ["date", "time", "order_date", "timestamp", "period", "day"]):
            try:
                dt_parsed = pd.to_datetime(res_df[c], errors="coerce")
                if dt_parsed.notna().sum() > len(res_df) * 0.6:
                    is_dt = True
                    dt_series = dt_parsed
            except Exception:
                pass
        elif is_dt:
            dt_series = res_df[c]

        if is_dt and 'dt_series' in locals():
            # Extract month, quarter, day of week, day of year, weekend
            m_col = f"{c}_month"
            q_col = f"{c}_quarter"
            dow_col = f"{c}_dayofweek"
            wknd_col = f"{c}_is_weekend"
            doy_col = f"{c}_dayofyear"
            
            res_df[m_col] = dt_series.dt.month.fillna(1).astype(int)
            res_df[q_col] = dt_series.dt.quarter.fillna(1).astype(int)
            res_df[dow_col] = dt_series.dt.dayofweek.fillna(0).astype(int)
            res_df[wknd_col] = (dt_series.dt.dayofweek >= 5).astype(int)
            res_df[doy_col] = dt_series.dt.dayofyear.fillna(1).astype(int)

            # Cyclical seasonality transformations (sine/cosine)
            sin_m = f"{c}_month_sin"
            cos_m = f"{c}_month_cos"
            res_df[sin_m] = np.sin(2 * np.pi * res_df[m_col] / 12.0)
            res_df[cos_m] = np.cos(2 * np.pi * res_df[m_col] / 12.0)

            new_features.extend([m_col, q_col, dow_col, wknd_col, doy_col, sin_m, cos_m])

    return res_df, new_features

