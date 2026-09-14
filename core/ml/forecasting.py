"""
core.ml.forecasting
===================
Institutional Time Series Forecasting & Diagnostic Suite for Quantum Analyst.
Supports Automated Diagnostics (Trend, Seasonality, Stationarity ADF, Date Gaps),
Multi-Model Benchmark (Naive, Seasonal Naive, Moving Average, Exponential Smoothing,
Holt, Holt-Winters, ARIMA, SARIMAX, Random Forest, HistGB, XGBoost, LightGBM),
Confidence Intervals, and AutoML Champion Selection.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Try optional boosting
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def check_stationarity_adf(series: np.ndarray) -> Dict[str, Any]:
    """Tests stationarity of the series using the Augmented Dickey-Fuller (ADF) test."""
    try:
        from statsmodels.tsa.stattools import adfuller
        clean_s = pd.Series(series).dropna()
        if len(clean_s) < 10:
            return {"is_stationary": False, "p_value": 1.0, "statistic": 0.0, "status": "Insufficient observations"}
        result = adfuller(clean_s.values, autolag="AIC")
        stat, p_val = float(result[0]), float(result[1])
        is_stat = p_val < 0.05
        return {
            "is_stationary": is_stat,
            "p_value": round(p_val, 4),
            "statistic": round(stat, 4),
            "status": "Stationary (p < 0.05)" if is_stat else "Non-Stationary (Trend/Unit Root present)",
        }
    except Exception as e:
        return {"is_stationary": False, "p_value": 1.0, "statistic": 0.0, "status": "Test unavailable"}


def auto_detect_time_series_meta(df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """Detects frequency, time gaps, trend, seasonality, and stationarity."""
    ts_df = df[[date_col, target_col]].dropna().copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.dropna(subset=[date_col]).sort_values(by=date_col)
    ts_df[target_col] = pd.to_numeric(ts_df[target_col], errors="coerce")
    ts_df = ts_df.dropna()

    if len(ts_df) < 4:
        return {
            "valid": False,
            "error": "At least 4 valid chronological observations required.",
        }

    # Deduplicate timestamps by averaging
    grouped = ts_df.groupby(date_col)[target_col].mean().reset_index()
    dates = grouped[date_col].values
    y = grouped[target_col].values

    # Frequency and Gaps
    diffs = pd.Series(dates).diff().dropna()
    median_diff = diffs.median() if len(diffs) > 0 else pd.Timedelta(days=1)
    if pd.isna(median_diff) or median_diff == pd.Timedelta(0):
        median_diff = pd.Timedelta(days=1)

    # Missing Gaps Check
    gap_count = 0
    if len(diffs) > 0:
        gap_count = int(np.sum(diffs > 1.8 * median_diff))

    # Trend Detection
    x_idx = np.arange(len(y))
    slope, intercept = np.polyfit(x_idx, y, 1) if len(y) >= 2 else (0.0, y[0])
    mean_val = max(1e-5, abs(float(np.mean(y))))
    rel_slope = (slope / mean_val) * 100
    if rel_slope > 0.5:
        trend_label = "Upward Trend"
    elif rel_slope < -0.5:
        trend_label = "Downward Trend"
    else:
        trend_label = "Stationary / Sideways"

    # Seasonality check (Autocorrelation at lag 7 or 12 or 4)
    detected_seasonality = None
    if len(y) >= 14:
        s = pd.Series(y)
        for lag in [7, 12, 4]:
            if len(y) >= lag * 2:
                autocorr = s.autocorr(lag=lag)
                if not np.isnan(autocorr) and autocorr > 0.40:
                    detected_seasonality = f"Seasonal (Period = {lag}, Autocorr = {autocorr:.2f})"
                    break
    if not detected_seasonality:
        detected_seasonality = "No Dominant Seasonality Detected"

    # Stationarity ADF
    adf_res = check_stationarity_adf(y)

    return {
        "valid": True,
        "n_observations": len(y),
        "start_date": str(pd.to_datetime(dates[0]).strftime("%Y-%m-%d")),
        "end_date": str(pd.to_datetime(dates[-1]).strftime("%Y-%m-%d")),
        "median_frequency": str(median_diff),
        "gap_count": gap_count,
        "trend": trend_label,
        "slope_per_period": round(float(slope), 4),
        "seasonality": detected_seasonality,
        "stationarity": adf_res,
        "grouped_df": grouped,
    }


def _fit_forecast_candidate(
    name: str,
    train_y: np.ndarray,
    test_len: int,
    horizon: int,
    full_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits candidate model on train_y to evaluate test_preds, and refits on full_y for future forecast."""
    n_train = len(train_y)
    n_full = len(full_y)

    if name == "naive":
        test_p = np.full(test_len, train_y[-1])
        fut_p = np.full(horizon, full_y[-1])
        return test_p, fut_p

    elif name == "seasonal_naive":
        period = 7 if n_train >= 14 else (4 if n_train >= 8 else 1)
        test_p = np.array([train_y[-(period - (i % period))] for i in range(test_len)])
        fut_p = np.array([full_y[-(period - (i % period))] for i in range(horizon)])
        return test_p, fut_p

    elif name == "moving_average":
        w = min(5, max(2, n_train // 3))
        ma_val = float(np.mean(train_y[-w:]))
        test_p = np.full(test_len, ma_val)
        ma_full = float(np.mean(full_y[-w:]))
        fut_p = np.full(horizon, ma_full)
        return test_p, fut_p

    elif name == "exponential_smoothing":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        mod_tr = SimpleExpSmoothing(train_y, initialization_method="estimated").fit()
        test_p = mod_tr.forecast(test_len)
        mod_full = SimpleExpSmoothing(full_y, initialization_method="estimated").fit()
        fut_p = mod_full.forecast(horizon)
        return np.asarray(test_p), np.asarray(fut_p)

    elif name == "holt":
        from statsmodels.tsa.holtwinters import Holt
        mod_tr = Holt(train_y, initialization_method="estimated").fit()
        test_p = mod_tr.forecast(test_len)
        mod_full = Holt(full_y, initialization_method="estimated").fit()
        fut_p = mod_full.forecast(horizon)
        return np.asarray(test_p), np.asarray(fut_p)

    elif name == "holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        sp = 7 if n_train >= 21 else None
        mod_tr = ExponentialSmoothing(
            train_y,
            trend="add" if n_train >= 8 else None,
            seasonal="add" if sp else None,
            seasonal_periods=sp,
            initialization_method="estimated",
        ).fit()
        test_p = mod_tr.forecast(test_len)

        mod_full = ExponentialSmoothing(
            full_y,
            trend="add" if n_full >= 8 else None,
            seasonal="add" if sp else None,
            seasonal_periods=sp,
            initialization_method="estimated",
        ).fit()
        fut_p = mod_full.forecast(horizon)
        return np.asarray(test_p), np.asarray(fut_p)

    elif name == "arima":
        from statsmodels.tsa.arima.model import ARIMA
        mod_tr = ARIMA(train_y, order=(1, 1, 1)).fit()
        test_p = mod_tr.forecast(test_len)
        mod_full = ARIMA(full_y, order=(1, 1, 1)).fit()
        fut_p = mod_full.forecast(horizon)
        return np.asarray(test_p), np.asarray(fut_p)

    elif name == "sarimax":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        sp = 7 if n_train >= 21 else 1
        mod_tr = SARIMAX(train_y, order=(1, 1, 1), seasonal_order=(1, 0, 1, sp) if sp > 1 else (0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        test_p = mod_tr.forecast(test_len)
        mod_full = SARIMAX(full_y, order=(1, 1, 1), seasonal_order=(1, 0, 1, sp) if sp > 1 else (0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fut_p = mod_full.forecast(horizon)
        return np.asarray(test_p), np.asarray(fut_p)

    elif name in ["ml_lag_rf", "ml_lag_histgb", "ml_lag_xgb", "ml_lag_lgb"]:
        # Build lag dataset
        def _build_lags(arr):
            df_lag = pd.DataFrame({"y": arr})
            df_lag["lag_1"] = df_lag["y"].shift(1)
            df_lag["lag_2"] = df_lag["y"].shift(2)
            df_lag["lag_3"] = df_lag["y"].shift(3)
            df_lag["roll_3"] = df_lag["y"].shift(1).rolling(3).mean()
            clean = df_lag.dropna()
            return clean[["lag_1", "lag_2", "lag_3", "roll_3"]].values, clean["y"].values

        X_tr, y_tr = _build_lags(train_y)
        if len(X_tr) < 2:
            raise ValueError("Insufficient lag samples")

        if name == "ml_lag_rf":
            reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        elif name == "ml_lag_histgb":
            reg = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        elif name == "ml_lag_xgb" and XGB_AVAILABLE:
            reg = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        elif name == "ml_lag_lgb" and LGBM_AVAILABLE:
            reg = lgb.LGBMRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=-1)
        else:
            reg = Ridge(alpha=1.0)

        reg.fit(X_tr, y_tr)

        # Predict backtest recursively
        curr_y = list(train_y[-3:])
        t_preds = []
        for _ in range(test_len):
            feat = np.array([[curr_y[-1], curr_y[-2], curr_y[-3], np.mean(curr_y[-3:])]])
            p = float(reg.predict(feat)[0])
            t_preds.append(p)
            curr_y.append(p)

        # Refit on full series
        X_full, y_full = _build_lags(full_y)
        reg.fit(X_full, y_full)
        curr_full = list(full_y[-3:])
        f_preds = []
        for _ in range(horizon):
            feat = np.array([[curr_full[-1], curr_full[-2], curr_full[-3], np.mean(curr_full[-3:])]])
            p = float(reg.predict(feat)[0])
            f_preds.append(p)
            curr_full.append(p)

        return np.asarray(t_preds), np.asarray(f_preds)

    raise ValueError(f"Unknown forecast candidate '{name}'")


def run_time_series_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 14,
    method: str = "auto",  # auto, exponential_smoothing, holt, holt_winters, arima, sarimax, ml_lag_rf, naive
    confidence_level: float = 0.90,
) -> Dict[str, Any]:
    """
    Complete institutional time series forecasting engine with AutoML model competition,
    backtest accuracy benchmarking, and confidence interval estimation.
    """
    meta = auto_detect_time_series_meta(df, date_col, target_col)
    if not meta["valid"]:
        raise ValueError(meta["error"])

    grouped_df = meta["grouped_df"]
    dates = grouped_df[date_col].values
    y = grouped_df[target_col].values

    # Determine backtest split: min(horizon, 20% of data)
    test_len = max(1, min(horizon, max(1, int(len(y) * 0.2))))
    if len(y) - test_len < 2:
        test_len = 1

    train_y = y[:-test_len]
    test_y = y[-test_len:]

    # Candidate models for AutoML
    candidate_keys = [
        "naive",
        "seasonal_naive",
        "moving_average",
        "exponential_smoothing",
        "holt",
        "holt_winters",
        "arima",
        "sarimax",
        "ml_lag_rf",
    ]
    if XGB_AVAILABLE:
        candidate_keys.append("ml_lag_xgb")
    if LGBM_AVAILABLE:
        candidate_keys.append("ml_lag_lgb")

    # If user selected specific method, put it first
    if method != "auto" and method in candidate_keys:
        candidate_keys = [method] + [k for k in candidate_keys if k != method]

    benchmark_records = []
    best_candidate = None
    best_mae = float("inf")
    best_test_preds = None
    best_future_preds = None

    for cand_name in candidate_keys:
        try:
            t_preds, f_preds = _fit_forecast_candidate(cand_name, train_y, test_len, horizon, y)
            mae = float(mean_absolute_error(test_y, t_preds[:len(test_y)]))
            rmse = float(np.sqrt(mean_squared_error(test_y, t_preds[:len(test_y)])))
            try:
                mape = float(np.mean(np.abs((test_y - t_preds[:len(test_y)]) / np.maximum(1e-5, np.abs(test_y)))) * 100)
            except Exception:
                mape = 0.0
            try:
                smape = float(100 * np.mean(2 * np.abs(t_preds[:len(test_y)] - test_y) / (np.abs(test_y) + np.abs(t_preds[:len(test_y)]) + 1e-5)))
            except Exception:
                smape = 0.0

            cand_display_name = cand_name.replace("_", " ").title()
            benchmark_records.append({
                "model_key": cand_name,
                "model_name": cand_display_name,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "smape": round(smape, 2),
            })

            # Check if winner (or if user requested a specific method)
            if method != "auto" and cand_name == method:
                best_candidate = cand_name
                best_mae = mae
                best_test_preds = t_preds
                best_future_preds = f_preds
                break
            elif mae < best_mae:
                best_mae = mae
                best_candidate = cand_name
                best_test_preds = t_preds
                best_future_preds = f_preds

        except Exception:
            continue

    # Fallback to naive if everything failed
    if best_future_preds is None:
        best_candidate = "naive"
        best_future_preds = np.full(horizon, y[-1])
        best_test_preds = np.full(test_len, train_y[-1])
        best_mae = float(mean_absolute_error(test_y, best_test_preds[:len(test_y)]))

    # Calculate Confidence Intervals
    z_val = 1.96 if confidence_level >= 0.95 else 1.645
    res_std = float(np.std(y[-min(30, len(y)):])) if len(y) > 3 else 1.0
    steps = np.arange(1, horizon + 1)
    spread = z_val * res_std * np.sqrt(steps)
    lower_ci = best_future_preds - spread
    upper_ci = best_future_preds + spread

    # Calculate growth & expected value
    start_val = float(best_future_preds[0])
    end_val = float(best_future_preds[-1])
    expected_mean = float(np.mean(best_future_preds))
    growth_pct = round(((end_val - start_val) / max(1e-5, abs(start_val))) * 100, 2)
    growth_direction = "Growth" if growth_pct > 1 else ("Decline" if growth_pct < -1 else "Neutral")

    # Construct Future Date Sequence
    median_td = pd.to_timedelta(meta["median_frequency"])
    last_date = pd.to_datetime(dates[-1])
    future_dates = [last_date + (i + 1) * median_td for i in range(horizon)]

    # Subsample historical points for fast rendering (up to 250 points)
    hist_step = max(1, len(dates) // 250)
    historical_points = [
        {"date": str(pd.to_datetime(dates[i]).strftime("%Y-%m-%d")), "actual": round(float(y[i]), 2)}
        for i in range(0, len(dates), hist_step)
    ]

    forecast_points = [
        {
            "date": future_dates[i].strftime("%Y-%m-%d"),
            "predicted": round(float(best_future_preds[i]), 2),
            "lower_ci": round(float(lower_ci[i]), 2),
            "upper_ci": round(float(upper_ci[i]), 2),
        }
        for i in range(horizon)
    ]

    # Benchmark Leaderboard sorting
    benchmark_records.sort(key=lambda item: item["mae"])

    trend_summary = (
        f"Forecasting {horizon} future periods using {best_candidate.replace('_', ' ').title()}. "
        f"Projected {growth_direction.lower()} of {growth_pct:+.1f}% (Expected Mean: {expected_mean:.2f}). "
        f"Validation Backtest MAE is {best_mae:.2f} across holdout window."
    )

    return {
        "best_model": best_candidate.replace("_", " ").title(),
        "best_model_key": best_candidate,
        "horizon": horizon,
        "frequency": meta["median_frequency"],
        "confidence_level": confidence_level,
        "diagnostics": {
            "trend": meta["trend"],
            "seasonality": meta["seasonality"],
            "stationarity": meta["stationarity"],
            "gaps_detected": meta["gap_count"],
            "n_observations": meta["n_observations"],
        },
        "metrics": {
            "mae": round(best_mae, 2),
            "expected_mean": round(expected_mean, 2),
            "projected_growth_pct": growth_pct,
            "growth_direction": growth_direction,
        },
        "leaderboard": benchmark_records,
        "historical": historical_points,
        "forecast": forecast_points,
        "trend_summary": trend_summary,
        "disclaimer": "Forecasts represent algorithmic statistical projections. Never treat projected outputs as financial guarantees.",
    }
