"""
core.ml.forecasting
===================
Time series forecasting engine supporting Exponential Smoothing (Holt-Winters),
ARIMA, ML Lag Regression, and Naive baselines with prediction intervals.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge


def run_time_series_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 14,
    method: str = "auto",  # auto, exponential_smoothing, arima, ml_lag, naive
) -> Dict[str, Any]:
    """
    Executes time series forecast and returns historical points, future predictions,
    confidence intervals, and backtest accuracy metrics.
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' or target column '{target_col}' not found.")

    # Prepare time series
    ts_df = df[[date_col, target_col]].dropna().copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.dropna(subset=[date_col]).sort_values(by=date_col)
    ts_df[target_col] = pd.to_numeric(ts_df[target_col], errors="coerce")
    ts_df = ts_df.dropna()

    if len(ts_df) < 4:
        raise ValueError("Insufficient time series observations (< 4 valid date-target rows).")

    # Aggregate if multiple entries per timestamp (e.g. sum or mean per day)
    ts_df = ts_df.groupby(date_col)[target_col].mean().reset_index()
    dates = ts_df[date_col].values
    y = ts_df[target_col].values

    # Determine frequency
    diffs = pd.Series(dates).diff().dropna()
    median_diff = diffs.median() if len(diffs) > 0 else pd.Timedelta(days=1)
    if pd.isna(median_diff) or median_diff == pd.Timedelta(0):
        median_diff = pd.Timedelta(days=1)

    # Train / Backtest Split (last min(10, 20%) as test)
    test_len = max(1, min(horizon, max(1, int(len(y) * 0.2))))
    if len(y) - test_len < 2:
        test_len = 1
    train_y = y[:-test_len]
    test_y = y[-test_len:]

    forecast_values = None
    lower_ci = None
    upper_ci = None
    method_used = method

    # 1. Exponential Smoothing
    if method in ["auto", "exponential_smoothing"]:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                train_y,
                trend="add" if len(train_y) > 15 else None,
                seasonal=None,
                initialization_method="estimated",
            ).fit()
            test_preds = model.forecast(test_len)

            # Refit on full series
            full_model = ExponentialSmoothing(
                y,
                trend="add" if len(y) > 15 else None,
                seasonal=None,
                initialization_method="estimated",
            ).fit()
            forecast_values = full_model.forecast(horizon)
            method_used = "Exponential Smoothing (Holt-Winters)"
        except Exception:
            if method == "exponential_smoothing":
                method_used = "ml_lag"

    # 2. ARIMA
    if (forecast_values is None and method == "auto") or method == "arima":
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_y, order=(1, 1, 1)).fit()
            test_preds = model.forecast(test_len)

            full_model = ARIMA(y, order=(1, 1, 1)).fit()
            res = full_model.get_forecast(horizon)
            forecast_values = res.predicted_mean
            ci = res.conf_int(alpha=0.10)  # 90% confidence
            lower_ci = ci[:, 0]
            upper_ci = ci[:, 1]
            method_used = "ARIMA(1,1,1)"
        except Exception:
            if method == "arima":
                method_used = "ml_lag"

    # 3. ML Lag Regression (Ridge autoregression)
    if forecast_values is None:
        try:
            # Build autoregressive tabular dataset
            lag_df = pd.DataFrame({"y": y})
            lag_df["lag_1"] = lag_df["y"].shift(1)
            lag_df["lag_2"] = lag_df["y"].shift(2)
            lag_df["lag_3"] = lag_df["y"].shift(3)
            lag_df["rolling_3"] = lag_df["y"].shift(1).rolling(3).mean()
            lag_clean = lag_df.dropna()

            X_lag = lag_clean[["lag_1", "lag_2", "lag_3", "rolling_3"]].values
            y_lag = lag_clean["y"].values

            reg = Ridge(alpha=1.0)
            reg.fit(X_lag, y_lag)

            # Recursive forecast
            curr_y = list(y[-3:])
            preds = []
            for _ in range(horizon):
                feat = np.array([[curr_y[-1], curr_y[-2], curr_y[-3], np.mean(curr_y[-3:])]])
                p = float(reg.predict(feat)[0])
                preds.append(p)
                curr_y.append(p)

            forecast_values = np.array(preds)
            test_preds = forecast_values[:test_len]
            method_used = "Machine Learning Autoregressive Lag Model"
        except Exception:
            pass

    # 4. Fallback: Naive Baseline
    if forecast_values is None:
        last_val = y[-1]
        forecast_values = np.full(horizon, last_val)
        test_preds = np.full(test_len, train_y[-1])
        method_used = "Naive Persistence Baseline"

    # Confidence Interval Fallback
    res_std = float(np.std(y[-min(30, len(y)):])) if len(y) > 3 else 1.0
    if lower_ci is None or upper_ci is None:
        steps = np.arange(1, horizon + 1)
        spread = 1.645 * res_std * np.sqrt(steps)  # 90% CI expanding with horizon
        lower_ci = forecast_values - spread
        upper_ci = forecast_values + spread

    # Accuracy Metrics on Backtest
    mae = float(mean_absolute_error(test_y, test_preds[:len(test_y)]))
    rmse = float(np.sqrt(mean_squared_error(test_y, test_preds[:len(test_y)])))
    try:
        mape = float(np.mean(np.abs((test_y - test_preds[:len(test_y)]) / np.maximum(1e-5, np.abs(test_y)))) * 100)
    except Exception:
        mape = 0.0

    # Build Future Date Sequence
    last_date = pd.to_datetime(dates[-1])
    future_dates = [last_date + (i + 1) * median_diff for i in range(horizon)]

    # Historical entries (subsample to 200 points for charts)
    hist_step = max(1, len(dates) // 200)
    historical_points = [
        {"date": str(pd.to_datetime(dates[i]).strftime("%Y-%m-%d")), "actual": round(float(y[i]), 2)}
        for i in range(0, len(dates), hist_step)
    ]

    forecast_points = [
        {
            "date": future_dates[i].strftime("%Y-%m-%d"),
            "predicted": round(float(forecast_values[i]), 2),
            "lower_ci": round(float(lower_ci[i]), 2),
            "upper_ci": round(float(upper_ci[i]), 2),
        }
        for i in range(horizon)
    ]

    # Trend Summary
    start_forecast = forecast_values[0]
    end_forecast = forecast_values[-1]
    pct_change = ((end_forecast - start_forecast) / max(1e-5, abs(start_forecast))) * 100
    direction = "upward growth" if pct_change > 2 else ("downward decline" if pct_change < -2 else "sideways stability")

    summary = (
        f"Forecasting {horizon} future periods using {method_used}. The model projects an overall {direction} "
        f"({pct_change:+.1f}%), with Backtest MAE of {mae:.2f} and MAPE of {mape:.1f}%."
    )

    return {
        "method_used": method_used,
        "horizon": horizon,
        "historical": historical_points,
        "forecast": forecast_points,
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
        },
        "trend_summary": summary,
        "frequency": str(median_diff),
    }
