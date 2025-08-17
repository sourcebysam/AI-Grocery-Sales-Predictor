"""
Model utilities for training and forecasting sales.
- Tries Prophet first; if unavailable, falls back to LinearRegression.
- Exposes: prepare_df, fit_prophet, fit_linear, forecast
"""
from __future__ import annotations
import pandas as pd
from typing import Tuple


def prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # Normalize column names
    rename_map = {c: c.strip() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    if "Date" not in df.columns or "Units_Sold" not in df.columns:
        raise ValueError("Input must have 'Date' and 'Units_Sold' columns.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Keep only necessary columns for forecasting
    model_df = df[["Date", "Units_Sold"]].rename(columns={"Date": "ds", "Units_Sold": "y"})
    model_df["y"] = pd.to_numeric(model_df["y"], errors="coerce").fillna(0)
    return model_df


def fit_prophet(df: pd.DataFrame):
    from prophet import Prophet
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    return m


def fit_linear(df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression
    import numpy as np

    x = (df["ds"] - df["ds"].min()).dt.days.values.reshape(-1, 1)
    y = df["y"].values
    reg = LinearRegression().fit(x, y)
    return reg


def forecast(df: pd.DataFrame, horizon_days: int = 30) -> Tuple[pd.DataFrame, str, object]:
    """
    Returns (forecast_df, model_name, model).
    - Prophet path returns the FULL forecast_df (with trend, seasonality, etc.)
    - LinearRegression fallback returns synthetic yhat, yhat_lower, yhat_upper
    """
    assert horizon_days > 0
    try:
        # Prophet path
        model = fit_prophet(df)
        future = model.make_future_dataframe(periods=horizon_days)
        fc = model.predict(future)  # <-- keep full DataFrame (has trend, yearly, weekly)
        return fc, "Prophet", model

    except Exception:
        # Fallback to Linear Regression
        import numpy as np

        model = fit_linear(df)
        last_day = (df["ds"].max() - df["ds"].min()).days
        future_x = (pd.RangeIndex(last_day + 1, last_day + 1 + horizon_days).values).reshape(-1, 1)

        # Predictions for historical + future
        hist_x = ((df["ds"] - df["ds"].min()).dt.days.values).reshape(-1, 1)
        hist_pred = model.predict(hist_x)
        fut_pred = model.predict(future_x)

        out = pd.DataFrame({
            "ds": pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon_days),
            "yhat": fut_pred,
        })

        # Symmetric confidence interval
        resid = (df["y"].values - hist_pred)
        pad = max(5.0, resid.std() * 1.96)
        out["yhat_lower"] = out["yhat"] - pad
        out["yhat_upper"] = out["yhat"] + pad

        return out, "LinearRegression (fallback)", model
