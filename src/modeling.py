"""Simple ML modeling for forecasting photo activity trends.

This module demonstrates:
- Feature engineering (lags, rolling stats, calendar fields)
- Train/validation split by time
- Baseline Linear Regression and RandomForestRegressor
- Mean Absolute Error evaluation
- A naive forecast using last observed value (baseline for sanity)

Note: This is intentionally lightweight to run quickly on sample data.
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

FEATURES = ['dayofweek', 'month', 'rolling_mean_7', 'lag_1', 'lag_7']

def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time order, not random shuffle."""
    n_train = int(len(df) * train_ratio)
    train = df.iloc[:n_train].dropna().copy()
    valid = df.iloc[n_train:].dropna().copy()
    return train, valid

def evaluate_models(feat_df: pd.DataFrame) -> Dict[str, float]:
    """Train/evaluate simple models and return MAEs."""
    train, valid = time_split(feat_df, train_ratio=0.8)
    X_train, y_train = train[FEATURES], train['total_photos']
    X_valid, y_valid = valid[FEATURES], valid['total_photos']

    # Baseline: naive forecast (y_t = lag_1)
    naive_pred = valid['lag_1']
    mae_naive = mean_absolute_error(y_valid, naive_pred)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_valid)
    mae_lr = mean_absolute_error(y_valid, lr_pred)

    # Random Forest (small to keep runtime low)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_valid)
    mae_rf = mean_absolute_error(y_valid, rf_pred)

    return {
        'naive_mae': float(mae_naive),
        'linear_regression_mae': float(mae_lr),
        'random_forest_mae': float(mae_rf)
    }

def make_future_df(last_date: pd.Timestamp, periods: int = 14) -> pd.DataFrame:
    """Create a future date index for simple forecasting horizons."""
    idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    return pd.DataFrame({'date': idx})

def rolling_forecast(feat_df: pd.DataFrame, model, horizon_days: int = 14) -> pd.DataFrame:
    """Very simple iterative forecast using a fitted model and feature updates.

    This uses last known values to roll forward lag/rolling features.
    """
    hist = feat_df.copy().reset_index(drop=True)
    future = make_future_df(hist['date'].iloc[-1], periods=horizon_days)

    # Seed with last knowns
    last_vals = hist.iloc[-7:].copy()  # at least 7 days for lag_7
    preds = []
    for _, row in future.iterrows():
        # compute features for this future day
        dayofweek = row['date'].dayofweek
        month = row['date'].month
        # derive lag_1 from last known (last row in last_vals)
        lag_1 = last_vals['total_photos'].iloc[-1]
        lag_7 = last_vals['total_photos'].iloc[-7] if len(last_vals) >= 7 else lag_1
        rolling_mean_7 = last_vals['total_photos'].rolling(window=7, min_periods=1).mean().iloc[-1]

        X = [[dayofweek, month, rolling_mean_7, lag_1, lag_7]]
        yhat = float(model.predict(X)[0])
        preds.append({'date': row['date'], 'predicted_total_photos': yhat})

        # append the prediction as the new "observed" to update lags/rolling
        last_vals = pd.concat([last_vals, pd.DataFrame([
            {'date': row['date'], 'total_photos': yhat}
        ])], ignore_index=True)

    return pd.DataFrame(preds)
