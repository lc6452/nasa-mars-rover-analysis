"""End-to-end runner for the NASA Mars Rover analysis demo.

This script:
1) Loads CSV metadata (synthetic sample is provided).
2) Aggregates daily totals and camera usage.
3) Produces time-series visualizations.
4) Engineers features and evaluates simple ML models.
5) Trains a RandomForestRegressor and generates a 14-day forecast.
6) Saves outputs into the results/ directory.

Run:
    python main.py

Author: Lucas Chang
"""
from __future__ import annotations
import os
import json
import pandas as pd
from src.preprocessing import load_dataset, aggregate_daily_photos, camera_usage, add_time_features
from src.visualization import plot_daily_totals, plot_camera_stacked
from src.modeling import evaluate_models, RandomForestRegressor, FEATURES, rolling_forecast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, 'data', 'rover_photos_metadata.csv')
RESULTS = os.path.join(BASE_DIR, 'results')

def ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)

def main():
    ensure_dirs()
    # 1) Load
    df = load_dataset(DATA)

    # 2) Aggregate
    daily = aggregate_daily_photos(df)
    cam_daily = camera_usage(df)

    # 3) Visualizations
    daily_plot = os.path.join(RESULTS, 'daily_totals.png')
    plot_daily_totals(daily, daily_plot)

    camera_plot = os.path.join(RESULTS, 'camera_usage_stacked.png')
    plot_camera_stacked(cam_daily, camera_plot)

    # 4) Feature engineering + 5) Train/Evaluate
    feat = add_time_features(daily)
    scores = evaluate_models(feat)

    # 6) Train final model and forecast
    # Using RandomForestRegressor with the same hyperparameters as evaluate_models
    train = feat.dropna().copy()
    X, y = train[FEATURES], train['total_photos']
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    forecast_df = rolling_forecast(train[['date', 'total_photos']].assign(
        dayofweek=train['date'].dt.dayofweek,
        month=train['date'].dt.month,
        rolling_mean_7=train['total_photos'].rolling(window=7, min_periods=1).mean(),
        lag_1=train['total_photos'].shift(1),
        lag_7=train['total_photos'].shift(7),
    ).dropna().reset_index(drop=True), model=model, horizon_days=14)

    # Save artifacts
    with open(os.path.join(RESULTS, 'model_scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)

    forecast_csv = os.path.join(RESULTS, 'forecast_14d.csv')
    forecast_df.to_csv(forecast_csv, index=False)

    print('Analysis complete. Outputs saved to results/:')
    print(' -', daily_plot)
    print(' -', camera_plot)
    print(' -', os.path.join(RESULTS, 'model_scores.json'))
    print(' -', forecast_csv)

if __name__ == '__main__':
    main()
