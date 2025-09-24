"""Preprocessing utilities for NASA Mars Rover photo metadata.

This module includes functions to load, validate, and transform the dataset
for downstream visualization and modeling.

The dataset columns are:
- earth_date (YYYY-MM-DD string)
- sol (int)
- rover (str) e.g., Curiosity, Perseverance
- camera (str) e.g., MAST, NAVCAM, etc.
- photos (int) number of photos taken by that rover+camera on that date

Author: Your Name (Lucas Chang)
"""
from __future__ import annotations
import pandas as pd

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the rover metadata CSV and coerce types.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with parsed dtypes and a proper `date` column.
    """
    df = pd.read_csv(csv_path)
    # Defensive type handling
    df['earth_date'] = pd.to_datetime(df['earth_date'], errors='coerce')
    df = df.dropna(subset=['earth_date'])
    df['sol'] = pd.to_numeric(df['sol'], errors='coerce').astype('Int64')
    df['photos'] = pd.to_numeric(df['photos'], errors='coerce').fillna(0).astype(int)
    df['rover'] = df['rover'].astype(str)
    df['camera'] = df['camera'].astype(str)
    df = df.rename(columns={'earth_date': 'date'})
    return df

def aggregate_daily_photos(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a daily total across rovers and cameras.

    Helpful for time-series visualizations and forecasting.

    Returns:
        DataFrame with columns: date, total_photos
    """
    daily = (
        df.groupby('date', as_index=False)['photos']
          .sum()
          .rename(columns={'photos': 'total_photos'})
          .sort_values('date')
          .reset_index(drop=True)
    )
    return daily

def camera_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total photos per camera per day.

    Returns:
        DataFrame with columns: date, camera, photos
    """
    return (
        df.groupby(['date', 'camera'], as_index=False)['photos']
          .sum()
          .sort_values(['date', 'camera'])
          .reset_index(drop=True)
    )

def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar-based features for modeling.

    Features:
        - dayofweek (0-6)
        - month (1-12)
        - rolling_mean_7 (7-day rolling average)
        - lag_1, lag_7

    Returns:
        DataFrame with added feature columns (NaNs may be present initially).
    """
    out = daily.copy()
    out['dayofweek'] = out['date'].dt.dayofweek
    out['month'] = out['date'].dt.month
    out['rolling_mean_7'] = out['total_photos'].rolling(window=7, min_periods=1).mean()
    out['lag_1'] = out['total_photos'].shift(1)
    out['lag_7'] = out['total_photos'].shift(7)
    return out
