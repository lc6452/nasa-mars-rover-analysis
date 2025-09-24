"""Visualization helpers for the Mars Rover analysis.

Generates simple, readable matplotlib plots saved into the results/ folder.
Guidelines followed:
- One chart per figure (no subplots).
- Do not set explicit colors; rely on defaults to keep it simple.
"""
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_daily_totals(daily: pd.DataFrame, out_path: str) -> str:
    """Plot total photos per day as a line chart.

    Args:
        daily: DataFrame with columns ['date', 'total_photos'].
        out_path: File path to save the PNG.

    Returns:
        The saved file path.
    """
    plt.figure()
    plt.plot(daily['date'], daily['total_photos'])
    plt.title('Mars Rover Total Photos per Day')
    plt.xlabel('Date')
    plt.ylabel('Total Photos')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def plot_camera_stacked(camera_df: pd.DataFrame, out_path: str) -> str:
    """Render a stacked area chart by camera over time.

    Args:
        camera_df: DataFrame with columns ['date', 'camera', 'photos'].
        out_path: File path to save the PNG.

    Returns:
        The saved file path.
    """
    pivot = camera_df.pivot(index='date', columns='camera', values='photos').fillna(0)
    plt.figure()
    plt.stackplot(pivot.index, pivot.T.values)
    plt.title('Camera Usage Over Time (Stacked)')
    plt.xlabel('Date')
    plt.ylabel('Photos')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
