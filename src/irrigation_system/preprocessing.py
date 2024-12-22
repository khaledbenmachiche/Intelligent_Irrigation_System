"""Shared preprocessing utilities for irrigation system pipelines."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------


def remove_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Drop rows where any numeric column has |z-score| >= *threshold*."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    z = np.abs(df[numeric_cols].apply(zscore))
    return df[(z < threshold).all(axis=1)].copy()


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------


def fit_standard_scaler(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on all numeric columns and return (scaled_df, scaler)."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    scaler = StandardScaler()
    scaled = df.copy()
    scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return scaled, scaler


def fit_minmax_scalers(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """Scale features and target **independently** with MinMaxScaler.

    Returns (scaled_df, feature_scaler, target_scaler).
    This avoids the double-scaling bug where the target is scaled as part
    of the feature matrix and then scaled again separately.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled = df.copy()
    scaled[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
    scaled[target_col] = target_scaler.fit_transform(df[[target_col]]).flatten()
    return scaled, feature_scaler, target_scaler


# ---------------------------------------------------------------------------
# Sequence builders
# ---------------------------------------------------------------------------


def build_sequences(
    x: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequence builder for contiguous time-series data."""
    x_seq, y_seq = [], []
    for i in range(len(x) - seq_len):
        x_seq.append(x[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(x_seq), np.array(y_seq)


def build_sequences_by_season(
    df: pd.DataFrame,
    seq_len: int,
    n_out: int,
    *,
    shuffle: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build windowed sequences, resetting the window at season boundaries.

    Expects the dataframe to contain a ``season`` column.  The last *n_out*
    columns (excluding ``season``) are treated as target(s).
    """
    input_columns = [
        col for col in df.columns if col != "season" and col not in df.columns[-n_out:].tolist()
    ]
    output_columns = df.columns[-n_out:]

    sequential_data: list[tuple[np.ndarray, list[float]]] = []
    prev_day: deque[list[float]] = deque(maxlen=seq_len)
    last_season = None

    for _, row in df.iterrows():
        current_season = row["season"]
        if last_season is not None and current_season != last_season:
            prev_day.clear()
        last_season = current_season

        prev_day.append([row[col] for col in input_columns])
        if len(prev_day) == seq_len:
            target = [row[col] for col in output_columns]
            sequential_data.append((np.array(prev_day), target))

    if shuffle:
        np.random.shuffle(sequential_data)

    x_data = np.array([s[0] for s in sequential_data])
    y_data = np.array([s[1] for s in sequential_data])
    return x_data, y_data
