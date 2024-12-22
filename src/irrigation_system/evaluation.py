"""Evaluation utilities for irrigation system pipelines."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics on original-scale values."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def top_error_analysis(
    pred_df: pd.DataFrame,
    n: int = 5,
) -> list[dict]:
    """Return the *n* worst predictions sorted by absolute error.

    Expects *pred_df* to contain columns: ``date``, ``actual``,
    ``prediction``, ``abs_error``.
    """
    worst = pred_df.sort_values("abs_error", ascending=False).head(n).copy()
    worst["date"] = pd.to_datetime(worst["date"]).dt.strftime("%Y-%m-%d")
    records = worst[["date", "actual", "prediction", "abs_error"]].to_dict(orient="records")
    return cast(list[dict[str, Any]], records)
