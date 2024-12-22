"""LSTM1 pipeline — Soil Water Content (SWTD) next-day prediction.

Preprocessing: Z-score outlier removal → StandardScaler.
Sequencing:    Season-aware sliding window (reset at season boundaries).
Architecture:  2-layer Bidirectional LSTM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .evaluation import regression_metrics, top_error_analysis
from .model_builder import build_lstm1_model
from .preprocessing import (
    build_sequences_by_season,
    fit_standard_scaler,
    remove_outliers_zscore,
)


def run_lstm1_swtd(
    config: Any,
    run_id: str,
    model_dir: Path,
    pred_dir: Path,
) -> dict[str, Any]:
    """Train & evaluate the SWTD prediction LSTM.  Returns a result dict."""
    data_cfg = config.data
    train_cfg = config.training

    # ------------------------------------------------------------------
    # 1. Load & prepare
    # ------------------------------------------------------------------
    df = pd.read_csv(config.dataset_path).dropna()
    df[data_cfg["date_column"]] = pd.to_datetime(df[data_cfg["date_column"]])
    df["season"] = df[data_cfg["date_column"]].dt.year % 2000
    df = df.set_index(data_cfg["date_column"]).sort_index()

    cols = [
        *data_cfg["feature_columns"],
        data_cfg["season_column"],
        data_cfg["target_column"],
    ]
    df = df[cols]

    # ------------------------------------------------------------------
    # 2. Train / test split (by year)
    # ------------------------------------------------------------------
    test_years = set(data_cfg["test_years"])
    date_index = pd.DatetimeIndex(df.index)
    train_df = df[~date_index.year.isin(test_years)].copy()
    test_df = df[date_index.year.isin(test_years)].copy()

    # ------------------------------------------------------------------
    # 3. Preprocessing — outlier removal + standard scaling
    # ------------------------------------------------------------------
    train_clean = remove_outliers_zscore(train_df)
    train_scaled, scaler = fit_standard_scaler(train_clean)

    x_train, y_train = build_sequences_by_season(
        train_scaled,
        seq_len=train_cfg["seq_len"],
        n_out=train_cfg["n_out"],
        shuffle=True,
    )

    # ------------------------------------------------------------------
    # 4. Build & train
    # ------------------------------------------------------------------
    model = build_lstm1_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        cfg=train_cfg,
        model_name="LSTM1_SWTD",
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=train_cfg["early_stopping_patience"],
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        validation_split=train_cfg["validation_split"],
        verbose=2,
        callbacks=[early],
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    scaled_test = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index,
    )
    x_test, y_test = build_sequences_by_season(
        scaled_test,
        seq_len=train_cfg["seq_len"],
        n_out=train_cfg["n_out"],
        shuffle=False,
    )
    y_pred = model.predict(x_test, verbose=0)

    # Inverse-transform to original scale
    target_idx = list(test_df.columns).index(data_cfg["target_column"])
    assert scaler.scale_ is not None
    assert scaler.mean_ is not None
    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    y_true_inv = y_test.flatten() * scale[target_idx] + mean[target_idx]
    y_pred_inv = y_pred.flatten() * scale[target_idx] + mean[target_idx]

    metrics = regression_metrics(y_true_inv, y_pred_inv)

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    seq_len = train_cfg["seq_len"]
    end_idx = seq_len + len(y_pred_inv)
    pred_dates = scaled_test.index[seq_len:end_idx]
    pred_df = pd.DataFrame(
        {
            "date": pred_dates,
            "actual": y_true_inv,
            "prediction": y_pred_inv,
        }
    )
    pred_df["abs_error"] = (pred_df["actual"] - pred_df["prediction"]).abs()

    model_path = model_dir / f"{run_id}.keras"
    pred_path = pred_dir / f"{run_id}.csv"
    model.save(model_path)
    pred_df.to_csv(pred_path, index=False)

    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
        "history_last": {
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "mae": float(history.history["mae"][-1]),
            "val_mae": float(history.history["val_mae"][-1]),
        },
        "data_summary": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "sequence_train_samples": len(x_train),
            "sequence_test_samples": len(x_test),
        },
        "failure_mode": {
            "top_errors": top_error_analysis(pred_df),
        },
    }
