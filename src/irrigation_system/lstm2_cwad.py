"""LSTM2 pipeline — Crop Weight at maturity (CWAD / yield) prediction.

Preprocessing: MinMaxScaler (features and target scaled **independently**).
Sequencing:    Contiguous sliding window.
Architecture:  2-layer Bidirectional LSTM with an extra hidden Dense layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .evaluation import regression_metrics, top_error_analysis
from .model_builder import build_lstm2_model
from .preprocessing import build_sequences, fit_minmax_scalers


def run_lstm2_cwad(
    config: Any,
    run_id: str,
    model_dir: Path,
    pred_dir: Path,
) -> dict[str, Any]:
    """Train & evaluate the CWAD yield-prediction LSTM.  Returns a result dict."""
    data_cfg = config.data
    train_cfg = config.training

    # ------------------------------------------------------------------
    # 1. Load & prepare
    # ------------------------------------------------------------------
    df = pd.read_csv(config.dataset_path)

    drop_cols = data_cfg.get("drop_columns", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna().copy()

    date_col = data_cfg["date_column"]
    target_col = data_cfg["target_column"]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Feature engineering — interaction terms
    if "PREC" in df.columns and "SWTD" in df.columns:
        df["PREC_SWTD"] = df["PREC"] * df["SWTD"]
    if "IRRC" in df.columns and "SWTD" in df.columns:
        df["IRRC_SWTD"] = df["IRRC"] * df["SWTD"]

    model_df = df.drop(columns=[date_col]).copy()

    # ------------------------------------------------------------------
    # 2. Scaling — features and target independently (no double-scaling)
    # ------------------------------------------------------------------
    scaled_df, _feature_scaler, target_scaler = fit_minmax_scalers(model_df, target_col)

    x = np.asarray(scaled_df.drop(columns=[target_col]).values, dtype=float)
    y = np.asarray(scaled_df[target_col].values, dtype=float)

    # ------------------------------------------------------------------
    # 3. Sequence building & train/test split
    # ------------------------------------------------------------------
    x_lstm, y_lstm = build_sequences(x, y, seq_len=train_cfg["seq_len"])

    x_train, x_test, y_train, y_test = train_test_split(
        x_lstm,
        y_lstm,
        test_size=train_cfg["test_size"],
        random_state=config.seed,
    )

    # ------------------------------------------------------------------
    # 4. Build & train
    # ------------------------------------------------------------------
    model = build_lstm2_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        cfg=train_cfg,
        model_name="LSTM2_CWAD",
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
        shuffle=True,
        callbacks=[early],
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    preds = model.predict(x_test, verbose=0)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    metrics = regression_metrics(y_test_inv, y_pred_inv)

    # Reconstruct dates for test predictions
    sampled_dates = df[date_col].iloc[train_cfg["seq_len"] :].reset_index(drop=True)
    sampled_dates = sampled_dates.iloc[: len(y_lstm)].reset_index(drop=True)
    _, test_idx = train_test_split(
        np.arange(len(y_lstm)),
        test_size=train_cfg["test_size"],
        random_state=config.seed,
    )

    pred_df = pd.DataFrame(
        {
            "date": sampled_dates.iloc[test_idx].values,
            "actual": y_test_inv,
            "prediction": y_pred_inv,
        }
    ).sort_values("date")
    pred_df["abs_error"] = (pred_df["actual"] - pred_df["prediction"]).abs()

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
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
            "raw_rows": len(df),
            "sequence_samples": len(x_lstm),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
        },
        "failure_mode": {
            "top_errors": top_error_analysis(pred_df),
        },
    }
