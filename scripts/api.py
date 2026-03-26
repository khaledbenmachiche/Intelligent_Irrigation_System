from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="Input time series with shape [seq_len, n_features]",
        min_length=1,
    )


class PredictResponse(BaseModel):
    predictions: list[float]


app = FastAPI(title="Irrigation Model API", version="1.0.0")

_model: tf.keras.Model | None = None
_model_load_error: str | None = None
_MODEL_PATH = os.getenv("MODEL_PATH", "models/latest.keras")
_MODEL_FEATURE_COUNT = int(os.getenv("MODEL_FEATURE_COUNT", "0"))


def _load_model() -> tf.keras.Model:
    model_path = Path(_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
    model = tf.keras.models.load_model(_MODEL_PATH, compile=False)
    return model


@app.on_event("startup")
def startup() -> None:
    global _model, _model_load_error
    try:
        _model = _load_model()
        _model_load_error = None
    except Exception as exc:  # pragma: no cover - defensive startup path
        _model = None
        _model_load_error = str(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_path": _MODEL_PATH,
        "model_loaded": _model is not None,
        "model_feature_count": _MODEL_FEATURE_COUNT or None,
        "model_load_error": _model_load_error,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _model is None:
        detail = "Model not loaded"
        if _model_load_error:
            detail = f"{detail}: {_model_load_error}"
        raise HTTPException(status_code=503, detail=detail)

    x = np.asarray(payload.sequence, dtype=np.float32)
    if x.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail="`sequence` must be a 2D array: [seq_len, n_features]",
        )
    if x.shape[1] == 0:
        raise HTTPException(
            status_code=422,
            detail="`sequence` must include at least one feature column",
        )
    if not np.isfinite(x).all():
        raise HTTPException(
            status_code=422,
            detail="`sequence` must contain only finite numeric values",
        )
    if _MODEL_FEATURE_COUNT and x.shape[1] != _MODEL_FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=(
                "`sequence` feature count does not match the deployed model: "
                f"expected {_MODEL_FEATURE_COUNT}, received {x.shape[1]}"
            ),
        )

    batch = np.expand_dims(x, axis=0)
    y = _model.predict(batch, verbose=0)

    if not isinstance(y, Sequence) and not isinstance(y, np.ndarray):
        raise HTTPException(status_code=500, detail="Unexpected model output")

    output = np.asarray(y, dtype=np.float32).reshape(-1)
    return PredictResponse(predictions=[float(v) for v in output])
