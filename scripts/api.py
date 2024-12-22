from __future__ import annotations

import os
from collections.abc import Sequence
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
_MODEL_PATH = os.getenv("MODEL_PATH", "models/latest.keras")


def _load_model() -> tf.keras.Model:
    model = tf.keras.models.load_model(_MODEL_PATH, compile=False)
    return model


@app.on_event("startup")
def startup() -> None:
    global _model
    _model = _load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": _MODEL_PATH,
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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

    batch = np.expand_dims(x, axis=0)
    y = _model.predict(batch, verbose=0)

    if not isinstance(y, Sequence):
        raise HTTPException(status_code=500, detail="Unexpected model output")

    output = np.asarray(y, dtype=np.float32).reshape(-1)
    return PredictResponse(predictions=[float(v) for v in output])
