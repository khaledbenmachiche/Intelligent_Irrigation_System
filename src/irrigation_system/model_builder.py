"""Shared Bidirectional-LSTM model builder for irrigation system pipelines."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Custom Keras metric
# ---------------------------------------------------------------------------


def r2_keras(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """R² (coefficient of determination) as a Keras metric."""
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _compile_model(
    inp: tf.Tensor,
    output: tf.Tensor,
    cfg: dict[str, Any],
    *,
    model_name: str,
) -> tf.keras.Model:
    """Compile a Keras model with the shared optimizer/loss/metrics policy."""
    lr = cfg["learning_rate"]

    metric_list: list[Any] = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        r2_keras,
    ]

    model = tf.keras.Model(inputs=inp, outputs=output, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse",
        metrics=metric_list,
    )
    return model


def build_lstm1_model(
    input_shape: tuple[int, int],
    cfg: dict[str, Any],
    model_name: str = "LSTM1_SWTD",
) -> tf.keras.Model:
    """Build LSTM1 architecture.

    Parameters
    ----------
    input_shape:
        ``(seq_len, n_features)``
    cfg:
        Must contain keys: ``n_out``, ``hidden_units``, ``dropout_size``,
        ``learning_rate``. Optional ``lstm1_architecture`` can override
        ``lstm1_units``, ``lstm2_units``, ``dropout1``, ``dropout2``.
    model_name:
        Keras model name for summary display.
    """
    n_out = int(cfg["n_out"])
    hidden_units = int(cfg["hidden_units"])
    dropout_size = float(cfg["dropout_size"])
    arch = cfg.get("lstm1_architecture", {})

    lstm1_units = int(arch.get("lstm1_units", hidden_units))
    lstm2_units = int(arch.get("lstm2_units", hidden_units))
    dropout1 = float(arch.get("dropout1", dropout_size))
    dropout2 = float(arch.get("dropout2", dropout_size))

    inp = Input(shape=input_shape, name="input_layer")

    x = Bidirectional(
        LSTM(lstm1_units, activation="tanh", return_sequences=True, name="lstm1"),
        name="blstm1",
    )(inp)
    x = Dropout(dropout1, name="dropout1")(x)

    x = Bidirectional(
        LSTM(lstm2_units, activation="tanh", return_sequences=False, name="lstm2"),
        name="blstm2",
    )(x)
    x = Dropout(dropout2, name="dropout2")(x)

    output = Dense(n_out, activation="linear", name="output_layer")(x)
    return _compile_model(inp, output, cfg, model_name=model_name)


def build_lstm2_model(
    input_shape: tuple[int, int],
    cfg: dict[str, Any],
    model_name: str = "LSTM2_CWAD",
) -> tf.keras.Model:
    """Build LSTM2 architecture.

    Optional ``lstm2_architecture`` can override ``lstm1_units``,
    ``lstm2_units``, ``dropout1``, ``dropout2``, ``dense_hidden_units``,
    and ``dense_activation``.
    """
    n_out = int(cfg["n_out"])
    hidden_units = int(cfg["hidden_units"])
    dropout_size = float(cfg["dropout_size"])
    arch = cfg.get("lstm2_architecture", {})

    lstm1_units = int(arch.get("lstm1_units", hidden_units))
    lstm2_units = int(arch.get("lstm2_units", hidden_units))
    dropout1 = float(arch.get("dropout1", dropout_size))
    dropout2 = float(arch.get("dropout2", dropout_size))
    dense_hidden_units = int(arch.get("dense_hidden_units", hidden_units))
    dense_activation = str(arch.get("dense_activation", "tanh"))

    inp = Input(shape=input_shape, name="input_layer")

    x = Bidirectional(
        LSTM(lstm1_units, activation="tanh", return_sequences=True, name="lstm1"),
        name="blstm1",
    )(inp)
    x = Dropout(dropout1, name="dropout1")(x)

    x = Bidirectional(
        LSTM(lstm2_units, activation="tanh", return_sequences=False, name="lstm2"),
        name="blstm2",
    )(x)
    x = Dropout(dropout2, name="dropout2")(x)

    x = Dense(
        dense_hidden_units,
        activation=dense_activation,
        name="dense_hidden",
    )(x)

    output = Dense(n_out, activation="linear", name="output_layer")(x)
    return _compile_model(inp, output, cfg, model_name=model_name)
