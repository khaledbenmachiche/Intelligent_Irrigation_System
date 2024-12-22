#!/usr/bin/env python3
"""CLI runner for notebook-aligned LSTM experiments."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Add src directory to Python path for importing irrigation_system package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Imports from irrigation_system must come after path setup
from irrigation_system.common import (  # noqa: E402
    ensure_dir,
    save_json,
    set_all_seeds,
    timestamp_utc,
)
from irrigation_system.config import load_config  # noqa: E402
from irrigation_system.lstm1_swtd import run_lstm1_swtd  # noqa: E402
from irrigation_system.lstm2_cwad import run_lstm2_cwad  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run notebook-aligned LSTM experiments"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    return parser.parse_args()


def main() -> None:
    """Main entry point for training LSTM models."""
    args = parse_args()
    cfg = load_config(args.config)

    set_all_seeds(cfg.seed)
    run_id = f"{cfg.experiment_name}_{timestamp_utc()}"

    artifact_dir = ensure_dir(cfg.artifact_dir)
    logs_dir = ensure_dir(artifact_dir / "logs")
    metrics_dir = ensure_dir(artifact_dir / "metrics")
    pred_dir = ensure_dir(artifact_dir / "predictions")
    model_dir = ensure_dir(cfg.model_output_dir)

    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] task={cfg.task} dataset={cfg.dataset_path}")

    start = time.perf_counter()
    if cfg.task == "swtd":
        result = run_lstm1_swtd(
            cfg, run_id=run_id, model_dir=model_dir, pred_dir=pred_dir
        )
    elif cfg.task == "cwad":
        result = run_lstm2_cwad(
            cfg, run_id=run_id, model_dir=model_dir, pred_dir=pred_dir
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")
    duration_sec = time.perf_counter() - start

    payload = {
        "run_id": run_id,
        "task": cfg.task,
        "experiment_name": cfg.experiment_name,
        "dataset_path": cfg.dataset_path,
        "seed": cfg.seed,
        "duration_sec": float(duration_sec),
        "metrics": result["metrics"],
        "history_last": result["history_last"],
        "data_summary": result["data_summary"],
        "model_path": result["model_path"],
        "predictions_path": result["predictions_path"],
        "failure_mode": result["failure_mode"],
    }

    metrics_path = metrics_dir / f"{run_id}.json"
    save_json(metrics_path, payload)

    final_line = (
        "[FINAL] "
        f"task={cfg.task} "
        f"rmse={payload['metrics']['rmse']:.6f} "
        f"r2={payload['metrics']['r2']:.6f} "
        f"mae={payload['metrics']['mae']:.6f} "
        f"duration_sec={duration_sec:.3f}"
    )
    print(final_line)

    final_path = logs_dir / f"{run_id}.final.log"
    final_path.write_text(final_line + "\n", encoding="utf-8")

    runs_csv = logs_dir / "experiment_runs.csv"
    row = {
        "run_id": run_id,
        "task": cfg.task,
        "experiment_name": cfg.experiment_name,
        "rmse": payload["metrics"]["rmse"],
        "r2": payload["metrics"]["r2"],
        "mae": payload["metrics"]["mae"],
        "duration_sec": duration_sec,
        "model_path": payload["model_path"],
        "predictions_path": payload["predictions_path"],
        "metrics_path": str(metrics_path),
    }
    write_header = not runs_csv.exists()
    with runs_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
