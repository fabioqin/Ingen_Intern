"""Run, persist, and independently read back a local MLflow experiment."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
from mlflow import MlflowClient

from safety_lens.baseline import train
from safety_lens.device import device_info


def run(seed=42, epochs=200, learning_rate=0.1, tracking_uri=None):
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("week01_textbook_baseline")
    params = {
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "model": "logistic_regression",
        "dataset": "synthetic_v1",
        "n_samples": 1000,
        "n_features": 10,
        "test_fraction": 0.25,
        "device": "cpu",
    }
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    start = time.perf_counter()
    with mlflow.start_run() as active:
        mlflow.log_params(params)
        metrics, losses, checkpoint, predictions = train(seed, epochs, learning_rate)
        metrics.update(
            {
                "elapsed_seconds": time.perf_counter() - start,
                "cpu_seconds": sum(process.cpu_times()[:2]) - cpu_start,
                "rss_end_mb": process.memory_info().rss / 1024**2,
            }
        )
        mlflow.log_metrics(metrics)
        for step, loss in enumerate(losses):
            mlflow.log_metric("train_loss", loss, step=step)
        mlflow.log_dict(device_info(), "device.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save(checkpoint, path / "model.pt")
            np.savetxt(
                path / "predictions.csv",
                predictions,
                delimiter=",",
                header="label,probability",
                comments="",
            )
            mlflow.log_artifacts(directory)
        run_id = active.info.run_id
    client = MlflowClient(tracking_uri=uri)
    saved = client.get_run(run_id)
    assert saved.info.status == "FINISHED"
    assert all(saved.data.params[k] == str(v) for k, v in params.items())
    assert all(saved.data.metrics[k] == v for k, v in metrics.items())
    assert len(client.get_metric_history(run_id, "train_loss")) == epochs
    assert {"model.pt", "predictions.csv", "device.json"} <= {
        artifact.path for artifact in client.list_artifacts(run_id)
    }
    return {
        "run_id": run_id,
        "tracking_uri": uri,
        "metrics": metrics,
        "readback_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--output", default="outputs/latest_run.json")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive")
    result = run(args.seed, args.epochs, args.learning_rate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
