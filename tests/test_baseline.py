import numpy as np
import pytest
import torch
from mlflow import MlflowClient
from sklearn.metrics import brier_score_loss, log_loss

from safety_lens.baseline import train
from safety_lens.experiment import run


def test_seed_reproduces_training():
    a, losses_a, _, predictions_a = train(epochs=30)
    b, losses_b, _, predictions_b = train(epochs=30)
    assert a == b
    assert losses_a == losses_b
    np.testing.assert_array_equal(predictions_a, predictions_b)
    assert losses_a[-1] < losses_a[0]


def test_probabilities_and_metrics():
    metrics, _, _, predictions = train(epochs=30)
    y, p = predictions.T
    assert len(y) == 250
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert metrics["brier_score"] == pytest.approx(np.mean((p - y) ** 2))
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert metrics["log_loss"] == pytest.approx(expected)


def test_mlflow_persistence_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = "sqlite:///tracking.db"
    result = run(epochs=30, tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    record = client.get_run(result["run_id"])
    assert result["readback_verified"]
    assert record.data.params["seed"] == "42"
    assert len(client.get_metric_history(result["run_id"], "train_loss")) == 30
    (tmp_path / "download").mkdir()
    folder = client.download_artifacts(result["run_id"], "", str(tmp_path / "download"))
    from pathlib import Path

    checkpoint = torch.load(Path(folder) / "model.pt", weights_only=True)
    model = torch.nn.Linear(10, 1)
    model.load_state_dict(checkpoint["state_dict"])
    # Replay held-out inputs using the persisted scaler and trained weights.
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        class_sep=1.0,
        random_state=42,
    )
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    transformed = (x_test - checkpoint["scaler_mean"]) / checkpoint["scaler_scale"]
    with torch.no_grad():
        replay = (
            torch.sigmoid(model(torch.tensor(transformed, dtype=torch.float32)))
            .numpy()
            .ravel()
        )
    predictions = np.loadtxt(
        Path(folder) / "predictions.csv", delimiter=",", skiprows=1
    )
    np.testing.assert_allclose(replay, predictions[:, 1], atol=1e-7)
    assert brier_score_loss(y_test, replay) == pytest.approx(
        record.data.metrics["brier_score"]
    )
    assert log_loss(y_test, replay) == pytest.approx(record.data.metrics["log_loss"])
