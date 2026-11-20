import numpy as np

from Week_02.intrusion import metrics


def test_false_alarm_budget_respects_ties():
    labels = np.array([0, 0, 1, 1])
    result = metrics(labels, np.array([0.1, 0.8, 0.8, 0.9]))
    assert result["tpr_at_1pct_fpr"] == 0.5
    assert result["recall"] == 1.0


def test_constant_baseline_at_low_false_alarm_rate():
    result = metrics(np.array([0, 1]), np.array([0.6, 0.6]))
    assert result["roc_auc"] == 0.5
    assert result["tpr_at_1pct_fpr"] == 0


def test_validation_threshold_keeps_tied_scores_together():
    from Week_02.intrusion import select_threshold

    scores = np.array([0.1, 0.8, 0.8, 0.9])
    assert select_threshold([0, 0, 1, 1], scores, 0.01) == 0.9
    assert select_threshold([0, 1], np.array([0.6, 0.6]), 0.01) > 1.0


def test_intrusion_pipeline(tmp_path, monkeypatch):
    import hashlib
    import json

    import pandas as pd
    from mlflow.tracking import MlflowClient

    from Week_02 import intrusion

    # Local CSV mirrors exercise the downloader and checksum without network I/O.
    monkeypatch.setattr(intrusion, "SOURCE", tmp_path.as_uri())
    monkeypatch.setattr(intrusion.Path, "home", lambda: tmp_path)
    for partition, offset in [("training", 0), ("testing", 100)]:
        ids = np.arange(offset, offset + 80)
        rows = pd.DataFrame(
            dict(
                id=ids,
                dur=ids / 10,
                service="dns",
                state="FIN",
                proto=np.where(ids % 3, "tcp", "udp") if offset == 0 else "unseen",
                attack_cat=np.where(ids % 2, "Attack", "Normal"),
                label=ids % 2,
            )
        )
        raw = rows.to_csv(index=False).encode()
        (tmp_path / f"UNSW_NB15_{partition}-set.csv").write_bytes(raw)
        digest = hashlib.sha1(
            f"blob {len(raw)}".encode() + bytes([0]) + raw
        ).hexdigest()
        monkeypatch.setitem(intrusion.BLOBS, partition, digest)
    cfg = intrusion.Config(development_size=50, test_size=30, seeds=(7,), max_epochs=2)
    out = tmp_path / "evidence"
    uri = f"sqlite:///{tmp_path / 'runs.db'}"
    old_uri = intrusion.mlflow.get_tracking_uri()
    try:
        intrusion.main(cfg, out, uri)
    finally:
        intrusion.mlflow.set_tracking_uri(old_uri)
    results = json.loads((out / "metrics.json").read_text())
    assert [n for n, _ in results["split_counts"]] == [40, 10, 30]
    assert set(results["deployment"]) == {"majority", "logistic", "mlp-7"}
    for point in results["deployment"].values():
        assert point["validation"]["fpr"] <= cfg.fpr_budget
        assert np.sum(point["test"]["confusion"]) == 30
    sweep = pd.read_csv(out / "sweep.csv")
    assert sweep.groupby("run").size().tolist() == [2, 2]
    assert json.loads((out / "config.json").read_text())["input_dim"] == 5
    assert (out / "curves.png").read_bytes().startswith(b"\x89PNG")
    assert "mlp-7" in (out / "summary.md").read_text()
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("week02-unsw-nb15")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 5 and all(r.info.status == "FINISHED" for r in runs)
