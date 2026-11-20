"""Reproduce the offline UNSW-NB15 binary classification benchmark."""

import copy
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from Week_02.W02_Foundations.two_layer_mlp import TwoLayerMLP

SOURCE = "https://raw.githubusercontent.com/Sambhram-Gaonkar/IDNResearch/d9901df38911adfe4904ae7d6e4ab7c4dd0bf74f/data"
BLOBS = {
    "training": "35b624e416c02f8ab5baa649b317ad9c8f9961fe",
    "testing": "928975eafe38c3c236df957ef7cf75dcd01831a1",
}
OUT = Path("outputs/week02")


def load_partition(name):
    path = Path.home() / ".cache" / "ingen-unsw" / f"{name}.csv"
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(f"{SOURCE}/UNSW_NB15_{name}-set.csv", timeout=120) as response:
            raw = response.read()
    header = f"blob {len(raw)}".encode() + bytes([0])
    if hashlib.sha1(header + raw).hexdigest() != BLOBS[name]:
        raise ValueError(f"Unexpected contents for {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return pd.read_csv(io.BytesIO(raw))


def prepare():
    train, test = (load_partition(k) for k in BLOBS)
    fields = train.columns.drop(["id", "attack_cat", "label"]).tolist()
    # Keep one copy of each feature vector, giving training partition priority.
    combined = pd.concat([train.assign(part="train"), test.assign(part="test")])
    unique = combined.drop_duplicates(subset=fields)
    dev = unique[unique.part == "train"]
    held = unique[unique.part == "test"]
    dev, _ = train_test_split(
        dev, train_size=20000, stratify=dev.label, random_state=42
    )
    fit, val = train_test_split(dev, test_size=0.2, stratify=dev.label, random_state=42)
    held, _ = train_test_split(
        held, train_size=10000, stratify=held.label, random_state=42
    )
    categories = ["proto", "service", "state"]
    numeric = [x for x in fields if x not in categories]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categories,
            ),
        ]
    ).fit(fit[fields])
    data = [
        (
            torch.from_numpy(transform.transform(d[fields]).astype("float32")),
            torch.from_numpy(d.label.to_numpy(dtype="int64")),
        )
        for d in (fit, val, held)
    ]
    return data, transform.get_feature_names_out()


def fit_mlp(data, seed, lr):
    (x, y), (v, z), _ = data
    torch.manual_seed(seed)
    model = TwoLayerMLP(x.shape[1], 64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history, best, best_epoch = [], float("inf"), 0
    for epoch in range(1, 51):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x)).split(256):
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(v), z).item()
        history.append([total / len(x), val_loss])
        mlflow.log_metrics(
            {"train_loss": history[-1][0], "validation_loss": val_loss}, step=epoch
        )
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(weights)
    return best, model, history, best_epoch


def metrics(y, score):
    pred = score >= 0.5
    fpr, tpr, _ = roc_curve(y, score)
    return dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred),
        roc_auc=roc_auc_score(y, score),
        average_precision=average_precision_score(y, score),
        tpr_at_1pct_fpr=float(tpr[fpr <= 0.01].max()),
    )


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    data, names = prepare()
    (x, y), (_, v), (test, target) = data
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("week02-unsw-nb15")
    revision = os.environ.get("GITHUB_SHA") or (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "local"
    )
    scores = {"majority": np.full(len(test), float(y.float().mean()))}
    logistic = LogisticRegression(max_iter=1000).fit(x, y)
    scores["logistic"] = logistic.predict_proba(test)[:, 1]
    coef = logistic.coef_[0]
    top = {names[i]: float(coef[i]) for i in np.argsort(abs(coef))[-5:][::-1]}
    selections = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for seed in (7, 21, 42):
        candidates = []
        for lr in (0.001, 0.003):
            with mlflow.start_run(run_name=f"mlp-{seed}-{lr}"):
                mlflow.log_params(
                    dict(
                        seed=seed,
                        learning_rate=lr,
                        hidden=64,
                        batch_size=256,
                        patience=5,
                        max_epochs=50,
                        split_seed=42,
                        git_revision=revision,
                    )
                )
                candidates.append((*fit_mlp(data, seed, lr), lr))
        loss, model, history, epoch, lr = min(candidates, key=lambda c: c[0])
        key = f"mlp-{seed}"
        with torch.no_grad():
            scores[key] = model(test).softmax(1)[:, 1].numpy()
        selections[key] = dict(epoch=epoch, lr=lr, validation_loss=loss)
        if seed == 42:
            axes[0].plot(
                np.arange(1, len(history) + 1), history, label=["Train", "Validation"]
            )
            axes[0].axvline(epoch, linestyle=":", color="gray")
    rows = {}
    matrices = {}
    for key, score in scores.items():
        rows[key] = metrics(target, score)
        matrices[key] = confusion_matrix(target, score >= 0.5).tolist()
        with mlflow.start_run(run_name=f"evaluation-{key}"):
            mlflow.log_param("git_revision", revision)
            mlflow.log_metrics(rows[key])
        if key in ("majority", "logistic", "mlp-42"):
            fpr, tpr, _ = roc_curve(target, score)
            precision, recall, _ = precision_recall_curve(target, score)
            axes[1].plot(fpr, tpr, label=key)
            axes[2].step(recall, precision, where="post", label=key)
    for ax, title, xlabel, ylabel in zip(
        axes,
        ["MLP loss (seed 42)", "Test ROC", "Test precision–recall"],
        ["Epoch", "False positive rate", "Recall"],
        ["Cross-entropy", "True positive rate", "Precision"],
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
    fig.savefig(OUT / "curves.png", dpi=140)
    plt.close(fig)
    result = dict(
        metrics=rows,
        confusion=matrices,
        selections=selections,
        top_logistic_coefficients=top,
        split_counts=[[len(labels), int(labels.sum())] for labels in (y, v, target)],
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
