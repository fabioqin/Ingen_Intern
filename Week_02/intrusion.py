"""Reproduce the offline UNSW-NB15 binary classification benchmark."""

import copy
import hashlib
import io
import json
import os
import subprocess
from dataclasses import asdict, dataclass
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
OUT = Path(__file__).resolve().parent / "evidence"


@dataclass(frozen=True)
class Config:
    development_size: int = 20000
    test_size: int = 10000
    validation_fraction: float = 0.2
    split_seed: int = 42
    seeds: tuple = (7, 21, 42)
    learning_rates: tuple = (0.001, 0.003)
    hidden: int = 64
    batch_size: int = 256
    patience: int = 5
    max_epochs: int = 50
    fpr_budget: float = 0.01


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


def prepare(config):
    train, test = (load_partition(k) for k in BLOBS)
    fields = train.columns.drop(["id", "attack_cat", "label"]).tolist()
    # Keep one copy of each feature vector, giving training partition priority.
    combined = pd.concat([train.assign(part="train"), test.assign(part="test")])
    unique = combined.drop_duplicates(subset=fields)
    dev = unique[unique.part == "train"]
    held = unique[unique.part == "test"]
    dev, _ = train_test_split(
        dev,
        train_size=config.development_size,
        stratify=dev.label,
        random_state=config.split_seed,
    )
    fit, val = train_test_split(
        dev,
        test_size=config.validation_fraction,
        stratify=dev.label,
        random_state=config.split_seed,
    )
    held, _ = train_test_split(
        held,
        train_size=config.test_size,
        stratify=held.label,
        random_state=config.split_seed,
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


def fit_mlp(data, seed, lr, config):
    (x, y), (v, z), _ = data
    torch.manual_seed(seed)
    model = TwoLayerMLP(x.shape[1], config.hidden, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history, best, best_epoch = [], float("inf"), 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x)).split(config.batch_size):
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
        if epoch - best_epoch >= config.patience:
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


def select_threshold(labels, scores, budget):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= budget)
    # ROC thresholds descend. On equal recall, keep the stricter threshold.
    threshold = thresholds[eligible[np.argmax(tpr[eligible])]]
    return float(threshold) if np.isfinite(threshold) else float(np.nextafter(1.0, 2.0))


def operating_point(labels, scores, threshold):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return dict(
        precision=float(tp / (tp + fp)) if tp + fp else 0.0,
        recall=float(tp / (tp + fn)),
        fpr=float(fp / (fp + tn)),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )


def main(config=Config(), out=OUT, tracking_uri="sqlite:///mlflow.db"):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    data, names = prepare(config)
    (x, y), (validation, v), (test, target) = data
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("week02-unsw-nb15")
    revision = os.environ.get("GITHUB_SHA") or (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "local"
    )
    scores = {"majority": np.full(len(test), float(y.float().mean()))}
    val_scores = {"majority": np.full(len(v), float(y.float().mean()))}
    logistic = LogisticRegression(max_iter=1000).fit(x, y)
    scores["logistic"] = logistic.predict_proba(test)[:, 1]
    val_scores["logistic"] = logistic.predict_proba(validation)[:, 1]
    coef = logistic.coef_[0]
    top = {names[i]: float(coef[i]) for i in np.argsort(abs(coef))[-5:][::-1]}
    selections = {}
    sweep = []
    deployment = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for seed in config.seeds:
        candidates = []
        for lr in config.learning_rates:
            with mlflow.start_run(run_name=f"mlp-{seed}-{lr}"):
                mlflow.log_params(
                    dict(
                        seed=seed,
                        learning_rate=lr,
                        hidden=config.hidden,
                        batch_size=config.batch_size,
                        patience=config.patience,
                        max_epochs=config.max_epochs,
                        split_seed=config.split_seed,
                        git_revision=revision,
                    )
                )
                trial = fit_mlp(data, seed, lr, config)
                candidates.append((*trial, lr))
                for step, (train_loss, val_loss) in enumerate(trial[2], 1):
                    sweep.append(
                        dict(
                            run=f"mlp-{seed}-{lr}",
                            epoch=step,
                            train_loss=train_loss,
                            validation_loss=val_loss,
                        )
                    )
        loss, model, history, epoch, lr = min(candidates, key=lambda c: c[0])
        key = f"mlp-{seed}"
        with torch.no_grad():
            val_scores[key] = model(validation).softmax(1)[:, 1].numpy()
            scores[key] = model(test).softmax(1)[:, 1].numpy()
        selections[key] = dict(epoch=epoch, lr=lr, validation_loss=loss)
        if seed == config.seeds[-1]:
            axes[0].plot(
                np.arange(1, len(history) + 1), history, label=["Train", "Validation"]
            )
            axes[0].axvline(epoch, linestyle=":", color="gray")
    rows = {}
    matrices = {}
    for key, score in scores.items():
        rows[key] = metrics(target, score)
        matrices[key] = confusion_matrix(target, score >= 0.5).tolist()
        threshold = select_threshold(v, val_scores[key], config.fpr_budget)
        point = operating_point(target, score, threshold)
        deployment[key] = dict(
            threshold=threshold,
            validation=operating_point(v, val_scores[key], threshold),
            test=point,
        )
        with mlflow.start_run(run_name=f"evaluation-{key}"):
            mlflow.log_param("git_revision", revision)
            mlflow.log_metrics(rows[key])
            mlflow.log_metrics(
                {f"fixed_test_{k}": point[k] for k in ("precision", "recall", "fpr")}
            )
            mlflow.log_param("validation_threshold", threshold)
        if key in ("majority", "logistic", f"mlp-{config.seeds[-1]}"):
            fpr, tpr, _ = roc_curve(target, score)
            precision, recall, _ = precision_recall_curve(target, score)
            axes[1].plot(fpr, tpr, label=key)
            axes[2].step(recall, precision, where="post", label=key)
    for ax, title, xlabel, ylabel in zip(
        axes,
        [f"MLP loss (seed {config.seeds[-1]})", "Test ROC", "Test precision–recall"],
        ["Epoch", "False positive rate", "Recall"],
        ["Cross-entropy", "True positive rate", "Precision"],
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    result = dict(
        metrics=rows,
        deployment=deployment,
        confusion=matrices,
        selections=selections,
        top_logistic_coefficients=top,
        split_counts=[[len(labels), int(labels.sum())] for labels in (y, v, target)],
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(sweep).to_csv(out / "sweep.csv", index=False)
    settings = dict(
        **asdict(config),
        input_dim=len(names),
        output_classes=2,
        optimizer="Adam",
        git_revision=revision,
        source=SOURCE,
        blobs=BLOBS,
    )
    (out / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    summary = [
        "# UNSW-NB15 results",
        "",
        f"MLP: {len(names)} inputs, {config.hidden} ReLU units, 2 output classes.",
        "",
        "Each threshold was chosen on validation flows to maximize recall within the false-alarm budget. The last three columns use that same threshold on the held-out test set.",
        "",
        "| Model | Threshold | Validation FPR | Test FPR | Test precision | Test recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in deployment.items():
        pt = value["test"]
        threshold_text = (
            f"{value['threshold']:.6f}" if value["threshold"] <= 1 else ">1"
        )
        summary.append(
            f"| {key} | {threshold_text} | {value['validation']['fpr']:.2%} | {pt['fpr']:.2%} | {pt['precision']:.2%} | {pt['recall']:.2%} |"
        )
    summary += [
        "",
        "Largest logistic coefficients by absolute value (signed): "
        + ", ".join(f"{name} ({weight:+.2f})" for name, weight in top.items())
        + ".",
        "",
        "![Loss, ROC and precision–recall curves](curves.png)",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"Saved intrusion experiment to {out}")


if __name__ == "__main__":
    main()
