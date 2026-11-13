"""Textbook logistic regression: linear logits + binary cross entropy."""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from safety_lens.seed import seed_everything


def train(seed=42, epochs=200, learning_rate=0.1):
    seed_everything(seed)
    torch.set_num_threads(1)
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        class_sep=1.0,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )
    scaler = StandardScaler().fit(x_train)
    x_train = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    target = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x_train), target
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_test)).numpy().ravel()
    metrics = {
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    predictions = np.column_stack((y_test, probabilities))
    return metrics, losses, checkpoint, predictions
