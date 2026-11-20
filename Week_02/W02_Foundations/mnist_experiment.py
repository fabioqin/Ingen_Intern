import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"


class MNISTMLP(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, images):
        return self.network(images)


def _download(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MNIST_BASE_URL}/{path.name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def _read_images(path):
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    return values.reshape(count, rows * columns).copy()


def _read_labels(path):
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    return values.reshape(count).copy()


def load_mnist(data_dir=None, download=True, limit=None):
    root = (
        Path(data_dir)
        if data_dir is not None
        else Path.home() / ".cache" / "ingen-mnist"
    )
    paths = {key: root / filename for key, filename in MNIST_FILES.items()}

    for path in paths.values():
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            _download(path)

    train_images = _read_images(paths["train_images"])
    train_labels = _read_labels(paths["train_labels"])
    test_images = _read_images(paths["test_images"])
    test_labels = _read_labels(paths["test_labels"])

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        train_images = train_images[:limit]
        train_labels = train_labels[:limit]
        test_images = test_images[:limit]
        test_labels = test_labels[:limit]

    train_x = torch.from_numpy(train_images).float().div_(255)
    train_y = torch.from_numpy(train_labels).long()
    test_x = torch.from_numpy(test_images).float().div_(255)
    test_y = torch.from_numpy(test_labels).long()

    return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)


def train_epoch(model, loader, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    if total_items == 0:
        raise ValueError("Training loader is empty.")
    return total_loss / total_items


def evaluate(model, loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_items = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    if total_items == 0:
        raise ValueError("Evaluation loader is empty.")
    return total_loss / total_items, correct / total_items


def run_experiment(
    data_dir=None,
    epochs=1,
    batch_size=128,
    limit=10000,
    seed=42,
    device=None,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    torch.manual_seed(seed)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_data, test_data = load_mnist(data_dir=data_dir, limit=limit)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = MNISTMLP().to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, selected_device)
        test_loss, accuracy = evaluate(model, test_loader, selected_device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": accuracy,
            }
        )

    return {
        "model": model,
        "history": history,
        "device": str(selected_device),
    }


if __name__ == "__main__":
    result = run_experiment()
    print(result["history"][-1])
