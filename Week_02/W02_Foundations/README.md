# Week 2 Foundations

A compact PyTorch exercise set covering tensor operations, a manually implemented linear layer, stable softmax and cross-entropy, a two-layer MLP, backpropagation, gradient checking, tensor rearrangement, and MNIST classification.

## Files

- tensor_drills.py: tensor creation, indexing, transpose, and matrix multiplication
- manual_linear.py: linear transformation implemented with trainable parameters
- softmax_cross_entropy.py: numerically stable softmax and cross-entropy
- two_layer_mlp.py: two-layer feed-forward classifier
- autograd_gradcheck.py: manual linear-layer gradients and PyTorch gradcheck
- einops_drills.py: rearrange, reduce, and repeat exercises, with native PyTorch fallback
- mnist_experiment.py: MNIST download, MLP training, and evaluation
- Repository tests: tests/test_w02_foundations.py

## Environment

The repository uses Python 3.11 and the dependencies declared in the root pyproject.toml. From the repository root, run:

    uv sync --locked

The Einops examples use the einops package when it is installed. Without it, equivalent tensor operations run through PyTorch.

## Run the exercises

From the repository root:

    uv run python -m Week_02.W02_Foundations.tensor_drills
    uv run python -m Week_02.W02_Foundations.manual_linear
    uv run python -m Week_02.W02_Foundations.softmax_cross_entropy
    uv run python -m Week_02.W02_Foundations.two_layer_mlp
    uv run python -m Week_02.W02_Foundations.autograd_gradcheck
    uv run python -m Week_02.W02_Foundations.einops_drills
    uv run python -m Week_02.W02_Foundations.mnist_experiment

The MNIST script downloads the official gzip IDX files into ~/.cache/ingen-mnist by default. It trains on up to 10,000 training examples and evaluates on up to 10,000 test examples. Set a different data directory or limit by calling run_experiment from Python.

## Run tests

    uv run pytest -q tests/test_w02_foundations.py

The tests use small in-memory tensors and do not download MNIST.
