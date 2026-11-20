Run `uv sync --locked` to set up the project, `uv run pytest -q` for tests, and `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` to view tracked runs.

| Week | Experiment | Run |
|---|---|---|
| 1 | [PyTorch logistic regression](Week_01/W01_Repo_Skeleton/) | `uv run python -m safety_lens.experiment` |
| 2 | [MNIST](Week_02/W02_Foundations_MNIST.ipynb) and [intrusion detection](Week_02/W02_Intrusion_Classifier_Report.md) | `uv run python -m Week_02.W02_Foundations.mnist_experiment` and `uv run python -m Week_02.intrusion` |
