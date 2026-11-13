# Execution verification

- Python 3.11.18; PyTorch 2.5.1 CPU on Linux x86_64.
- `uv sync --locked`: passed.
- `ruff check .` and `black --check .`: passed.
- `pre-commit run --all-files`: Ruff and Black passed.
- `pytest -q`: 3 passed (one third-party Pydantic deprecation warning).
- Temporary deliberately failing test: pytest returned exit code 1 as expected;
  the test was removed afterwards.
- Two separate 200-epoch CLI runs: matching Brier score, log loss and ROC-AUC.
- Both runs read MLflow parameters, metrics, 200 loss steps and artifact names
  back successfully. The integration test downloaded weights, restored the
  scaler and reproduced all held-out prediction probabilities.

The JSON and CSV files alongside this note are real outputs. Resource measurements
vary across runs. GitHub Actions reports its own independent runner validation.

- MLflow UI startup check: HTTP 200 with an HTML response.
- First GitHub Actions attempt: https://github.com/cocofabio/Ingen-Intern/actions/runs/43365594417
  completed with failure before any job steps ran (runner_id 0, empty steps).
  The job-log endpoint returned 404, so the remote failure cause is unresolved.
  Local checks above passed; remote CI has not been validated as green.
