# Week 02 — Intrusion classifier

I compared three models on UNSW-NB15. After removing duplicate flows, I used 16,000 rows for training, 4,000 for validation and 10,000 for testing. I dropped the ID and attack category and fitted preprocessing on the training rows.

| Model | Precision | Recall | ROC-AUC | TPR at ≤1% FPR |
|---|---:|---:|---:|---:|
| Majority class | 0.000 | 0.000 | 0.500 | 0.000 |
| Logistic regression | 0.574 | 0.947 | 0.902 | 0.424 |
| MLP, mean ± SD | 0.629 ± 0.012 | 0.948 ± 0.004 | 0.937 ± 0.004 | 0.556 ± 0.008 |

I tried two learning rates and kept the epoch with the lowest validation loss for each of three seeds. At the default threshold, logistic regression and the MLP both caught about 95% of attacks. With the false-alarm rate capped at 1% on this test set, the MLP caught about 56% and logistic regression caught 42%. I would check that gap on another capture.

![Loss, ROC and precision–recall curves](evidence/curves.png)

Run `uv run python -m Week_02.intrusion` from the repo root. [Results by seed](evidence/metrics.json) · [Training losses](evidence/sweep.csv) · [UNSW-NB15 dataset and papers, Moustafa & Slay](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
