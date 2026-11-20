# Week 02 — Intrusion classifier

On 10,000 held-out UNSW-NB15 flows, the two-layer MLP detected 55.57% of attacks at an empirical false-positive rate of at most 1%, averaged across three training seeds. Logistic regression detected 42.40% under the same constraint. The 13.17 percentage-point difference matters for a detector whose operator can investigate only a limited number of alerts.

## Data and evaluation

Each row represents a labeled network flow. The positive class is an attack. We used the published training and testing partitions, dropped the record ID, attack category and binary label from the features, and removed duplicate feature vectors before sampling. The training partition took precedence when the same vector appeared in both files. A stratified sample of 20,000 training-partition flows was split into 16,000 fit and 4,000 validation examples. The test sample contained 10,000 flows from the separate test partition. The numbers of attacks were 7,795, 1,949 and 3,606, respectively. Split seed 42 stayed fixed across all runs.

Numeric features were standardized. Protocol, service and connection state were one-hot encoded. All transformations were fitted on the training subset. A constant training-prevalence score supplied the majority baseline. The second baseline was logistic regression. The MLP used a 64-unit ReLU hidden layer, two output logits, cross-entropy, Adam and batches of 256. For each seed we trained at learning rates 0.001 and 0.003 for up to 50 epochs, stopped after five epochs without a new lowest validation loss, and restored the best checkpoint. Validation loss also selected the learning rate.

## Test results

| Model | Precision | Recall | ROC-AUC | Average precision | TPR at ≤1% FPR |
|---|---:|---:|---:|---:|---:|
| Majority | 0.0000 | 0.0000 | 0.5000 | 0.3606 | 0.0000 |
| Logistic regression | 0.5742 | 0.9473 | 0.9021 | 0.8557 | 0.4240 |
| MLP, seed 7 | 0.6154 | 0.9451 | 0.9324 | 0.8996 | 0.5477 |
| MLP, seed 21 | 0.6352 | 0.9520 | 0.9391 | 0.9081 | 0.5635 |
| MLP, seed 42 | 0.6356 | 0.9479 | 0.9386 | 0.9082 | 0.5560 |
| MLP, mean ± sample SD | 0.6287 ± 0.0115 | 0.9483 ± 0.0035 | 0.9367 ± 0.0037 | 0.9053 ± 0.0049 | 0.5557 ± 0.0079 |

Precision and recall use a score threshold of 0.5. The majority baseline raises zero alerts, so its precision is recorded as zero by convention. For each model, the last column takes the highest observed test ROC point satisfying the 1% budget, which allows up to 63 false alarms among 6,394 benign flows. An operational threshold would need separate calibration and prospective validation.

![Training and validation loss, ROC and precision–recall curves](evidence/curves.png)

At the default threshold the seed-42 model produced 3,418 true positives, 188 missed attacks, 1,960 false alarms and 4,434 true negatives. Its 94.79% recall comes with a 30.65% benign false-alarm rate. The gap between that result and 55.60% recall at the tighter operating point explains why overall recall alone gives an incomplete picture.

The best checkpoints occurred at epochs 25, 18 and 29 for seeds 7, 21 and 42. For seed 42 validation loss reached 0.1982 at epoch 29 and rose to 0.2009 by epoch 34, when training loss was 0.1757. The strongest logistic coefficients by absolute magnitude included destination TTL (+6.12) and destination load (−3.55). These are associations conditional on the encoding and correlated features. The fixed split measures variation from training seeds only. Exact deduplication still allows related flows from the same capture, and the benchmark's attack mix and network environment limit generalization.

## Reproduce

From the repository root, run the following commands with network access for the first dataset download.

```sh
uv sync --locked
uv run python -m Week_02.W02_Foundations.mnist_experiment
uv run python -m Week_02.intrusion
uv run pytest -q
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The intrusion script verifies both downloaded CSVs against pinned Git blob IDs, writes metrics and plots to `outputs/week02/`, and logs the six training trials and five evaluations to the MLflow experiment `week02-unsw-nb15`. The MNIST CPU run used seed 42, one epoch and 10,000 examples in each split. Its training loss was 0.9346, test loss was 0.4293, and test accuracy was 88.42%.

[Machine-readable results](evidence/metrics.json) · [Learning-rate sweep by epoch](evidence/sweep.csv) · [MNIST result](evidence/mnist.json) · [Experiment code at evaluation](https://github.com/fabioqin/Ingen_Intern/tree/69eeddd6921184cc9c53139af6a1e9a863079a63/Week_02)

## Data references

N. Moustafa and J. Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS, 2015. N. Moustafa and J. Slay, *The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset*, Information Security Journal, 2016. [Dataset authors and partition description](https://research.unsw.edu.au/projects/unsw-nb15-dataset) · [Pinned CSV mirror](https://github.com/Sambhram-Gaonkar/IDNResearch/tree/d9901df38911adfe4904ae7d6e4ab7c4dd0bf74f/data)
