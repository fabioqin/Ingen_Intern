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
