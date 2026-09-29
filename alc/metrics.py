"""Evaluation metrics.

The Interspeech 2011 Speaker State Challenge, which introduced the ALC
intoxication task, ranks systems by unweighted average recall (UAR). With
about two sober recordings for every intoxicated one, accuracy rewards a
model that mostly predicts "sober"; UAR does not.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


def uar(y_true, y_pred):
    """Unweighted average recall (mean of per-class recalls)."""
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def summary(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "uar": float(uar(y_true, y_pred)),
        "accuracy": float((tp + tn) / len(y_true)),
        "recall_sober": float(tn / (tn + fp)) if tn + fp else 0.0,
        "recall_intoxicated": float(tp / (tp + fn)) if tp + fn else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
