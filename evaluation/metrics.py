"""Binary retinal segmentation metrics, applied per mask channel."""

import numpy as np


def binary_metrics(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    scores = probabilities.ravel()
    labels = truth.ravel().astype(bool)
    positive = labels.sum()
    negative = labels.size - positive
    predicted = scores >= 0.5
    tp = np.logical_and(predicted, labels).sum()
    tn = np.logical_and(~predicted, ~labels).sum()
    fp = np.logical_and(predicted, ~labels).sum()
    fn = np.logical_and(~predicted, labels).sum()
    eps = 1e-8
    values = {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "accuracy": (tp + tn) / labels.size,
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
    }
    if positive and negative:
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
        tps = np.r_[0, np.cumsum(sorted_labels)[distinct_ends]]
        fps = np.r_[0, 1 + distinct_ends - np.cumsum(sorted_labels)[distinct_ends]]
        values["auc"] = np.trapz(tps / positive, fps / negative)
    else:
        values["auc"] = float("nan")
    return values
