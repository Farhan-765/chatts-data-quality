"""
Evaluation metrics for anomaly classification results.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def compute_metrics(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str] | None = None,
) -> dict:
    """
    Compute classification metrics comparing model predictions to ground truth.

    Parameters
    ----------
    predictions  : list of predicted category codes, e.g. ['A', 'B', 'E', ...]
    ground_truth : list of true category codes
    labels       : optional ordered label list (default: unique sorted labels)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix, report
    """
    if labels is None:
        labels = sorted(set(ground_truth + predictions))

    acc  = accuracy_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions, labels=labels,
                           average='weighted', zero_division=0)
    rec  = recall_score(ground_truth, predictions, labels=labels,
                        average='weighted', zero_division=0)
    f1   = f1_score(ground_truth, predictions, labels=labels,
                    average='weighted', zero_division=0)
    cm   = confusion_matrix(ground_truth, predictions, labels=labels)
    rep  = classification_report(ground_truth, predictions, labels=labels,
                                 zero_division=0)

    return {
        'accuracy':         acc,
        'precision':        prec,
        'recall':           rec,
        'f1':               f1,
        'confusion_matrix': cm,
        'report':           rep,
        'labels':           labels,
        'n_samples':        len(predictions),
        'n_correct':        int(sum(p == t for p, t in zip(predictions, ground_truth))),
    }


def threshold_sweep(
    probs: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Sweep classification threshold for binary anomaly detection.

    Parameters
    ----------
    probs       : model output probabilities, shape (N,)
    true_labels : binary ground truth, shape (N,)
    thresholds  : array of threshold values to test (default: 0.1..0.95)

    Returns
    -------
    dict with 'thresholds', 'f1', 'precision', 'recall', 'best_threshold', 'best_f1'
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.96, 0.05)

    f1s, precs, recs = [], [], []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1s.append(f1_score(true_labels, preds, zero_division=0))
        precs.append(precision_score(true_labels, preds, zero_division=0))
        recs.append(recall_score(true_labels, preds, zero_division=0))

    best_idx = int(np.argmax(f1s))
    return {
        'thresholds':      thresholds,
        'f1':              np.array(f1s),
        'precision':       np.array(precs),
        'recall':          np.array(recs),
        'best_threshold':  float(thresholds[best_idx]),
        'best_f1':         float(f1s[best_idx]),
    }
