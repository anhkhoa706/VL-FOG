import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .log import log


def calculate_metrics(targets, preds, probs, log_file):
    """
    Calculate and log evaluation metrics.
    Args:
        targets: ground truth labels
        preds: predicted labels
        probs: predicted probabilities for class 1
        log_file: file handle for logging

    Returns:
        final_score: weighted average of F1, accuracy, AUC
        acc: accuracy
        f1: macro F1-score
        auc: ROC-AUC
    """
    # Compute metrics
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.0  # Only one class present in y_true

    final_score = 0.5 * f1 + 0.3 * acc + 0.2 * auc

    return final_score, acc, f1, auc

def find_best_threshold_weighted(y_true, y_probs):
    """
    Search threshold that maximizes weighted final score:
    final_score = 0.5 * F1 + 0.3 * Accuracy + 0.2 * AUC
    """
    best_score, best_t = 0.0, 0.5
    for t in [i / 100 for i in range(20, 81)]:  # test 0.20 → 0.80
        preds = [1 if p > t else 0 for p in y_probs]

        f1 = f1_score(y_true, preds, average="macro")
        acc = accuracy_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, y_probs)
        except:
            auc = 0.0  # fallback for edge cases

        final_score = 0.5 * f1 + 0.3 * acc + 0.2 * auc

        if final_score > best_score:
            best_score = final_score
            best_t = t

    return best_t 