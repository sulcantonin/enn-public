"""OOD-detection and calibration metrics. Convention: higher score = more OOD."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def fpr_at_95_tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """FPR at 95% TPR (OOD = positive class), linearly interpolated on the ROC curve."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    if tpr.max() < 0.95:
        return float(fpr[-1])
    return float(np.interp(0.95, tpr, fpr))


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """AUROC, FPR95, AUPR-In and AUPR-Out for one ID/OOD pair."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return {
        'AUROC': float(roc_auc_score(labels, scores)),
        'FPR95': fpr_at_95_tpr(id_scores, ood_scores),
        'AUPR_IN': float(average_precision_score(1 - labels, -scores)),
        'AUPR_OUT': float(average_precision_score(labels, scores)),
    }


def expected_calibration_error(confidences, predictions, labels, n_bins: int = 15) -> float:
    """Top-label ECE with ``n_bins`` equal-width bins on [0, 1]."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = (np.asarray(predictions) == np.asarray(labels)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        if not np.any(in_bin):
            continue
        ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)
