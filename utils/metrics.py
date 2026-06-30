"""
Evaluation metrics matching Section 5.4:
    Detection: ACC, PRE, REC, F1, MCC, AUC-ROC                     Eq. (60-63)
    Trust-aware: mean trust T, mean risk R, HRR                    Eq. (64-66)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def detection_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """y_true: binary ground truth (1=authentic). y_prob: predicted A(I) in [0,1]."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred, zero_division=0),
        "REC": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred) if len(set(y_true.tolist())) > 1 else 0.0,
    }
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["AUC"] = float("nan")
    try:
        metrics["AP"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["AP"] = float("nan")
    return metrics


def roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


def trust_aware_metrics(trust: np.ndarray, risk: np.ndarray, decisions: np.ndarray) -> dict:
    """Eq. (64-66): average trust, average risk, human review rate."""
    n = len(decisions)
    hrr = float((decisions == 1).sum()) / max(n, 1)  # Decision.HUMAN_REVIEW == 1
    return {
        "mean_trust": float(np.mean(trust)),
        "mean_risk": float(np.mean(risk)),
        "HRR": hrr,
        "verified_pct": float((decisions == 0).sum()) / max(n, 1),
        "human_review_pct": hrr,
        "rejected_pct": float((decisions == 2).sum()) / max(n, 1),
    }


def computational_metrics(latency_ms: float, num_params: float, throughput_fps: float | None = None) -> dict:
    out = {"latency_ms": latency_ms, "params_M": num_params}
    if throughput_fps is None and latency_ms > 0:
        throughput_fps = 1000.0 / latency_ms
    out["fps"] = throughput_fps
    return out
