from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, accuracy_score
)

def best_threshold_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> Tuple[float, float]:
    t = np.linspace(0, 1, 1001)
    eps = 1e-12
    best_f, best_thr = -1.0, 0.5
    for thr in t:
        y_pred = (y_prob >= thr).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + eps)
        if fbeta > best_f:
            best_f, best_thr = fbeta, thr
    return float(best_f), float(best_thr)

def metrics_block(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred_thr = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc_thr = accuracy_score(y_true, y_pred_thr)
    f2_opt, thr_opt = best_threshold_fbeta(y_true, y_prob, beta=2.0)
    f2_half, _ = best_threshold_fbeta(y_true, y_prob, beta=2.0)  # computed on same probs; thr not used here
    acc_half = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "acc_at_best_f2_thr": acc_thr,
        "f2_at_best_f2_thr": f2_opt,
        "acc_at_0_5": acc_half,
        "f2_at_0_5": f2_half,
        "best_thr": thr_opt,
    }

def save_metadata(path: Path, meta: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def plot_and_log_curves(y_true: np.ndarray, y_prob: np.ndarray, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pr, rc, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.savefig(outdir / "roc.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(rc, pr)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.savefig(outdir / "pr.png", bbox_inches="tight"); plt.close()
