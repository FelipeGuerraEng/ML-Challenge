from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def class_balance(y: np.ndarray) -> Dict[str, float]:
    """
    Return class counts and positive rate for a binary target array.
    """
    try:
        y = np.asarray(y).astype(int).ravel()
        n = y.shape[0]
        pos = int(np.sum(y))
        neg = int(n - pos)
        rate = float(pos / n) if n > 0 else 0.0
        return {"n": float(n), "positive": float(pos), "negative": float(neg), "positive_rate": rate}
    except Exception as e:
        raise RuntimeError(f"Failed to compute class balance: {e}")


def is_imbalanced(y: np.ndarray, minority_threshold: float = 0.35) -> bool:
    """
    Return True if the minority class proportion is below the threshold.
    """
    try:
        stats = class_balance(y)
        p = stats["positive_rate"]
        minority = min(p, 1.0 - p)
        return bool(minority < float(minority_threshold))
    except Exception as e:
        raise RuntimeError(f"Failed to detect imbalance: {e}")


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute ROC-AUC, PR-AUC, and Accuracy at the given threshold.
    """
    try:
        y_true = np.asarray(y_true).astype(int).ravel()
        y_proba = np.asarray(y_proba).astype(float).ravel()
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
        y_pred = (y_proba >= float(threshold)).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        return {"roc_auc": roc_auc, "pr_auc": pr_auc, "accuracy": acc, "threshold": float(threshold)}
    except Exception as e:
        raise RuntimeError(f"Failed to compute classification metrics: {e}")
