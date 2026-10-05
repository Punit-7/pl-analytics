import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def binary_report(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "goals": int(y.sum()),
        "xg_sum": float(p.sum()),
    }


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates, over equal-count bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            total += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total)


def rps(p, outcome) -> np.ndarray:
    """Ranked probability score per match. p: (n, 3) as home/draw/away; outcome: 0/1/2."""
    p = np.asarray(p, dtype=float)
    o = np.zeros_like(p)
    o[np.arange(len(p)), outcome] = 1.0
    cum_p, cum_o = p.cumsum(axis=1), o.cumsum(axis=1)
    return ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(axis=1) / 2


def multiclass_log_loss(p, outcome) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return float(-np.log(p[np.arange(len(p)), outcome]).mean())
