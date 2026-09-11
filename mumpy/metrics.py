"""mumpy.metrics: evaluation metrics for ML/DL (pure numpy)."""
from __future__ import annotations

import numpy as np

__all__ = [
    "mse", "rmse", "mae", "mape", "r2_score", "explained_variance",
    "accuracy", "precision", "recall", "f1", "confusion_matrix",
    "classification_report", "log_loss", "roc_auc", "silhouette_score",
]


def mse(y_true, y_pred):
    y_true = np.asanyarray(y_true, dtype=float).ravel()
    y_pred = np.asanyarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true = np.asanyarray(y_true, dtype=float).ravel()
    y_pred = np.asanyarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true = np.asanyarray(y_true, dtype=float).ravel()
    y_pred = np.asanyarray(y_pred, dtype=float).ravel()
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score(y_true, y_pred):
    y_true = np.asanyarray(y_true, dtype=float).ravel()
    y_pred = np.asanyarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def explained_variance(y_true, y_pred):
    y_true = np.asanyarray(y_true, dtype=float).ravel()
    y_pred = np.asanyarray(y_pred, dtype=float).ravel()
    return float(1 - np.var(y_true - y_pred) / np.var(y_true)) if np.var(y_true) != 0 else 0.0


def accuracy(y_true, y_pred):
    y_true = np.asanyarray(y_true).ravel()
    y_pred = np.asanyarray(y_pred).ravel()
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asanyarray(y_true).ravel()
    y_pred = np.asanyarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asanyarray(labels)
    idx = {v: i for i, v in enumerate(labels.tolist())}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m, labels


def _prf(y_true, y_pred, average="binary", pos_label=1):
    y_true = np.asanyarray(y_true).ravel()
    y_pred = np.asanyarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
        fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
        fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f
    ps, rs, fs = [], [], []
    for lab in labels:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    if average == "macro":
        return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))
    # micro
    tp = sum(int(((y_pred == lab) & (y_true == lab)).sum()) for lab in labels)
    fp = sum(int(((y_pred == lab) & (y_true != lab)).sum()) for lab in labels)
    fn = sum(int(((y_pred != lab) & (y_true == lab)).sum()) for lab in labels)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def precision(y_true, y_pred, average="binary", pos_label=1):
    return _prf(y_true, y_pred, average, pos_label)[0]


def recall(y_true, y_pred, average="binary", pos_label=1):
    return _prf(y_true, y_pred, average, pos_label)[1]


def f1(y_true, y_pred, average="binary", pos_label=1):
    return _prf(y_true, y_pred, average, pos_label)[2]


def classification_report(y_true, y_pred):
    y_true = np.asanyarray(y_true).ravel()
    y_pred = np.asanyarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    lines = {}
    for lab in labels.tolist():
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        sup = int((y_true == lab).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        lines[str(lab)] = {"precision": p, "recall": r, "f1": f, "support": sup}
    lines["accuracy"] = float((y_true == y_pred).mean())
    return lines


def log_loss(y_true, y_pred, eps=1e-15):
    y_true = np.asanyarray(y_true, dtype=float)
    y_pred = np.clip(np.asanyarray(y_pred, dtype=float), eps, 1 - eps)
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    # multiclass: y_true = class indices
    n = y_true.shape[0]
    return float(-np.mean(np.log(y_pred[np.arange(n), y_true.astype(int).ravel()])))


def roc_auc(y_true, y_score):
    y_true = np.asanyarray(y_true).ravel()
    y_score = np.asanyarray(y_score, dtype=float).ravel()
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_true) + 1)
    sum_rank_pos = ranks[y_true == 1].sum()
    return float((sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def silhouette_score(X, labels):
    X = np.asanyarray(X, dtype=float)
    labels = np.asanyarray(labels).ravel()
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    # pairwise distances (ok for moderate n; sample if huge)
    n = len(X)
    if n > 2000:
        idx = np.random.default_rng(0).choice(n, 2000, replace=False)
        X, labels = X[idx], labels[idx]
        n = 2000
    try:
        from scipy.spatial.distance import cdist  # optional fast path
        D = cdist(X, X)
    except Exception:
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    sils = []
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        other = labels != labels[i]
        a = D[i][same].mean() if same.any() else 0.0
        b = min(D[i][labels == u].mean() for u in uniq if u != labels[i])
        sils.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(sils))
