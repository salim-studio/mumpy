"""mumpy.stats: descriptive statistics & exploratory analysis for analysts."""
from __future__ import annotations

import numpy as np

__all__ = [
    "describe", "corr", "cov", "zscore", "iqr_outliers", "z_outliers",
    "histogram", "percentile", "mode", "skew", "kurtosis",
    "correlation_matrix", "crosstab", "value_counts",
]


def describe(a, percentiles=(25, 50, 75)):
    a = np.asanyarray(a, dtype=float).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        **{f"{int(p)}%": float(np.percentile(a, p)) for p in percentiles},
        "max": float(a.max()),
    }


def corr(x, y):
    x = np.asanyarray(x, dtype=float).ravel()
    y = np.asanyarray(y, dtype=float).ravel()
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cov(x, y, ddof=0):
    x = np.asanyarray(x, dtype=float).ravel()
    y = np.asanyarray(y, dtype=float).ravel()
    mask = ~(np.isnan(x) | np.isnan(y))
    return float(np.cov(x[mask], y[mask], ddof=ddof)[0, 1])


def correlation_matrix(X):
    X = np.asanyarray(X, dtype=float)
    col_mean = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X = X.copy()
    X[idx] = np.take(col_mean, idx[1])
    return np.corrcoef(X, rowvar=False)


def zscore(a, axis=None, ddof=0):
    a = np.asanyarray(a, dtype=float)
    m = np.nanmean(a, axis=axis, keepdims=True)
    s = np.nanstd(a, axis=axis, ddof=ddof, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (a - m) / s


def iqr_outliers(a, factor=1.5):
    a = np.asanyarray(a, dtype=float).ravel()
    q1, q3 = np.nanpercentile(a, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return (a < lo) | (a > hi), (float(lo), float(hi))


def z_outliers(a, thresh=3.0):
    z = np.abs(zscore(a))
    return (z > thresh).ravel() if z.ndim == 1 else (z > thresh), thresh


def histogram(a, bins=10, range=None):
    a = np.asanyarray(a, dtype=float).ravel()
    counts, edges = np.histogram(a[~np.isnan(a)], bins=bins, range=range)
    return counts, edges


def percentile(a, q):
    return np.nanpercentile(np.asanyarray(a, dtype=float), q)


def mode(a):
    vals, counts = np.unique(np.asanyarray(a).ravel(), return_counts=True)
    return vals[int(np.argmax(counts))]


def skew(a):
    a = np.asanyarray(a, dtype=float).ravel()
    a = a[~np.isnan(a)]
    m, s = a.mean(), a.std()
    if s == 0 or a.size < 3:
        return 0.0
    return float((((a - m) / s) ** 3).mean())


def kurtosis(a, fisher=True):
    a = np.asanyarray(a, dtype=float).ravel()
    a = a[~np.isnan(a)]
    m, s = a.mean(), a.std()
    if s == 0 or a.size < 4:
        return 0.0
    k = float((((a - m) / s) ** 4).mean())
    return k - 3.0 if fisher else k


def value_counts(a):
    vals, counts = np.unique(np.asanyarray(a).ravel(), return_counts=True)
    order = np.argsort(-counts)
    return list(zip(vals[order].tolist(), counts[order].tolist()))


def crosstab(a, b):
    a = np.asanyarray(a).ravel()
    b = np.asanyarray(b).ravel()
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)), dtype=int)
    np.add.at(table, (ia, ib), 1)
    return table, ua.tolist(), ub.tolist()
