"""mumpy.viz: one-line plots for quick EDA (requires matplotlib, optional)."""
from __future__ import annotations

import numpy as np

__all__ = ["hist", "scatter", "line", "heatmap", "corr_heatmap", "boxplot"]


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("pip install matplotlib") from e


def hist(a, bins=30, title=None, show=False):
    plt = _plt()
    plt.figure()
    plt.hist(np.asanyarray(a, dtype=float).ravel(), bins=bins)
    if title:
        plt.title(title)
    if show:
        plt.show()
    return plt.gca()


def scatter(x, y, title=None, show=False):
    plt = _plt()
    plt.figure()
    plt.scatter(np.asanyarray(x).ravel(), np.asanyarray(y).ravel(), alpha=0.6)
    if title:
        plt.title(title)
    if show:
        plt.show()
    return plt.gca()


def line(*series, title=None, show=False):
    plt = _plt()
    plt.figure()
    for s in series:
        plt.plot(np.asanyarray(s).ravel())
    if title:
        plt.title(title)
    if show:
        plt.show()
    return plt.gca()


def heatmap(m, title=None, show=False):
    plt = _plt()
    plt.figure()
    plt.imshow(np.asanyarray(m, dtype=float), aspect="auto")
    plt.colorbar()
    if title:
        plt.title(title)
    if show:
        plt.show()
    return plt.gca()


def corr_heatmap(X, labels=None, show=False):
    X = np.asanyarray(X, dtype=float)
    C = np.corrcoef(X, rowvar=False)
    ax = heatmap(C, title="Correlation", show=False)
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels, rotation=45)
        ax.set_yticks(range(len(labels)), labels)
    if show:
        _plt().show()
    return ax


def boxplot(*cols, labels=None, show=False):
    plt = _plt()
    plt.figure()
    plt.boxplot([np.asanyarray(c, dtype=float).ravel() for c in cols], labels=labels)
    if show:
        plt.show()
    return plt.gca()
