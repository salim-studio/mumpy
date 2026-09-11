"""mumpy utils: seeds, timing, chunking, memory helpers, missing-value utils."""
from __future__ import annotations

import time
import numpy as np
from contextlib import contextmanager

__all__ = [
    "seed", "timer", "timeit", "chunks", "asnumpy", "is_mumpy",
    "memory_usage", "count_nan", "drop_nan", "fill_nan",
    "normalize", "standardize", "one_hot",
]


def seed(s=None):
    """Set global seeds for reproducibility (numpy + mumpy.random + python random)."""
    import random as _pr
    np.random.seed(s if s is None else int(s) % (2**32 - 1))
    try:
        from . import random as _cr
        _cr.seed(s)
    except Exception:
        pass
    _pr.seed(s)


@contextmanager
def timer(name="elapsed"):
    t0 = time.perf_counter()
    yield
    print(f"{name}: {(time.perf_counter() - t0) * 1000:.2f}ms")


def timeit(func, *args, repeat=5, warmup=1, **kw):
    for _ in range(warmup):
        func(*args, **kw)
    best = min(
        (lambda: (t0 := time.perf_counter(), func(*args, **kw),
                  time.perf_counter() - t0)[2])()
        for _ in range(repeat)
    )
    return best


def chunks(n, size):
    for i in range(0, n, size):
        yield (i, min(i + size, n))


def asnumpy(a):
    return np.asanyarray(a).view(np.ndarray)


def is_mumpy(a):
    from ._core import ndarray
    return isinstance(a, ndarray)


# backward compat with the old `cumpy` name
is_cumpy = is_mumpy


def memory_usage(a):
    a = np.asanyarray(a)
    return int(a.nbytes)


def count_nan(a):
    return int(np.isnan(np.asanyarray(a, dtype=float)).sum())


def drop_nan(a):
    a = np.asanyarray(a)
    mask = ~np.isnan(a.astype(float)) if np.issubdtype(a.dtype, np.number) else np.zeros(len(a), bool)
    return a[mask] if a.ndim == 1 else a


def fill_nan(a, value=0.0):
    a = np.asanyarray(a).astype(float, copy=True)
    a[np.isnan(a)] = value
    return a


def normalize(a, axis=None):
    a = np.asanyarray(a, dtype=float)
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def standardize(a, axis=0):
    a = np.asanyarray(a, dtype=float)
    m = a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True)
    s[s == 0] = 1.0
    return (a - m) / s


def one_hot(labels, num_classes=None):
    labels = np.asanyarray(labels).astype(int).ravel()
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    out = np.zeros((labels.size, num_classes), dtype=float)
    out[np.arange(labels.size), labels] = 1.0
    return out
