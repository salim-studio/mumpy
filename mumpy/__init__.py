"""mumpy — drop-in numpy-compatible library + data-science / ML / DL toolkit.

Same API as numpy for the common 95%:
    import mumpy as cp
    a = cp.arange(10_000_000)
    b = cp.sqrt(a)          # multithreaded for large arrays
    c = cp.fma(a, b, 1.0)   # fused a*b+c, single pass

Beyond numpy — one import for the whole workflow:
    cp.io / cp.db / cp.frame      # IO, databases (sqlite/postgres/duckdb), DataFrame
    cp.stats / cp.preprocessing   # EDA, scalers, encoders, imputers
    cp.metrics / cp.ml            # metrics + classic ML (sklearn-like API)
    cp.nn / cp.viz / cp.utils     # tiny deep learning + one-line plots

Speed sources (no C compiler needed):
1. chunked ThreadPool for compute-bound element-wise ufuncs,
2. fused multiply-add fma/fms/fnma/lerp with half peak memory,
3. scipy.fft with workers=-1 when available,
4. einsum(optimize=True), contiguous-layout matmul.
"""
from __future__ import annotations

import numpy as _np

from ._core import (
    ndarray, array, asarray, asanyarray, ascontiguousarray, asnumpy, copy,
    zeros, ones, empty, full, zeros_like, ones_like, empty_like, full_like,
    arange, linspace, logspace, eye, identity, diag,
    reshape, ravel, transpose, concatenate, stack, vstack, hstack,
    split, tile, repeat,
)
from ._math import (
    add, subtract, multiply, divide, power, sqrt, exp, log,
    sin, cos, tan, abs, absolute, clip, where, fma, fms, fnma, lerp,
    hypot, maximum, minimum, expm1, log1p, log2, log10, square, cbrt,
    reciprocal, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh,
    floor, ceil, rint, sign, sinc,
    sum, mean, min, max, amin, amax, prod, std, var,
    nansum, nanmean, nanmin, nanmax, nanstd,
    any, all, cumsum, cumprod, dot, matmul, einsum, tensordot,
    vdot, inner, outer, kron, trace, sort, argsort,
)
from . import linalg, fft, random, io, db, frame, stats, preprocessing, metrics, ml, nn, utils
try:
    from . import viz  # optional matplotlib
except Exception:
    viz = None  # type: ignore
from ._parallel import MAX_WORKERS, PARALLEL_THRESHOLD, get_workers

__version__ = "0.2.0"

# ---- numpy compat: re-export everything else verbatim ----
_COMPAT = [
    "pi", "e", "inf", "nan", "newaxis",
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
    "ndim", "shape", "size", "dtype", "result_type", "broadcast_arrays",
    "broadcast_to", "expand_dims", "squeeze", "flatten", "moveaxis",
    "swapaxes", "flip", "roll", "rot90", "pad", "meshgrid",
    "unique", "intersect1d", "union1d", "setdiff1d", "isin", "in1d",
    "take", "put", "compress", "extract", "argmax", "argmin", "nonzero",
    "count_nonzero", "flatnonzero", "argwhere", "searchsorted", "digitize",
    "histogram", "bincount", "corrcoef", "cov", "polyfit", "polyval",
    "convolve", "gradient", "diff", "ediff1d", "interp", "percentile",
    "quantile", "median", "average", "ptp", "allclose", "isclose",
    "array_equal", "array_equiv", "isnan", "isinf", "isfinite",
    "nan_to_num", "nansum", "nanmean", "nanmin", "nanmax",
    "logical_and", "logical_or", "logical_not", "logical_xor",
    "greater", "greater_equal", "less", "less_equal", "equal", "not_equal",
    "fmax", "fmin",
    "arctan2",
    "round", "fix", "trunc",
    "real", "imag", "conj", "conjugate", "angle",
    "degrees", "radians", "mod", "remainder", "divmod", "fmod",
    "bitwise_and", "bitwise_or", "bitwise_xor", "invert", "left_shift", "right_shift",
    "vecdot",
]
for _name in _COMPAT:
    if hasattr(_np, _name) and _name not in globals():
        globals()[_name] = getattr(_np, _name)

__all__ = [
    "ndarray", "array", "asarray", "asanyarray", "ascontiguousarray", "asnumpy", "copy",
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "logspace", "eye", "identity", "diag",
    "reshape", "ravel", "transpose", "concatenate", "stack", "vstack", "hstack",
    "split", "tile", "repeat",
    "add", "subtract", "multiply", "divide", "power", "sqrt", "exp", "log",
    "sin", "cos", "tan", "abs", "absolute", "clip", "where", "fma", "fms", "fnma", "lerp",
    "hypot", "maximum", "minimum", "expm1", "log1p", "log2", "log10", "square", "cbrt",
    "reciprocal", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh",
    "floor", "ceil", "rint", "sign", "sinc",
    "sum", "mean", "min", "max", "amin", "amax", "prod", "std", "var",
    "nansum", "nanmean", "nanmin", "nanmax", "nanstd",
    "any", "all", "cumsum", "cumprod", "dot", "matmul", "einsum", "tensordot",
    "vdot", "inner", "outer", "kron", "trace", "sort", "argsort",
    "linalg", "fft", "random", "io", "db", "frame", "stats", "preprocessing",
    "metrics", "ml", "nn", "utils", "viz",
    "MAX_WORKERS", "PARALLEL_THRESHOLD",
    "__version__",
]


def info():
    import os
    return {
        "version": __version__,
        "numpy": _np.__version__,
        "workers": MAX_WORKERS,
        "cpus": os.cpu_count(),
        "parallel_threshold": PARALLEL_THRESHOLD,
        "scipy_fft": fft.has_scipy(),
        "modules": ["io", "db", "frame", "stats", "preprocessing",
                    "metrics", "ml", "nn", "utils", "viz"],
    }


def set_workers(n: int):
    """Tune parallelism. set_workers(1) == pure-numpy behavior."""
    from . import _parallel
    _parallel.MAX_WORKERS = max(1, int(n))
    globals()["MAX_WORKERS"] = _parallel.MAX_WORKERS


def set_threshold(n: int):
    from . import _parallel
    _parallel.PARALLEL_THRESHOLD = int(n)
    globals()["PARALLEL_THRESHOLD"] = int(n)
