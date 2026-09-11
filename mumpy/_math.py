"""mumpy math: numpy-compatible functions with parallel fast paths + fused ops."""
from __future__ import annotations

import builtins
import numpy as np
from ._parallel import parallel_ewise

__all__ = [
    "add", "subtract", "multiply", "divide", "power", "sqrt", "exp", "log",
    "sin", "cos", "tan", "abs", "absolute", "clip", "where", "fma", "fms",
    "fnma", "lerp", "hypot", "maximum", "minimum",
    "expm1", "log1p", "log2", "log10", "square", "cbrt", "reciprocal",
    "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh",
    "floor", "ceil", "rint", "sign", "sinc",
    "sum", "mean", "min", "max", "amin", "amax", "prod", "std", "var",
    "any", "all", "cumsum", "cumprod", "nansum", "nanmean", "nanmin", "nanmax",
    "nanstd", "dot", "matmul", "einsum", "tensordot",
    "vdot", "inner", "outer", "kron", "trace", "sort", "argsort",
]


def _ew(numpy_func):
    def wrapper(*args, out=None, **kw):
        outs = [np.asanyarray(a).view(np.ndarray) if isinstance(a, np.ndarray) else a
                for a in args]
        try:
            size = builtins.max((np.asanyarray(o).size for o in outs), default=0)
        except Exception:
            size = 0
        from ._parallel import PARALLEL_THRESHOLD
        from ._core import _wrap
        if size >= PARALLEL_THRESHOLD and not kw:
            plains = [np.asanyarray(a).view(np.ndarray) if isinstance(a, np.ndarray) else a
                      for a in args]

            def _f(*views, out=None):
                numpy_func(*views, out=out, **kw)
            try:
                return _wrap(parallel_ewise(_f, *plains, out=out))
            except Exception:
                pass
        if out is None:
            r = numpy_func(*[np.asanyarray(a).view(np.ndarray)
                             if isinstance(a, np.ndarray) else a for a in args], **kw)
        else:
            r = numpy_func(*[np.asanyarray(a).view(np.ndarray)
                             if isinstance(a, np.ndarray) else a for a in args],
                           out=out, **kw)
            return out
        return _wrap(r) if isinstance(r, np.ndarray) else r
    wrapper.__name__ = getattr(numpy_func, "__name__", "ufunc")
    wrapper.__doc__ = getattr(numpy_func, "__doc__", "")
    return wrapper


add = _ew(np.add)
subtract = _ew(np.subtract)
multiply = _ew(np.multiply)
divide = _ew(np.divide)
power = _ew(np.power)
sqrt = _ew(np.sqrt)
exp = _ew(np.exp)
log = _ew(np.log)
sin = _ew(np.sin)
cos = _ew(np.cos)
tan = _ew(np.tan)
absolute = _ew(np.absolute)
abs = absolute
hypot = _ew(np.hypot)
maximum = _ew(np.maximum)
minimum = _ew(np.minimum)
expm1 = _ew(np.expm1)
log1p = _ew(np.log1p)
log2 = _ew(np.log2)
log10 = _ew(np.log10)
square = _ew(np.square)
cbrt = _ew(np.cbrt)
reciprocal = _ew(np.reciprocal)
arcsin = _ew(np.arcsin)
arccos = _ew(np.arccos)
arctan = _ew(np.arctan)
arctan2 = _ew(np.arctan2)
sinh = _ew(np.sinh)
cosh = _ew(np.cosh)
tanh = _ew(np.tanh)
floor = _ew(np.floor)
ceil = _ew(np.ceil)
rint = _ew(np.rint)
sign = _ew(np.sign)
sinc = _ew(np.sinc)


def clip(a, a_min, a_max, out=None):
    a = np.asanyarray(a)
    if a.size >= 50000:
        def _f(v, out=None):
            np.clip(v, a_min, a_max, out=out)
        from ._core import _wrap as _w
        try:
            return _w(parallel_ewise(_f, a, out=out))
        except Exception:
            pass
    return np.clip(a, a_min, a_max, out=out)


def where(condition, x=None, y=None):
    return np.where(np.asanyarray(condition) if not np.isscalar(condition) else condition,
                    x, y)


def _plain(a):
    a = np.asanyarray(a)
    return a.view(np.ndarray) if isinstance(a, np.ndarray) else a


def fma(a, b, c, out=None):
    """Fused multiply-add: a*b + c with ONE allocation (half peak memory)."""
    a, b, c = _plain(a), _plain(b), _plain(c)
    if out is None:
        out = np.multiply(a, b)
        np.add(out, c, out=out)
        from ._core import _wrap as _w
        return _w(out)
    np.multiply(a, b, out=out)
    np.add(out, c, out=out)
    return out


def fms(a, b, c, out=None):
    """Fused multiply-subtract: a*b - c."""
    a, b, c = _plain(a), _plain(b), _plain(c)
    if out is None:
        out = np.multiply(a, b)
        np.subtract(out, c, out=out)
        from ._core import _wrap as _w
        return _w(out)
    np.multiply(a, b, out=out)
    np.subtract(out, c, out=out)
    return out


def fnma(a, b, c, out=None):
    """Fused neg-multiply-add: c - a*b."""
    a, b, c = _plain(a), _plain(b), _plain(c)
    if out is None:
        out = np.multiply(a, b)
        np.subtract(c, out, out=out)
        from ._core import _wrap as _w
        return _w(out)
    np.multiply(a, b, out=out)
    # need temp-free: out = c - out
    np.subtract(c, out, out=out)
    return out


def lerp(a, b, t, out=None):
    """Linear interpolation: a + (b-a)*t  (single-pass fused)."""
    a, b, t = _plain(a), _plain(b), _plain(t)
    if out is None:
        out = np.subtract(b, a)
        np.multiply(out, t, out=out)
        np.add(out, a, out=out)
        from ._core import _wrap as _w
        return _w(out)
    np.subtract(b, a, out=out)
    np.multiply(out, t, out=out)
    np.add(out, a, out=out)
    return out


# ---- reductions: numpy is already bandwidth-optimal single-threaded ----
def sum(a, axis=None, dtype=None, out=None, keepdims=False, **kw):
    return np.sum(_plain(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kw)


def mean(a, axis=None, dtype=None, out=None, keepdims=False, **kw):
    return np.mean(_plain(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kw)


def min(a, axis=None, out=None, keepdims=False, **kw):
    return np.min(_plain(a), axis=axis, out=out, keepdims=keepdims, **kw)


def max(a, axis=None, out=None, keepdims=False, **kw):
    return np.max(_plain(a), axis=axis, out=out, keepdims=keepdims, **kw)


amin, amax = min, max


def prod(a, axis=None, dtype=None, out=None, keepdims=False, **kw):
    return np.prod(_plain(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kw)


def any(a, axis=None, out=None, keepdims=False, **kw):
    return np.any(a, axis=axis, out=out, keepdims=keepdims, **kw)


def all(a, axis=None, out=None, keepdims=False, **kw):
    return np.all(a, axis=axis, out=out, keepdims=keepdims, **kw)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return np.std(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return np.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False, **kw):
    return np.nansum(_plain(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kw)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, **kw):
    return np.nanmean(_plain(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kw)


def nanmin(a, axis=None, out=None, keepdims=False, **kw):
    return np.nanmin(_plain(a), axis=axis, out=out, keepdims=keepdims, **kw)


def nanmax(a, axis=None, out=None, keepdims=False, **kw):
    return np.nanmax(_plain(a), axis=axis, out=out, keepdims=keepdims, **kw)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, **kw):
    return np.nanstd(_plain(a), axis=axis, dtype=dtype, out=out, ddof=ddof,
                     keepdims=keepdims, **kw)


def cumsum(a, axis=None, dtype=None, out=None):
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)


def cumprod(a, axis=None, dtype=None, out=None):
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)


# ---- linear algebra entry points (BLAS-backed, same speed + optimize=True) ----
def dot(a, b, out=None):
    return np.dot(a, b, out=out)


def matmul(a, b, out=None):
    return np.matmul(a, b, out=out)


def einsum(*args, **kwargs):
    kwargs.setdefault("optimize", True)
    return np.einsum(*args, **kwargs)


def tensordot(a, b, axes=2):
    return np.tensordot(a, b, axes=axes)


def vdot(a, b):
    return np.vdot(a, b)


def inner(a, b):
    return np.inner(a, b)


def outer(a, b, out=None):
    return np.outer(a, b, out=out)


def kron(a, b):
    return np.kron(a, b)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    return np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)


def sort(a, axis=-1, kind=None, order=None):
    return np.sort(a, axis=axis, kind=kind, order=order)


def argsort(a, axis=-1, kind=None, order=None):
    return np.argsort(a, axis=axis, kind=kind, order=order)
