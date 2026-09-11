"""mumpy core: numpy-compatible ndarray + creation routines."""
from __future__ import annotations

import builtins
import numpy as np
builtins_max = builtins.max

__all__ = [
    "ndarray", "array", "asarray", "asanyarray", "ascontiguousarray",
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like",
    "full_like", "arange", "linspace", "logspace", "eye", "identity", "diag",
    "reshape", "ravel", "transpose", "concatenate", "stack", "vstack", "hstack",
    "split", "tile", "repeat", "copy", "asnumpy",
]


class ndarray(np.ndarray):
    """mumpy array: subclasses np.ndarray so every numpy API accepts it.

    Adds convenience fused/fast methods that avoid temporaries.
    """

    def __new__(cls, *args, **kwargs):
        obj = np.asarray(*args, **kwargs).view(cls)
        return obj

    def __array_finalize__(self, obj):
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Route large element-wise ops through the parallel engine so that
        # BOTH `cp.sqrt(a)` and plain `a / N`, `a * b + c` syntax are faster.
        import numpy as _np
        try:
            from ._parallel import parallel_ewise, PARALLEL_THRESHOLD
            _PARALLEL_UFUNCS = frozenset({
                "add", "subtract", "multiply", "divide", "true_divide",
                "power", "sqrt", "exp", "log", "sin", "cos", "tan",
                "absolute", "maximum", "minimum",
            })
            if method == "__call__" and ufunc.__name__ in _PARALLEL_UFUNCS:
                if kwargs.get("where", None) is None and kwargs.get("casting", None) is None:
                    arrs = [x for x in inputs if isinstance(x, _np.ndarray)]
                    size = builtins_max([x.size for x in arrs], default=0) if arrs else 0
                    if size >= PARALLEL_THRESHOLD:
                        out = kwargs.pop("out", None)
                        o = out[0] if isinstance(out, tuple) and len(out) == 1 else out

                        def _f(*views, out=None):
                            ufunc(*views, out=out)
                        r = parallel_ewise(_f, *inputs, out=o)
                        if o is not None:
                            return o
                        return r.view(ndarray) if isinstance(r, _np.ndarray) else r
        except Exception:
            pass
        # default path: unwrap to plain ndarray and call the ufunc
        # (calling super().__array_ufunc__ returns NotImplemented on numpy>=2)
        def _plain(x):
            if isinstance(x, ndarray):
                return x.view(_np.ndarray)
            if isinstance(x, _np.ndarray):
                return x.view(_np.ndarray)
            return x
        inputs = tuple(_plain(x) for x in inputs)
        if "out" in kwargs and kwargs["out"] is not None:
            out = kwargs["out"]
            kwargs["out"] = tuple(_plain(o) if o is not None else None for o in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return _wrap(result) if isinstance(result, _np.ndarray) else result

    # -- fast methods (avoid extra temporaries) --
    def fma(self, b, c):
        """self * b + c in one pass (no intermediate)."""
        from ._math import fma
        return fma(self, b, c)

    def squared(self):
        return np.multiply(self, self)

    def asum(self):
        return np.abs(np.asanyarray(self)).sum()

    def to_numpy(self):
        return np.asanyarray(self).view(np.ndarray)


def _wrap(x):
    if isinstance(x, np.ndarray) and not isinstance(x, ndarray):
        return x.view(ndarray)
    return x


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0, **kw):
    return _wrap(np.array(obj, dtype=dtype, copy=copy, order=order,
                          subok=subok, ndmin=ndmin, **kw))


def asarray(a, dtype=None, order=None):
    return _wrap(np.asarray(a, dtype=dtype, order=order))


def asanyarray(a, dtype=None, order=None):
    return _wrap(np.asanyarray(a, dtype=dtype, order=order))


def ascontiguousarray(a, dtype=None):
    return _wrap(np.ascontiguousarray(a, dtype=dtype))


def asnumpy(a):
    """Unwrap to plain np.ndarray (zero-copy view)."""
    return np.asanyarray(a).view(np.ndarray)


def copy(a, order="K"):
    return _wrap(np.array(a, copy=True, order=order, subok=True))


def zeros(shape, dtype=float, order="C"):
    return _wrap(np.zeros(shape, dtype=dtype, order=order))


def ones(shape, dtype=float, order="C"):
    return _wrap(np.ones(shape, dtype=dtype, order=order))


def empty(shape, dtype=float, order="C"):
    return _wrap(np.empty(shape, dtype=dtype, order=order))


def full(shape, fill_value, dtype=None, order="C"):
    return _wrap(np.full(shape, fill_value, dtype=dtype, order=order))


def zeros_like(a, dtype=None, order="K", subok=False, shape=None):
    return _wrap(np.zeros_like(a, dtype=dtype, order=order, subok=True, shape=shape))


def ones_like(a, dtype=None, order="K", subok=False, shape=None):
    return _wrap(np.ones_like(a, dtype=dtype, order=order, subok=True, shape=shape))


def empty_like(a, dtype=None, order="K", subok=False, shape=None):
    return _wrap(np.empty_like(a, dtype=dtype, order=order, subok=True, shape=shape))


def full_like(a, fill_value, dtype=None, order="K", subok=False, shape=None):
    return _wrap(np.full_like(a, fill_value, dtype=dtype, order=order, subok=True, shape=shape))


def arange(*args, **kwargs):
    return _wrap(np.arange(*args, **kwargs))


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    r = np.linspace(start, stop, num=num, endpoint=endpoint,
                    retstep=retstep, dtype=dtype, axis=axis)
    if retstep:
        return _wrap(r[0]), r[1]
    return _wrap(r)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    return _wrap(np.logspace(start, stop, num=num, endpoint=endpoint,
                             base=base, dtype=dtype, axis=axis))


def eye(N, M=None, k=0, dtype=float, order="C"):
    return _wrap(np.eye(N, M=M, k=k, dtype=dtype, order=order))


def identity(n, dtype=float):
    return _wrap(np.identity(n, dtype=dtype))


def diag(v, k=0):
    return _wrap(np.diag(v, k=k))


def reshape(a, shape, order="C"):
    return _wrap(np.reshape(a, shape, order=order))


def ravel(a, order="C"):
    return _wrap(np.ravel(a, order=order))


def transpose(a, axes=None):
    return _wrap(np.transpose(a, axes=axes))


def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    return _wrap(np.concatenate(arrays, axis=axis, out=out, dtype=dtype, casting=casting))


def stack(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    return _wrap(np.stack(arrays, axis=axis, out=out, dtype=dtype, casting=casting))


def vstack(tup, dtype=None, casting="same_kind"):
    return _wrap(np.vstack(tup, dtype=dtype, casting=casting))


def hstack(tup, dtype=None, casting="same_kind"):
    return _wrap(np.hstack(tup, dtype=dtype, casting=casting))


def split(ary, indices_or_sections, axis=0):
    return [_wrap(x) for x in np.split(ary, indices_or_sections, axis=axis)]


def tile(A, reps):
    return _wrap(np.tile(A, reps))


def repeat(a, repeats, axis=None):
    return _wrap(np.repeat(a, repeats, axis=axis))
