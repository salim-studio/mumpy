"""mumpy parallel engine: chunked multithreaded execution for memory-bound ops.

Why faster than plain numpy for many ops:
- numpy element-wise ufuncs are single-threaded. We split large arrays
  into chunks and run the ufunc in a ThreadPool (GIL is released in C loops).
- reductions are done per-chunk in parallel then combined.
- fft uses scipy.fft with all workers when available.
"""
from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_ncpu = os.cpu_count() or 4
MAX_WORKERS = max(1, min(32, _ncpu))

# Only parallelize above this many elements (avoid thread overhead).
PARALLEL_THRESHOLD = 50_000


def get_workers(n_elements: int | None = None) -> int:
    if n_elements is not None and n_elements < PARALLEL_THRESHOLD:
        return 1
    return MAX_WORKERS


def _split_first_axis(n: int, workers: int) -> list[tuple[int, int]]:
    chunk = (n + workers - 1) // workers
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def parallel_ewise(func, *arrays, out=None):
    """Apply element-wise C func over chunks in parallel.

    func: callable (*views, out_view) executed on numpy views.
    arrays: broadcastable arrays (broadcasted first, then chunked on flat view).
    """
    # strip subclasses: work on plain ndarrays to avoid __array_ufunc__ recursion
    arrays = [np.asanyarray(a).view(np.ndarray) for a in arrays]
    if out is not None:
        out = np.asanyarray(out).view(np.ndarray)
    # fast path: all same shape & contiguous -> zero-copy ravel views
    try:
        shape0 = arrays[0].shape
        if all(a.shape == shape0 for a in arrays):
            size = arrays[0].size
            workers = get_workers(size)
            if workers <= 1 or size < PARALLEL_THRESHOLD:
                if out is None:
                    res = np.empty(shape0, dtype=np.result_type(*arrays))
                    func(*arrays, out=res)
                    return res
                func(*arrays, out=out)
                return out
            flats = [a.reshape(-1) if a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]
                     else np.ascontiguousarray(a).reshape(-1) for a in arrays]
            if out is None:
                res_flat = np.empty(size, dtype=np.result_type(*arrays))
            else:
                res_flat = np.asanyarray(out).reshape(-1)
            ranges = _split_first_axis(size, workers)

            def _job(r):
                s, e = r
                views = [f[s:e] for f in flats]
                func(*views, out=res_flat[s:e])

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_job, ranges))
            if out is None:
                return res_flat.reshape(shape0)
            return out
    except Exception:
        pass
    try:
        bcast = np.broadcast_arrays(*arrays)
    except ValueError:
        # fall back to plain call (e.g. matmul-like)
        return func(*arrays, out=out) if out is not None else func(*arrays)
    shape = bcast[0].shape
    size = bcast[0].size
    workers = get_workers(size)
    if workers <= 1 or size < PARALLEL_THRESHOLD:
        outs = [np.ascontiguousarray(b) for b in bcast]
        if out is None:
            res = np.empty(shape, dtype=np.result_type(*arrays))
            func(*outs, out=res.reshape(-1) if res.ndim else res)
            return res
        func(*outs, out=np.asanyarray(out).reshape(-1))
        return out

    # flatten broadcasted (copies only if needed for non-contiguous)
    flats = [np.ascontiguousarray(b).reshape(-1) for b in bcast]
    if out is None:
        res_flat = np.empty(size, dtype=np.result_type(*arrays))
    else:
        res_flat = np.asanyarray(out).reshape(-1)
    ranges = _split_first_axis(size, workers)

    def _job(r):
        s, e = r
        views = [f[s:e] for f in flats]
        func(*views, out=res_flat[s:e])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_job, ranges))
    if out is None:
        return res_flat.reshape(shape)
    return out


def parallel_reduce(op, a, axis=None, dtype=None, keepdims=False, op_name="sum"):
    """Parallel reduction over the flattened array when axis is None,
    else chunked along the given axis."""
    a = np.asanyarray(a)
    workers = get_workers(a.size)
    _no_dtype = op_name in ("min", "max", "any", "all")
    if workers <= 1 or a.size < PARALLEL_THRESHOLD or axis is not None:
        if _no_dtype:
            return op(a, axis=axis, keepdims=keepdims)
        return op(a, axis=axis, dtype=dtype, keepdims=keepdims)
    # flat parallel reduce: partials then combine
    flats = np.ascontiguousarray(a).reshape(-1)
    ranges = _split_first_axis(flats.size, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        if _no_dtype:
            partials = list(ex.map(lambda r: op(flats[r[0]:r[1]]), ranges))
        else:
            partials = list(ex.map(lambda r: op(flats[r[0]:r[1]], dtype=dtype), ranges))
    partials = np.asanyarray(partials)
    if op_name in ("min", "max", "any", "all"):
        res = op(partials)
    elif op_name in ("sum", "prod"):
        res = op(partials, dtype=dtype)
    elif op_name == "mean":
        res = np.mean(partials, dtype=dtype) if False else np.sum(partials, dtype=dtype) / flats.size
    else:
        res = op(partials)
    if keepdims:
        res = np.asanyarray(res).reshape((1,) * a.ndim)
    return res
