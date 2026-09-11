"""mumpy.io: fast loading/saving for data-science workflows.

Supports (stdlib + numpy, no hard deps):
  csv / json / npy / npz / txt / memmap
Optional (used if installed):
  parquet via pandas/pyarrow, excel via pandas/openpyxl

All loaders return mumpy.ndarray (== np.ndarray subclass).
"""
from __future__ import annotations

import csv
import json
import os
import numpy as np

__all__ = [
    "load_csv", "save_csv", "load_json", "save_json",
    "load_npy", "save_npy", "load_npz", "save_npz",
    "load_txt", "save_txt", "memmap", "load_parquet", "save_parquet",
    "load_excel", "save_excel", "read_auto", "save_auto",
]


def _wrap(x):
    try:
        from ._core import _wrap as _w
        return _w(x) if isinstance(x, np.ndarray) else x
    except Exception:
        return x


def load_csv(path, delimiter=",", header=True, dtype=float, skip_rows=0,
             usecols=None, encoding="utf-8"):
    """Load numeric CSV fast. Returns (data, header_names|None)."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if skip_rows:
        rows = rows[skip_rows:]
    names = None
    if header and rows:
        names = rows[0]
        rows = rows[1:]
    if usecols is not None:
        idx = list(usecols) if not isinstance(usecols[0], str) else \
            [names.index(c) for c in usecols]
        rows = [[r[i] for i in idx] for r in rows]
        if names is not None and not isinstance(usecols[0], str):
            names = [names[i] for i in idx]
        elif names is not None:
            names = list(usecols)
    data = np.array(rows, dtype=dtype) if rows else np.empty((0, 0))
    return _wrap(data), names


def save_csv(path, data, header=None, delimiter=",", fmt="%.8g"):
    data = np.asanyarray(data)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        if header:
            w.writerow(list(header))
        if data.ndim == 1:
            for v in data:
                f.write(f"{v}\n")
        elif data.ndim == 2:
            for row in data:
                w.writerow([fmt % v if isinstance(v, (float, np.floating)) else v
                            for v in row])
        else:
            np.savetxt(f, data.reshape(data.shape[0], -1), delimiter=delimiter)
    return path


def load_json(path, key=None):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if key is not None and isinstance(obj, dict):
        obj = obj[key]
    return np.array(obj) if isinstance(obj, list) else obj


def save_json(path, obj, indent=2):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    return path


def load_npy(path, mmap_mode=None):
    return _wrap(np.load(path, mmap_mode=mmap_mode, allow_pickle=True))


def save_npy(path, arr):
    np.save(path, np.asanyarray(arr))
    return path


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: _wrap(z[k]) for k in z.files}


def save_npz(path, compress=True, **arrays):
    if compress:
        np.savez_compressed(path, **{k: np.asanyarray(v) for k, v in arrays.items()})
    else:
        np.savez(path, **{k: np.asanyarray(v) for k, v in arrays.items()})
    return path


def load_txt(path, **kw):
    kw.setdefault("delimiter", None)
    return _wrap(np.loadtxt(path, **kw))


def save_txt(path, arr, **kw):
    np.savetxt(path, np.asanyarray(arr), **kw)
    return path


def memmap(path, dtype=float, mode="r+", shape=None):
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
        return pd
    except ImportError as e:
        raise ImportError("pandas is required for parquet/excel support: pip install pandas pyarrow") from e


def load_parquet(path, columns=None):
    pd = _require_pandas()
    df = pd.read_parquet(path, columns=columns)
    return df


def save_parquet(path, data, columns=None):
    pd = _require_pandas()
    import pandas as _pd
    if isinstance(data, _pd.DataFrame):
        data.to_parquet(path)
    else:
        arr = np.asanyarray(data)
        df = _pd.DataFrame(arr, columns=columns)
        df.to_parquet(path)
    return path


def load_excel(path, sheet=0):
    pd = _require_pandas()
    return pd.read_excel(path, sheet_name=sheet)


def save_excel(path, data, sheet="Sheet1", columns=None):
    pd = _require_pandas()
    import pandas as _pd
    if isinstance(data, _pd.DataFrame):
        data.to_excel(path, sheet_name=sheet, index=False)
    else:
        _pd.DataFrame(np.asanyarray(data), columns=columns).to_excel(
            path, sheet_name=sheet, index=False)
    return path


def read_auto(path, **kw):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return load_csv(path, **kw)
    if ext == ".json":
        return load_json(path, **kw)
    if ext == ".npy":
        return load_npy(path, **kw)
    if ext in (".npz",):
        return load_npz(path, **kw)
    if ext in (".txt", ".tsv", ".dat"):
        return load_txt(path, **kw)
    if ext == ".parquet":
        return load_parquet(path, **kw)
    raise ValueError(f"Unknown extension: {ext}")


def save_auto(path, data, **kw):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return save_csv(path, data, **kw)
    if ext == ".json":
        return save_json(path, data, **kw)
    if ext == ".npy":
        return save_npy(path, data)
    if ext == ".parquet":
        return save_parquet(path, data, **kw)
    return save_txt(path, data, **kw)
