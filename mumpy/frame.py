"""mumpy.frame: lightweight DataFrame for analysts (numpy-backed, pandas-compatible).

No hard dependency on pandas. Columns are numpy arrays; numeric ops are
vectorized + multithreaded via mumpy core.

    import mumpy as cp
    df = cp.frame.DataFrame({"age": [20, 30, 40], "salary": [100, 200, 300]})
    df.head(), df.describe(), df["age"].mean()
    df.filter(df["age"] > 25).sort("salary").groupby("age").mean()
    df.to_pandas() / DataFrame.from_pandas(pdf) / DataFrame.read_csv(...) / read_sql(...)
"""
from __future__ import annotations

import numpy as np

__all__ = ["DataFrame", "Series", "read_csv", "concat"]


def _as_arr(v):
    if isinstance(v, np.ndarray):
        return v
    try:
        return np.array(v)
    except Exception:
        return np.array(list(v), dtype=object)


class Series:
    def __init__(self, data, name=None):
        self.data = _as_arr(data)
        self.name = name

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Series({self.name!r}, {self.data!r})"

    def to_numpy(self):
        return np.asanyarray(self.data)

    # numeric conveniences
    def mean(self, **kw):
        return float(np.nanmean(self.data.astype(float), **kw)) if self.data.size else float("nan")

    def sum(self, **kw):
        return np.nansum(self.data.astype(float), **kw) if self.data.size else 0

    def min(self):
        try:
            return np.nanmin(self.data)
        except Exception:
            return self.data.min()

    def max(self):
        try:
            return np.nanmax(self.data)
        except Exception:
            return self.data.max()

    def std(self, ddof=0):
        return float(np.nanstd(self.data.astype(float), ddof=ddof))

    def unique(self):
        return np.unique(self.data)

    def value_counts(self):
        vals, counts = np.unique(self.data, return_counts=True)
        order = np.argsort(-counts)
        return list(zip(vals[order].tolist(), counts[order].tolist()))

    # operators -> numpy arrays
    def _binop(self, other, op):
        o = other.data if isinstance(other, Series) else other
        return op(np.asanyarray(self.data), np.asanyarray(o) if not np.isscalar(o) else o)

    def __gt__(self, o): return self._binop(o, np.greater)
    def __ge__(self, o): return self._binop(o, np.greater_equal)
    def __lt__(self, o): return self._binop(o, np.less)
    def __le__(self, o): return self._binop(o, np.less_equal)
    def __eq__(self, o): return self._binop(o, np.equal)
    def __ne__(self, o): return self._binop(o, np.not_equal)
    def __add__(self, o): return Series(self._binop(o, np.add), self.name)
    def __sub__(self, o): return Series(self._binop(o, np.subtract), self.name)
    def __mul__(self, o): return Series(self._binop(o, np.multiply), self.name)
    def __truediv__(self, o): return Series(self._binop(o, np.true_divide), self.name)


class _GroupBy:
    def __init__(self, df, keys):
        self.df = df
        self.keys = [keys] if isinstance(keys, str) else list(keys)

    def _groups(self):
        key_cols = [np.asanyarray(self.df._cols[k]) for k in self.keys]
        combo = np.array(["|".join(map(str, t)) for t in zip(*[c.tolist() for c in key_cols])])
        uniq, inv = np.unique(combo, return_inverse=True)
        return uniq, inv

    def agg(self, func="mean"):
        uniq, inv = self._groups()
        out = {k: [] for k in self.keys}
        num_cols = [c for c in self.df.columns if c not in self.keys]
        agg_cols = {c: [] for c in num_cols}
        for g in range(len(uniq)):
            mask = inv == g
            parts = uniq[g].split("|")
            for k, p in zip(self.keys, parts):
                out[k].append(p)
            for c in num_cols:
                col = np.asanyarray(self.df._cols[c])
                try:
                    vals = col[mask].astype(float)
                    if func == "mean":
                        agg_cols[c].append(float(np.nanmean(vals)))
                    elif func == "sum":
                        agg_cols[c].append(float(np.nansum(vals)))
                    elif func == "min":
                        agg_cols[c].append(float(np.nanmin(vals)))
                    elif func == "max":
                        agg_cols[c].append(float(np.nanmax(vals)))
                    elif func == "count":
                        agg_cols[c].append(int(mask.sum()))
                    else:
                        agg_cols[c].append(float(func(vals)))
                except Exception:
                    agg_cols[c].append(None)
        merged = {**out, **agg_cols}
        return DataFrame(merged)

    def mean(self): return self.agg("mean")
    def sum(self): return self.agg("sum")
    def count(self): return self.agg("count")
    def min(self): return self.agg("min")
    def max(self): return self.agg("max")


class DataFrame:
    """Minimal pandas-like table backed by numpy columns."""

    def __init__(self, data=None, columns=None):
        self._cols: dict[str, np.ndarray] = {}
        self.columns: list[str] = []
        if data is None:
            return
        if isinstance(data, dict):
            for k, v in data.items():
                self._cols[str(k)] = _as_arr(v)
                self.columns.append(str(k))
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            for k in keys:
                self._cols[str(k)] = _as_arr([r[k] for r in data])
            self.columns = [str(k) for k in keys]
        elif isinstance(data, np.ndarray):
            arr = data
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            cols = columns or [f"c{i}" for i in range(arr.shape[1])]
            for i, c in enumerate(cols):
                self._cols[str(c)] = _as_arr(arr[:, i])
            self.columns = [str(c) for c in cols]
        else:
            raise ValueError("Unsupported data type for DataFrame")

    # ---------- basics ----------
    def __len__(self):
        return len(next(iter(self._cols.values()))) if self._cols else 0

    @property
    def shape(self):
        return (len(self), len(self.columns))

    def __repr__(self):
        return f"DataFrame(shape={self.shape}, columns={self.columns})\n{self.head(5).to_dict()}"

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._cols[key], name=key)
        if isinstance(key, (list, tuple)):
            return self.select(list(key))
        if isinstance(key, (np.ndarray, list)):
            mask = np.asanyarray(key)
            if mask.dtype == bool:
                return self.filter(mask)
            return self.iloc(mask.tolist())
        raise KeyError(key)

    def __setitem__(self, key, values):
        self._cols[str(key)] = _as_arr(values.data if isinstance(values, Series) else values)
        if str(key) not in self.columns:
            self.columns.append(str(key))

    def head(self, n=5):
        return self.iloc(list(range(min(n, len(self)))))

    def tail(self, n=5):
        return self.iloc(list(range(max(0, len(self) - n), len(self))))

    def iloc(self, idx):
        idx = np.asanyarray(idx)
        return DataFrame({c: np.asanyarray(v)[idx] for c, v in self._cols.items()})

    def select(self, cols):
        return DataFrame({c: self._cols[c] for c in cols})

    def drop(self, cols):
        cols = [cols] if isinstance(cols, str) else list(cols)
        keep = [c for c in self.columns if c not in cols]
        return self.select(keep)

    def filter(self, mask):
        mask = np.asanyarray(mask, dtype=bool)
        return DataFrame({c: np.asanyarray(v)[mask] for c, v in self._cols.items()})

    def query(self, expr):
        """Simple query like 'age > 30'. Supports single comparison."""
        import re
        m = re.match(r"\s*(\w+)\s*(>=|<=|>|<|==|!=)\s*(.+)\s*", expr)
        if not m:
            raise ValueError(f"Unsupported query: {expr}")
        col, op, val = m.groups()
        try:
            val = float(val) if "." in val else int(val)
        except Exception:
            val = val.strip("'\"")
        s = self[col].data
        ops = {">": np.greater, ">=": np.greater_equal, "<": np.less,
               "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
        return self.filter(ops[op](np.asanyarray(s), val))

    def sort(self, by, ascending=True):
        by = [by] if isinstance(by, str) else list(by)
        keys = [np.asanyarray(self._cols[c]) for c in by]
        try:
            order = np.lexsort(keys[::-1])
        except Exception:
            order = np.argsort(keys[0].astype(str))
        if not ascending:
            order = order[::-1]
        return self.iloc(order)

    def groupby(self, keys):
        return _GroupBy(self, keys)

    def fillna(self, value=0.0):
        out = {}
        for c, v in self._cols.items():
            a = np.asanyarray(v)
            if np.issubdtype(a.dtype, np.number):
                a = a.astype(float, copy=True)
                a[np.isnan(a)] = value
            out[c] = a
        return DataFrame(out)

    def dropna(self):
        mask = np.ones(len(self), bool)
        for v in self._cols.values():
            a = np.asanyarray(v)
            if np.issubdtype(a.dtype, np.number):
                mask &= ~np.isnan(a.astype(float))
        return self.filter(mask)

    def describe(self):
        rows = {}
        for c, v in self._cols.items():
            a = np.asanyarray(v)
            if np.issubdtype(a.dtype, np.number):
                af = a.astype(float)
                rows[c] = {
                    "count": int((~np.isnan(af)).sum()),
                    "mean": float(np.nanmean(af)),
                    "std": float(np.nanstd(af)),
                    "min": float(np.nanmin(af)),
                    "25%": float(np.nanpercentile(af, 25)),
                    "50%": float(np.nanpercentile(af, 50)),
                    "75%": float(np.nanpercentile(af, 75)),
                    "max": float(np.nanmax(af)),
                }
            else:
                vals, counts = np.unique(a, return_counts=True)
                rows[c] = {"count": len(a), "unique": len(vals),
                           "top": str(vals[np.argmax(counts)])}
        return rows

    def corr(self):
        num = [c for c in self.columns
               if np.issubdtype(np.asanyarray(self._cols[c]).dtype, np.number)]
        if not num:
            return np.empty((0, 0))
        m = np.column_stack([np.asanyarray(self._cols[c]).astype(float) for c in num])
        # nan -> column mean
        col_mean = np.nanmean(m, axis=0)
        idx = np.where(np.isnan(m))
        m[idx] = np.take(col_mean, idx[1])
        return np.corrcoef(m, rowvar=False)

    def merge(self, other, on, how="inner"):
        on = [on] if isinstance(on, str) else list(on)
        lkey = np.array(["|".join(map(str, t)) for t in zip(
            *[np.asanyarray(self._cols[k]).tolist() for k in on])])
        rkey = np.array(["|".join(map(str, t)) for t in zip(
            *[np.asanyarray(other._cols[k]).tolist() for k in on])])
        r_index = {}
        for i, k in enumerate(rkey.tolist()):
            r_index.setdefault(k, []).append(i)
        l_idx, r_idx = [], []
        if how == "inner":
            for i, k in enumerate(lkey.tolist()):
                for j in r_index.get(k, []):
                    l_idx.append(i)
                    r_idx.append(j)
        elif how == "left":
            for i, k in enumerate(lkey.tolist()):
                js = r_index.get(k, [None])
                for j in js:
                    l_idx.append(i)
                    r_idx.append(j)
        else:
            raise ValueError("only inner/left supported in lite merge")
        out = {}
        for c in self.columns:
            out[c] = np.asanyarray(self._cols[c])[l_idx]
        for c in other.columns:
            if c in on:
                continue
            rv = np.asanyarray(other._cols[c])
            col = []
            for j in r_idx:
                col.append(rv[j] if j is not None else None)
            out[c] = np.array(col)
        return DataFrame(out)

    # ---------- conversions ----------
    def to_numpy(self, cols=None):
        cols = cols or self.columns
        return np.column_stack([np.asanyarray(self._cols[c]) for c in cols])

    def to_dict(self, orient="list"):
        if orient == "list":
            return {c: np.asanyarray(v).tolist() for c, v in self._cols.items()}
        return [{c: np.asanyarray(v)[i].tolist() if hasattr(np.asanyarray(v)[i], "tolist") else np.asanyarray(v)[i]
                 for c, v in self._cols.items()} for i in range(len(self))]

    def to_csv(self, path, **kw):
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(self.columns)
            for i in range(len(self)):
                w.writerow([np.asanyarray(self._cols[c])[i] for c in self.columns])
        return path

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pip install pandas") from e
        return pd.DataFrame({c: np.asanyarray(v) for c, v in self._cols.items()})

    @classmethod
    def from_pandas(cls, pdf):
        return cls({c: np.asanyarray(pdf[c].values) for c in pdf.columns})

    @classmethod
    def from_dict(cls, d):
        return cls(d)

    @classmethod
    def from_numpy(cls, arr, columns=None):
        return cls(arr, columns=columns)

    @classmethod
    def read_csv(cls, path, delimiter=",", header=True, **kw):
        from .io import load_csv
        data, names = load_csv(path, delimiter=delimiter, header=header, dtype=str)
        if data.size == 0:
            return cls({})
        if names is None:
            names = [f"c{i}" for i in range(data.shape[1])]
        out = {}
        for j, name in enumerate(names):
            col = data[:, j]
            for conv in (int, float):
                try:
                    # vectorized attempt
                    test = col.astype(float)
                    if conv is int and np.all(test == test.astype(int)):
                        col = test.astype(int)
                    else:
                        col = test
                    break
                except Exception:
                    continue
            out[name] = col
        return cls(out)

    @classmethod
    def read_sql(cls, sql, db_or_path, **kw):
        from .db import Database
        if isinstance(db_or_path, Database):
            rows = db_or_path.query(sql)
        else:
            with Database(db_or_path) as db:
                rows = db.query(sql)
        if not rows:
            return cls({})
        cols: dict[str, list] = {k: [] for k in rows[0].keys()}
        for r in rows:
            for k, v in r.items():
                cols[k].append(v)
        return cls({k: _as_arr(v) for k, v in cols.items()})


def read_csv(path, **kw):
    return DataFrame.read_csv(path, **kw)


def concat(dfs, axis=0):
    if axis == 0:
        cols = dfs[0].columns
        return DataFrame({c: np.concatenate([np.asanyarray(d._cols[c]) for d in dfs])
                          for c in cols})
    # axis=1
    out = {}
    for d in dfs:
        out.update(d._cols)
    return DataFrame(out)
