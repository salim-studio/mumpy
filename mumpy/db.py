"""mumpy.db: unified database layer for analysts & developers.

Zero-dependency core: sqlite3 (stdlib) — full CRUD, bulk insert from numpy,
read to numpy / dict / pandas (optional).
Optional backends (auto-detected): sqlalchemy (postgres/mysql/sqlite),
duckdb (analytical OLAP, parquet direct).

Example:
    import mumpy as cp
    db = cp.db.connect("data.db")
    db.create_table("users", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    db.insert_many("users", [{"name": "salim", "age": 30}])
    arr = db.read_numpy("SELECT age FROM users")
    df  = db.read_pandas("SELECT * FROM users")
"""
from __future__ import annotations

import sqlite3
import numpy as np

__all__ = ["Database", "connect", "has_sqlalchemy", "has_duckdb", "has_pandas"]


def has_sqlalchemy() -> bool:
    try:
        import sqlalchemy  # noqa: F401
        return True
    except Exception:
        return False


def has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


def has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except Exception:
        return False


class Database:
    """Thin, fast wrapper over sqlite3 (+ optional sqlalchemy/duckdb engines)."""

    def __init__(self, path=":memory:", engine="sqlite", echo=False, **kw):
        self.path = path
        self.engine = engine
        self.echo = echo
        self._sa_engine = None
        self._duck = None
        if engine == "sqlite":
            self.conn = sqlite3.connect(path, **{k: v for k, v in kw.items()
                                                 if k in ("timeout", "detect_types",
                                                          "isolation_level", "check_same_thread")})
            self.conn.row_factory = sqlite3.Row
        elif engine == "sqlalchemy":
            if not has_sqlalchemy():
                raise ImportError("pip install sqlalchemy")
            from sqlalchemy import create_engine
            url = kw.pop("url", f"sqlite:///{path}" if path != ":memory:" else "sqlite://")
            self._sa_engine = create_engine(url, echo=echo, **kw)
            self.conn = self._sa_engine.connect()
        elif engine == "duckdb":
            if not has_duckdb():
                raise ImportError("pip install duckdb")
            import duckdb
            self._duck = duckdb.connect(path if path != ":memory:" else ":memory:", **kw)
            self.conn = None
        else:
            raise ValueError(f"unknown engine: {engine}")

    # ---------- core execution ----------
    def execute(self, sql, params=None):
        if self.echo:
            print(sql, params or "")
        if self.engine == "duckdb":
            return self._duck.execute(sql, params or [])
        if self.engine == "sqlalchemy":
            from sqlalchemy import text
            if params:
                return self.conn.execute(text(sql), params)
            return self.conn.execute(text(sql))
        cur = self.conn.cursor()
        cur.execute(sql, params or [])
        self.conn.commit()
        return cur

    def executemany(self, sql, seq):
        if self.engine == "duckdb":
            self._duck.executemany(sql, seq)
            return None
        if self.engine == "sqlalchemy":
            from sqlalchemy import text
            self.conn.execute(text(sql), seq)
            try:
                self.conn.commit()
            except Exception:
                pass
            return None
        cur = self.conn.cursor()
        cur.executemany(sql, seq)
        self.conn.commit()
        return cur

    # ---------- DDL helpers ----------
    def create_table(self, name, schema: dict, if_not_exists=True):
        cols = ", ".join(f'"{k}" {v}' for k, v in schema.items())
        ine = "IF NOT EXISTS " if if_not_exists else ""
        self.execute(f'CREATE TABLE {ine}"{name}" ({cols})')

    def drop_table(self, name, if_exists=True):
        ie = "IF EXISTS " if if_exists else ""
        self.execute(f'DROP TABLE {ie}"{name}"')

    def tables(self):
        if self.engine == "duckdb":
            rows = self._duck.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()
            return [r[0] for r in rows]
        cur = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        try:
            return [r[0] for r in cur.fetchall()]
        except Exception:
            return [r[0] for r in cur]

    # ---------- writes ----------
    def insert_many(self, table, rows):
        """rows: list[dict] | list[tuple] | 2D ndarray (+ columns kw)."""
        if isinstance(rows, np.ndarray):
            raise ValueError("use write_numpy(table, arr, columns=...) for ndarrays")
        if not rows:
            return 0
        if isinstance(rows[0], dict):
            cols = list(rows[0].keys())
            placeholders = ", ".join(["?"] * len(cols))
            sql = f'INSERT INTO "{table}" ({", ".join(chr(34)+c+chr(34) for c in cols)}) VALUES ({placeholders})'
            seq = [tuple(r[c] for c in cols) for r in rows]
        else:
            n = len(rows[0])
            placeholders = ", ".join(["?"] * n)
            sql = f'INSERT INTO "{table}" VALUES ({placeholders})'
            seq = [tuple(r) for r in rows]
        self.executemany(sql, seq)
        return len(seq)

    def write_numpy(self, table, arr, columns=None, dtypes=None):
        """Bulk-write 2D numpy array as a table (creates table if needed)."""
        arr = np.asanyarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        ncols = arr.shape[1]
        if columns is None:
            columns = [f"c{i}" for i in range(ncols)]
        if dtypes is None:
            # infer SQL types from numpy dtype
            def _t(dt):
                dt = np.dtype(dt)
                if np.issubdtype(dt, np.integer):
                    return "INTEGER"
                if np.issubdtype(dt, np.floating):
                    return "REAL"
                return "TEXT"
            if arr.ndim == 2:
                dtypes = [_t(arr.dtype)] * ncols
            else:
                dtypes = ["REAL"] * ncols
        schema = {c: t for c, t in zip(columns, dtypes)}
        self.create_table(table, schema)
        placeholders = ", ".join(["?"] * ncols)
        sql = f'INSERT INTO "{table}" ({", ".join(chr(34)+c+chr(34) for c in columns)}) VALUES ({placeholders})'
        self.executemany(sql, [tuple(r) for r in arr.tolist()])
        return arr.shape[0]

    def from_csv(self, table, csv_path, delimiter=",", header=True):
        from .io import load_csv
        data, names = load_csv(csv_path, delimiter=delimiter, header=header,
                               dtype=str)
        if names is None:
            names = [f"c{i}" for i in range(data.shape[1])] if data.size else []
        # try numeric conversion per column
        cols = {}
        for j, name in enumerate(names):
            col = data[:, j] if data.size else np.array([])
            try:
                cols[name] = col.astype(float)
            except Exception:
                cols[name] = col
        import numpy as _np
        n = data.shape[0] if data.size else 0
        schema = {}
        for name in names:
            v = cols[name]
            try:
                v.astype(float)
                schema[name] = "REAL"
            except Exception:
                schema[name] = "TEXT"
        self.create_table(table, schema)
        if n:
            placeholders = ", ".join(["?"] * len(names))
            sql = f'INSERT INTO "{table}" VALUES ({placeholders})'
            self.executemany(sql, [tuple(data[i, j] for j in range(len(names)))
                                   for i in range(n)])
        return n

    # ---------- reads ----------
    def query(self, sql, params=None):
        """Return list[dict]."""
        if self.engine == "duckdb":
            cur = self._duck.execute(sql, params or [])
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]
        if self.engine == "sqlalchemy":
            from sqlalchemy import text
            res = self.conn.execute(text(sql), params or {})
            cols = list(res.keys())
            return [dict(zip(cols, r)) for r in res.fetchall()]
        cur = self.conn.cursor()
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def read_numpy(self, sql, params=None):
        rows = self.query(sql, params)
        if not rows:
            return np.empty((0, 0))
        keys = list(rows[0].keys())
        try:
            arr = np.array([[r[k] for k in keys] for r in rows], dtype=float)
        except Exception:
            arr = np.array([[r[k] for k in keys] for r in rows], dtype=object)
        try:
            from ._core import _wrap as _w
            return _w(arr)
        except Exception:
            return arr

    def read_pandas(self, sql, params=None):
        if not has_pandas():
            raise ImportError("pip install pandas")
        import pandas as pd
        if self.engine == "duckdb":
            return self._duck.execute(sql).df()
        if self.engine == "sqlalchemy":
            return pd.read_sql(sql, self._sa_engine, params=params)
        return pd.read_sql_query(sql, self.conn, params=params)

    def to_parquet(self, sql, path, params=None):
        rows = self.query(sql, params)
        if not has_pandas():
            raise ImportError("pip install pandas pyarrow")
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    # ---------- maintenance ----------
    def vacuum(self):
        try:
            self.execute("VACUUM")
        except Exception:
            pass

    def close(self):
        try:
            if self.engine == "duckdb":
                self._duck.close()
            elif self.engine == "sqlalchemy":
                self.conn.close()
            else:
                self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def connect(path=":memory:", engine="sqlite", **kw):
    return Database(path, engine=engine, **kw)
