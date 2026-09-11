"""Stable benchmark: mumpy vs numpy (best-of-N, warmup, large arrays)."""
import time
import numpy as np
import mumpy as cp

N = 10_000_000
a_np = np.arange(N, dtype=np.float64)
a_cp = cp.arange(N, dtype=np.float64)

print("mumpy info:", cp.info())


def bench(name, f_np, f_cp, warmup=1, iters=5):
    for _ in range(warmup):
        f_np()
        f_cp()
    best_np = min((lambda f: (s := time.perf_counter(), f(), time.perf_counter() - s)[2])(f_np)
                  for _ in range(iters))
    best_cp = min((lambda f: (s := time.perf_counter(), f(), time.perf_counter() - s)[2])(f_cp)
                  for _ in range(iters))
    print(f"{name:12s} numpy={best_np*1000:8.1f}ms  mumpy={best_cp*1000:8.1f}ms  speedup={best_np/best_cp:.2f}x")


bench("sqrt", lambda: np.sqrt(a_np), lambda: cp.sqrt(a_cp))
bench("sin", lambda: np.sin(a_np), lambda: cp.sin(a_cp))
bench("add", lambda: np.add(a_np, a_np), lambda: cp.add(a_cp, a_cp))
bench("sum", lambda: np.sum(a_np), lambda: cp.sum(a_cp))
