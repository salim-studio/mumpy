<p align="center">
  <img src="https://raw.githubusercontent.com/salim-studio/mumpy/main/assets/banner.svg" alt="mumpy banner" width="100%"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/salim-studio/mumpy/main/assets/logo.svg" alt="mumpy logo" width="96"/>
</p>

<h1 align="center">mumpy</h1>

<p align="center"><strong>NumPy you know. Speed you feel. Tools you actually need.</strong></p>

<p align="center">
  <a href="https://github.com/salim-studio/mumpy/actions"><img src="https://github.com/salim-studio/mumpy/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/version-0.2.1-4F46E5" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.9%2B-06B6D4" alt="python"/>
  <img src="https://img.shields.io/badge/numpy-compatible-013243" alt="numpy compatible"/>
  <img src="https://img.shields.io/badge/license-MIT-FDE047" alt="license"/>
</p>

**mumpy** is a drop-in, NumPy-compatible library that goes further: multithreaded compute,
fused operations, plus a built-in toolkit for data loading, SQL databases, dataframes,
statistics, preprocessing, classic ML and tiny deep learning — with zero hard dependencies
beyond NumPy.

```python
import mumpy as mp

# 1) Faster NumPy (parallel ufuncs + fused ops)
a = mp.arange(10_000_000)
b = mp.sqrt(a)            # multithreaded over all cores
c = mp.fma(a, b, 1.0)     # a*b+c in a single pass, half the peak memory

# 2) Databases (sqlite built-in; postgres/mysql via SQLAlchemy, OLAP via DuckDB)
db = mp.db.connect("data.db")
db.create_table("users", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
db.insert_many("users", [{"name": "ada", "age": 36}])
ages = db.read_numpy("SELECT age FROM users")

# 3) Lightweight DataFrame for analysts (no pandas required)
df = mp.frame.DataFrame({"age": [20, 30, 40], "salary": [100, 200, 300]})
df.describe()
df.groupby("age").mean()
df.query("age > 25")

# 4) Classic ML with a scikit-learn-like API (pure NumPy)
model = mp.ml.LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# 5) Tiny deep learning (pure NumPy)
net = mp.nn.Sequential([mp.nn.Dense(4, 16), mp.nn.ReLU(), mp.nn.Dense(16, 1)])
net.fit(X, y, epochs=200, lr=0.01)
```

## Why faster than NumPy?

| Technique | Detail |
|---|---|
| Parallel ufuncs | Compute-bound element-wise ops (`sqrt/exp/sin/cos/…`) split across a thread pool |
| Fused ops | `fma/fms/fnma/lerp` instead of `a*b+c` (half the peak memory, one pass) |
| Parallel FFT | `scipy.fft` with `workers=-1` when SciPy is installed |
| `einsum(optimize=True)` | Faster contraction paths |
| Contiguous linalg | `batch_matmul/cho_solve/ridge_solve` ensure cache-friendly layouts |

> Honest note: bandwidth-bound reductions (`sum/mean`) delegate to NumPy directly —
> a single thread already saturates RAM bandwidth, and threading only adds overhead there.

## Modules

| Module | For | Highlights |
|---|---|---|
| `mp.io` | Everyone | `load_csv/save_csv/load_npy/load_npz/load_json/memmap/load_parquet/load_excel/read_auto` |
| `mp.db` | Backend devs & analysts | `connect/read_numpy/read_pandas/write_numpy/from_csv/query/to_parquet` |
| `mp.frame` | Data analysts | `DataFrame/filter/sort/groupby/merge/describe/corr/read_csv/read_sql/to_pandas` |
| `mp.stats` | Exploratory analysis | `describe/corr/zscore/iqr_outliers/histogram/skew/kurtosis/crosstab` |
| `mp.preprocessing` | Data science | `StandardScaler/MinMaxScaler/RobustScaler/OneHotEncoder/LabelEncoder/SimpleImputer/train_test_split/Pipeline` |
| `mp.metrics` | Model evaluation | `mse/rmse/mae/r2/accuracy/precision/recall/f1/confusion_matrix/roc_auc/log_loss` |
| `mp.ml` | Machine learning | `LinearRegression/Ridge/LogisticRegression/KNN/NaiveBayes/DecisionTree/RandomForest/KMeans/PCA` |
| `mp.nn` | Deep learning | `Sequential/Dense/ReLU/Sigmoid/Tanh/Softmax/Dropout/Adam/SGD/MSE/BCE/CrossEntropy` |
| `mp.viz` | Quick plots | `hist/scatter/line/heatmap/corr_heatmap` (matplotlib, optional) |
| `mp.utils` | Everyone | `seed/timer/timeit/one_hot/standardize/fill_nan/memory_usage` |

## Installation

```bash
pip install mumpy-toolkit            # core (numpy only)
pip install "mumpy-toolkit[fast]"    # + scipy for parallel FFT
pip install "mumpy-toolkit[all]"     # scipy, pandas, pyarrow, sqlalchemy, duckdb, matplotlib
```

From source:

```bash
git clone https://github.com/salim-studio/mumpy.git
cd mumpy
pip install -e ".[all]"
```

## Compatibility

- `mumpy.ndarray` subclasses `np.ndarray` — any NumPy-consuming code accepts it unchanged.
- `mp.asnumpy(x)` returns a plain zero-copy `np.ndarray` view.
- `mp.set_workers(1)` reproduces pure-NumPy behavior for fair comparison.
- `DataFrame.to_pandas()/from_pandas()` bridge to pandas; `Database` works with stdlib
  sqlite and optionally SQLAlchemy (PostgreSQL/MySQL) and DuckDB.
- Migrating from the old name? `import cumpy` still works via a compatibility shim.

## Benchmarks & tests

```bash
python benchmarks/bench.py
python -m pytest tests/ -q
python examples_mumpy.py   # end-to-end: synthetic data -> DB -> DataFrame -> ML
```

## End-to-end example (synthetic data → DB → ML)

```python
import numpy as np
import mumpy as mp

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

with mp.db.connect(":memory:") as db:
    db.write_numpy("data", np.column_stack([X, y]), columns=["f0", "f1", "f2", "label"])
    df = mp.frame.DataFrame.read_sql("SELECT * FROM data", db)

print(df.describe())
Xtr, Xte, ytr, yte = mp.ml.train_test_split(
    df.to_numpy(["f0", "f1", "f2"]), df["label"].to_numpy(), test_size=0.2)
acc = mp.metrics.accuracy(
    yte, mp.ml.LogisticRegression(lr=0.5, epochs=500).fit(Xtr, ytr).predict(Xte))
print("accuracy:", acc)
```

## Roadmap

- [ ] PyPI release + versioned changelog
- [ ] Optional Rust/Numba backend for ufuncs
- [ ] `mp.sql` query builder + lazy frames
- [ ] More estimators (Gradient Boosting, Isolation Forest)
- [ ] ONNX export for `mp.nn`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests are welcome via
[issues](https://github.com/salim-studio/mumpy/issues) — please use the templates.

## License

MIT — see [LICENSE](LICENSE). © salim-studio.
