# mumpy ⚡ — numpy-compatible, faster + full data-science toolkit

مكتبة Python بنفس API الخاص بـ numpy لكن أسرع في العمليات الكبيرة،
**ومعها كل ما يحتاجه المطور ومحلل البيانات ومهندس تعلم الآلة**:

```python
import mumpy as cp

# 1) numpy أسرع (متوازي + fused)
a = cp.arange(10_000_000)
b = cp.sqrt(a)
c = cp.fma(a, b, 1.0)          # a*b+c بتمريرة واحدة وذاكرة أقل
cp.linalg.solve, cp.fft.fft, cp.random.rand

# 2) قواعد بيانات (sqlite افتراضياً + postgres/mysql/duckdb اختياري)
db = cp.db.connect("data.db")
db.create_table("users", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
db.insert_many("users", [{"name": "salim", "age": 30}])
arr = db.read_numpy("SELECT age FROM users")

# 3) DataFrame خفيف للمحللين (بدون pandas)
df = cp.frame.DataFrame({"age": [20, 30, 40], "salary": [100, 200, 300]})
df.describe(), df.groupby("age").mean(), df.query("age > 25")

# 4) ML بأسلوب sklearn (pure numpy)
model = cp.ml.LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# 5) Deep learning خفيف (pure numpy)
net = cp.nn.Sequential([cp.nn.Dense(4, 16), cp.nn.ReLU(), cp.nn.Dense(16, 1)])
net.fit(X, y, epochs=200, lr=0.01)
```

## لماذا أسرع من numpy؟

| التقنية | التفاصيل |
|---|---|
| ufuncs متوازية | `sqrt/exp/sin/cos/...` (compute-bound) تتقسم على ThreadPool |
| دمج العمليات | `fma/fms/fnma/lerp` بدل `a*b+c` (نصف الذروة memory) |
| FFT متوازي | `scipy.fft workers=-1` عند توفره |
| `einsum(optimize=True)` | مسارات contraction أسرع |
| linalg متجاورة | `batch_matmul/cho_solve/ridge_solve` بذاكرة contiguous |

> ملاحظة أمانة: عمليات `sum/mean` المرتبطة بعرض النطاق (bandwidth-bound) تستدعي
> numpy مباشرة — خيط واحد يشبع الرام أصلاً.

## الوحدات

| الوحدة | لمن؟ | أهم الدوال |
|---|---|---|
| `cp.io` | الجميع | `load_csv/save_csv/load_npy/load_npz/load_json/memmap/load_parquet` |
| `cp.db` | مطورو backend/تحليل | `connect/read_numpy/read_pandas/write_numpy/from_csv/query` |
| `cp.frame` | محللو البيانات | `DataFrame/filter/sort/groupby/merge/describe/corr/read_csv/read_sql` |
| `cp.stats` | محللو البيانات | `describe/corr/zscore/iqr_outliers/histogram/skew/kurtosis/crosstab` |
| `cp.preprocessing` | data science | `StandardScaler/MinMaxScaler/OneHotEncoder/LabelEncoder/SimpleImputer/train_test_split/Pipeline` |
| `cp.metrics` | ML | `mse/rmse/mae/r2/accuracy/precision/recall/f1/confusion_matrix/roc_auc` |
| `cp.ml` | ML | `LinearRegression/Ridge/LogisticRegression/KNN/NaiveBayes/DecisionTree/RandomForest/KMeans/PCA` |
| `cp.nn` | Deep learning | `Sequential/Dense/ReLU/Sigmoid/Softmax/Adam/SGD/MSE/BCE/CrossEntropy` |
| `cp.viz` | تحليل سريع | `hist/scatter/line/heatmap/corr_heatmap` |
| `cp.utils` | الجميع | `seed/timer/one_hot/standardize/fill_nan` |

## التوافق

- `mumpy.ndarray` ترث من `np.ndarray` → أي كود numpy يقبلها بدون تعديل.
- `cp.asnumpy(x)` ترجع `np.ndarray` عادية zero-copy.
- `cp.set_workers(1)` = سلوك numpy الصرف (للمقارنة).
- `DataFrame.to_pandas()/from_pandas()` للتكامل مع pandas.
- `Database` تعمل مع sqlite stdlib، وتدعم sqlalchemy (postgres/mysql) و duckdb إن وُجدت.

## تثبيت

```
pip install -e .
pip install -e ".[fast]"   # scipy للـ fft
pip install -e ".[all]"    # كل الإضافات: scipy/pandas/pyarrow/sqlalchemy/duckdb/matplotlib
```

## بنش مارك واختبارات

```
python benchmarks/bench.py
python -m pytest tests/test_mumpy.py tests/test_mumpy2.py -q
```

## مثال end-to-end (CSV → DB → ML)

```python
import mumpy as cp
import numpy as np

# توليد بيانات
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)); y = (X[:, 0] + X[:, 1] > 0).astype(int)

# حفظ واسترجاع عبر DB
with cp.db.connect(":memory:") as db:
    db.write_numpy("data", np.column_stack([X, y]), columns=["f0", "f1", "f2", "label"])
    df = cp.frame.DataFrame.read_sql("SELECT * FROM data", db)

print(df.describe())
Xtr, Xte, ytr, yte = cp.ml.train_test_split(df.to_numpy(["f0", "f1", "f2"]),
                                            df["label"].to_numpy(), test_size=0.2)
acc = cp.metrics.accuracy(yte, cp.ml.LogisticRegression(lr=0.5, epochs=500).fit(Xtr, ytr).predict(Xte))
print("accuracy:", acc)
```
