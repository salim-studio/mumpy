"""Extended tests for mumpy 0.2: io, db, frame, stats, preprocessing, metrics, ml, nn."""
import numpy as np
import mumpy as cp


def test_fused_ops():
    a = cp.arange(1000.0)
    b = cp.arange(1000.0)
    assert np.allclose(cp.fma(a, b, 1.0), a * b + 1.0)
    assert np.allclose(cp.fms(a, b, 1.0), a * b - 1.0)
    assert np.allclose(cp.fnma(a, b, 1.0), 1.0 - a * b)
    assert np.allclose(cp.lerp(a, b, 0.5), a + (b - a) * 0.5)
    assert np.allclose(cp.expm1(a[:5]), np.expm1(np.arange(5.0)))
    assert np.allclose(cp.log1p(a[1:6]), np.log1p(np.arange(1.0, 6.0)))


def test_io(tmp_path):
    p = str(tmp_path / "a.csv")
    cp.io.save_csv(p, np.arange(6).reshape(3, 2), header=["x", "y"])
    data, names = cp.io.load_csv(p)
    assert data.shape == (3, 2) and names == ["x", "y"]
    p2 = str(tmp_path / "a.npy")
    cp.io.save_npy(p2, np.arange(5))
    assert np.array_equal(cp.io.load_npy(p2), np.arange(5))
    p3 = str(tmp_path / "a.json")
    cp.io.save_json(p3, {"v": [1, 2, 3]})
    assert cp.io.load_json(p3) == {"v": [1, 2, 3]}


def test_db():
    db = cp.db.connect(":memory:")
    db.create_table("t", {"id": "INTEGER PRIMARY KEY", "x": "REAL"})
    db.insert_many("t", [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
    arr = db.read_numpy("SELECT x FROM t ORDER BY x")
    assert np.allclose(arr.ravel(), [1, 2, 3])
    assert db.query("SELECT COUNT(*) c FROM t")[0]["c"] == 3
    n = db.write_numpy("m", np.array([[1.0, 2.0], [3.0, 4.0]]), columns=["a", "b"])
    assert n == 2
    db.close()


def test_frame():
    df = cp.frame.DataFrame({"age": [20, 30, 40], "s": [100.0, 200.0, 300.0]})
    assert df.shape == (3, 2)
    assert df.head(2).shape == (2, 2)
    assert df["age"].mean() == 30.0
    f = df.filter(np.array([True, False, True]))
    assert len(f) == 2
    g = df.groupby("age").mean()
    assert len(g) == 3
    d = df.describe()
    assert d["age"]["mean"] == 30.0
    m = df.merge(cp.frame.DataFrame({"age": [20, 30], "c": ["a", "b"]}), on="age")
    assert len(m) == 2
    q = df.query("age > 25")
    assert len(q) == 2


def test_stats_preprocessing_metrics():
    x = np.array([1.0, 2.0, 3.0, 100.0])
    assert cp.stats.describe(x)["count"] == 4
    mask, _ = cp.stats.iqr_outliers(x)
    assert bool(mask[-1]) is True
    sc = cp.preprocessing.StandardScaler().fit(x.reshape(-1, 1))
    assert abs(sc.transform(x.reshape(-1, 1)).mean()) < 1e-9
    assert cp.metrics.mse([1, 2], [1, 2]) == 0.0
    assert cp.metrics.accuracy([0, 1, 1], [0, 1, 0]) == 2 / 3
    assert cp.metrics.r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    Xtr, Xte, ytr, yte = cp.preprocessing.train_test_split(
        np.arange(10), np.arange(10), test_size=0.2, random_state=0)
    assert len(Xte) == 2 and len(Xtr) == 8


def test_ml():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = cp.ml.LogisticRegression(lr=0.5, epochs=300).fit(X, y)
    assert cp.metrics.accuracy(y, clf.predict(X)) > 0.9
    Xr = rng.normal(size=(40, 1))
    yr = 3 * Xr.ravel() + 1
    reg = cp.ml.LinearRegression().fit(Xr, yr)
    assert reg.score(Xr, yr) > 0.99
    km = cp.ml.KMeans(n_clusters=2, seed=0).fit(rng.normal(size=(30, 2)))
    assert len(set(km.labels_.tolist())) == 2
    pca = cp.ml.PCA(n_components=1).fit(rng.normal(size=(20, 3)))
    assert pca.transform(rng.normal(size=(5, 3))).shape == (5, 1)
    knn = cp.ml.KNNClassifier(k=3).fit(X, y)
    assert knn.predict(X).shape == (60,)


def test_nn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    model = cp.nn.Sequential([cp.nn.Dense(2, 8, seed=0), cp.nn.ReLU(),
                              cp.nn.Dense(8, 1, seed=1), cp.nn.Sigmoid()])
    hist = model.fit(X, y, epochs=50, lr=0.05, loss="bce", verbose=False)
    assert hist[-1] < hist[0]
    pred = (model.predict(X) > 0.5).astype(int).ravel()
    assert (pred == y.ravel()).mean() > 0.8


def test_utils_fft_linalg():
    assert cp.utils.memory_usage(np.zeros((10, 10))) == 800
    assert cp.fft.has_scipy() in (True, False)
    assert np.allclose(cp.fft.fft([1, 2, 3, 4]), np.fft.fft([1, 2, 3, 4]))
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    assert np.allclose(cp.linalg.cho_solve(A, b), np.linalg.solve(A, b))
    assert cp.random.randint(0, 5, size=(3,)).shape == (3,)
