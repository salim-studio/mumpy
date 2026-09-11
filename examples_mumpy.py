"""End-to-end mumpy demo: CSV -> DB -> DataFrame -> preprocessing -> ML -> metrics."""
import numpy as np
import mumpy as cp

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

print("mumpy:", cp.info())

with cp.db.connect(":memory:") as db:
    db.write_numpy("data", np.column_stack([X, y]), columns=["f0", "f1", "f2", "label"])
    df = cp.frame.DataFrame.read_sql("SELECT * FROM data", db)

print("describe f0:", cp.stats.describe(df["f0"].to_numpy()))
print("corr f0/f1:", cp.stats.corr(df["f0"].to_numpy(), df["f1"].to_numpy()))

X_all = df.to_numpy(["f0", "f1", "f2"])
y_all = df["label"].to_numpy()
Xtr, Xte, ytr, yte = cp.ml.train_test_split(X_all, y_all, test_size=0.2, random_state=0)

sc = cp.preprocessing.StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

for name, model in [
    ("logreg", cp.ml.LogisticRegression(lr=0.5, epochs=500)),
    ("knn", cp.ml.KNNClassifier(k=5)),
    ("tree", cp.ml.DecisionTreeClassifier(max_depth=5)),
]:
    model.fit(Xtr_s, ytr)
    print(f"{name}: acc={cp.metrics.accuracy(yte, model.predict(Xte_s)):.3f}")

# deep learning
net = cp.nn.Sequential([cp.nn.Dense(3, 16, seed=0), cp.nn.ReLU(),
                        cp.nn.Dense(16, 1, seed=1), cp.nn.Sigmoid()])
h = net.fit(Xtr_s, ytr.reshape(-1, 1), epochs=100, lr=0.05, loss="bce")
print(f"MLP: loss {h[0]:.4f} -> {h[-1]:.4f}")
print("OK - mumpy end-to-end works")
