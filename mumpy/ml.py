"""mumpy.ml: classic machine learning in pure numpy (sklearn-like API).

Estimators: LinearRegression / Ridge / LogisticRegression / KNN /
NaiveBayes / DecisionTree / RandomForest-lite / KMeans / PCA / train_test_split

    from mumpy import ml
    model = ml.LogisticRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    ml.cross_val_score(model, X, y, cv=5)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "LinearRegression", "Ridge", "LogisticRegression",
    "KNNClassifier", "KNNRegressor", "GaussianNB",
    "DecisionTreeClassifier", "RandomForestClassifier",
    "KMeans", "PCA", "train_test_split", "cross_val_score",
    "kfold", "standardize_for",
]


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    rng = np.random.default_rng(random_state)
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    n = len(X)
    if stratify is not None:
        # stratified split
        st = np.asanyarray(stratify)
        tr_idx, te_idx = [], []
        for u in np.unique(st):
            idx = np.where(st == u)[0]
            if shuffle:
                idx = rng.permutation(idx)
            k = int(len(idx) * test_size)
            te_idx += idx[:k].tolist()
            tr_idx += idx[k:].tolist()
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        if shuffle:
            tr_idx = rng.permutation(tr_idx)
            te_idx = rng.permutation(te_idx)
    else:
        idx = np.arange(n)
        if shuffle:
            idx = rng.permutation(idx)
        k = int(n * test_size) if isinstance(test_size, float) else int(test_size)
        te_idx, tr_idx = idx[:k], idx[k:]
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]


def kfold(n, cv=5, shuffle=True, random_state=None):
    idx = np.arange(n)
    if shuffle:
        idx = np.random.default_rng(random_state).permutation(idx)
    folds = np.array_split(idx, cv)
    for i in range(cv):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(cv) if j != i])
        yield tr, te


def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    from .metrics import accuracy, r2_score
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    scores = []
    for tr, te in kfold(len(X), cv=cv):
        from copy import deepcopy
        est = deepcopy(estimator)
        est.fit(X[tr], y[tr])
        p = est.predict(X[te])
        scores.append(accuracy(y[te], p) if scoring == "accuracy" else r2_score(y[te], p))
    return np.array(scores)


def standardize_for(X_train, X_test=None):
    from .preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train)
    if X_test is None:
        return sc.transform(X_train), sc
    return sc.transform(X_train), sc.transform(X_test), sc


def _add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


class LinearRegression:
    """Ordinary least squares via lstsq (stable) or normal equations."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        A = _add_bias(X) if self.fit_intercept else X
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        return self

    def predict(self, X):
        X = np.asanyarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))


class Ridge(LinearRegression):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(fit_intercept)
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        if self.fit_intercept:
            mu_x, mu_y = X.mean(0), y.mean()
            Xc, yc = X - mu_x, y - mu_y
            A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
            self.intercept_ = float(mu_y - mu_x @ self.coef_)
        else:
            A = X.T @ X + self.alpha * np.eye(X.shape[1])
            self.coef_ = np.linalg.solve(A, X.T @ y)
            self.intercept_ = 0.0
        return self


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1 + e)
    return out


class LogisticRegression:
    """Binary (+ softmax multinomial) logistic regression with L2."""

    def __init__(self, lr=0.1, epochs=1000, l2=1e-4, tol=1e-6, verbose=False, seed=0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.classes_ = np.unique(y)
        n, d = X.shape
        if len(self.classes_) == 2:
            yc = (y == self.classes_[1]).astype(float)
            w = np.zeros(d)
            b = 0.0
            prev = np.inf
            for ep in range(self.epochs):
                z = X @ w + b
                p = _sigmoid(z)
                err = p - yc
                gw = X.T @ err / n + self.l2 * w
                gb = err.mean()
                w -= self.lr * gw
                b -= self.lr * gb
                loss = -(yc * np.log(p + 1e-12) + (1 - yc) * np.log(1 - p + 1e-12)).mean()
                if abs(prev - loss) < self.tol:
                    break
                prev = loss
            self.coef_ = w
            self.intercept_ = float(b)
        else:
            K = len(self.classes_)
            W = rng.normal(0, 0.01, (d, K))
            B = np.zeros(K)
            idx = {v: i for i, v in enumerate(self.classes_.tolist())}
            Y = np.zeros((n, K))
            Y[np.arange(n), [idx[v] for v in y.tolist()]] = 1.0
            for ep in range(self.epochs):
                Z = X @ W + B
                Z -= Z.max(1, keepdims=True)
                P = np.exp(Z)
                P /= P.sum(1, keepdims=True)
                G = (P - Y) / n
                W -= self.lr * (X.T @ G + self.l2 * W)
                B -= self.lr * G.sum(0)
            self.coef_ = W
            self.intercept_ = B
        return self

    def predict_proba(self, X):
        X = np.asanyarray(X, dtype=float)
        if len(self.classes_) == 2:
            p1 = _sigmoid(X @ self.coef_ + self.intercept_)
            return np.column_stack([1 - p1, p1])
        Z = X @ self.coef_ + self.intercept_
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]


class _KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asanyarray(X, dtype=float)
        self.y_ = np.asanyarray(y)
        return self

    def _neighbors(self, X):
        X = np.asanyarray(X, dtype=float)
        # chunked distances to save memory
        idx = []
        bs = 512
        for s in range(0, len(X), bs):
            d2 = ((X[s:s + bs, None, :] - self.X_[None, :, :]) ** 2).sum(-1)
            idx.append(np.argpartition(d2, self.k, axis=1)[:, :self.k])
        return np.vstack(idx)


class KNNClassifier(_KNN):
    def predict(self, X):
        idx = self._neighbors(X)
        out = []
        for row in idx:
            vals, counts = np.unique(self.y_[row], return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.array(out)

    def predict_proba(self, X):
        idx = self._neighbors(X)
        classes = np.unique(self.y_)
        P = np.zeros((len(X), len(classes)))
        for i, row in enumerate(idx):
            for j, c in enumerate(classes):
                P[i, j] = (self.y_[row] == c).mean()
        return P


class KNNRegressor(_KNN):
    def predict(self, X):
        idx = self._neighbors(X)
        return np.array([self.y_[row].astype(float).mean() for row in idx])


class GaussianNB:
    def fit(self, X, y):
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y).ravel()
        self.classes_ = np.unique(y)
        self.theta_ = np.array([X[y == c].mean(0) for c in self.classes_])
        self.sigma_ = np.array([X[y == c].var(0) + 1e-9 for c in self.classes_])
        self.priors_ = np.array([(y == c).mean() for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asanyarray(X, dtype=float)
        jll = []
        for t, s, p in zip(self.theta_, self.sigma_, self.priors_):
            jll.append(-0.5 * np.sum(np.log(2 * np.pi * s) + (X - t) ** 2 / s, axis=1) + np.log(p))
        return self.classes_[np.argmax(np.column_stack(jll), axis=1)]


class DecisionTreeClassifier:
    """CART binary tree (gini), max_depth + min_samples_split."""

    def __init__(self, max_depth=8, min_samples_split=2, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(np.asanyarray(y).ravel())
        self._map = {v: i for i, v in enumerate(self.classes_.tolist())}
        yi = np.array([self._map[v] for v in np.asanyarray(y).ravel().tolist()])
        self.tree_ = self._build(np.asanyarray(X, dtype=float), yi, 0)
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return 1 - (p ** 2).sum()

    def _build(self, X, y, depth):
        n, d = X.shape
        counts = np.bincount(y, minlength=len(self.classes_))
        pred = int(np.argmax(counts))
        if depth >= self.max_depth or n < self.min_samples_split or self._gini(y) == 0:
            return {"leaf": True, "pred": pred}
        best = (1e9, None, None)
        rng = np.random.default_rng(self.seed + depth)
        feats = rng.choice(d, min(d, max(1, int(np.sqrt(d)))), replace=False) if d > 4 else range(d)
        for f in feats:
            ths = np.unique(X[:, f])
            if len(ths) > 20:
                ths = np.percentile(ths, np.linspace(0, 100, 20))
            for t in ths:
                l, r = y[X[:, f] <= t], y[X[:, f] > t]
                if len(l) == 0 or len(r) == 0:
                    continue
                g = (len(l) * self._gini(l) + len(r) * self._gini(r)) / n
                if g < best[0]:
                    best = (g, f, t)
        if best[1] is None:
            return {"leaf": True, "pred": pred}
        _, f, t = best
        lm = X[:, f] <= t
        return {"leaf": False, "f": int(f), "t": float(t),
                "l": self._build(X[lm], y[lm], depth + 1),
                "r": self._build(X[~lm], y[~lm], depth + 1)}

    def _predict_one(self, x, node):
        while not node["leaf"]:
            node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
        return node["pred"]

    def predict(self, X):
        X = np.asanyarray(X, dtype=float)
        idx = np.array([self._predict_one(x, self.tree_) for x in X])
        return self.classes_[idx]


class RandomForestClassifier:
    def __init__(self, n_estimators=20, max_depth=8, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        n = len(X)
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, n)
            t = DecisionTreeClassifier(max_depth=self.max_depth, seed=self.seed + i)
            t.fit(X[idx], y[idx])
            self.trees_.append(t)
        self.classes_ = self.trees_[0].classes_
        return self

    def predict(self, X):
        votes = np.column_stack([t.predict(X) for t in self.trees_])
        out = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.array(out)


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, seed=0, n_init=3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.n_init = n_init

    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        best_inertia, best = np.inf, None
        for init in range(self.n_init):
            rng = np.random.default_rng(self.seed + init)
            C = X[rng.choice(len(X), self.n_clusters, replace=False)]
            for _ in range(self.max_iter):
                d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
                lab = d2.argmin(1)
                Cn = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k]
                               for k in range(self.n_clusters)])
                if np.linalg.norm(Cn - C) < self.tol:
                    C = Cn
                    break
                C = Cn
            inertia = float(((X - C[lab]) ** 2).sum())
            if inertia < best_inertia:
                best_inertia, best = inertia, (C, lab)
        self.cluster_centers_, labels = best
        self.labels_ = labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asanyarray(X, dtype=float)
        d2 = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(-1)
        return d2.argmin(1)

    def fit_predict(self, X):
        return self.fit(X).labels_


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S ** 2) / (len(X) - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        return self

    def transform(self, X):
        return (np.asanyarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asanyarray(Z) @ self.components_ + self.mean_
