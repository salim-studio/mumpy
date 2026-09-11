"""mumpy.preprocessing: sklearn-like transformers in pure numpy.

StandardScaler / MinMaxScaler / RobustScaler / MaxAbsScaler
OneHotEncoder / LabelEncoder / OrdinalEncoder
SimpleImputer / train_test_split / PolynomialFeatures / Pipeline
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "OneHotEncoder", "LabelEncoder", "SimpleImputer",
    "train_test_split", "shuffle", "PolynomialFeatures", "Pipeline",
]


class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = np.nanstd(X, axis=0) if self.with_std else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asanyarray(X, dtype=float).copy()
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asanyarray(X, dtype=float).copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        rng = self.data_max_ - self.data_min_
        rng[rng == 0] = 1.0
        lo, hi = self.feature_range
        self.scale_ = (hi - lo) / rng
        self.min_ = lo - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        X = np.asanyarray(X, dtype=float).copy()
        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asanyarray(X, dtype=float).copy()
        return (X - self.min_) / self.scale_


class RobustScaler:
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        qmin, qmax = self.quantile_range
        self.center_ = np.nanmedian(X, axis=0)
        q1 = np.nanpercentile(X, qmin, axis=0)
        q3 = np.nanpercentile(X, qmax, axis=0)
        self.scale_ = q3 - q1
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asanyarray(X, dtype=float) - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MaxAbsScaler:
    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        self.max_abs_ = np.nanmax(np.abs(X), axis=0)
        self.max_abs_[self.max_abs_ == 0] = 1.0
        return self

    def transform(self, X):
        return np.asanyarray(X, dtype=float) / self.max_abs_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self, sparse=False):
        self.sparse = sparse
        self.categories_ = None

    def fit(self, y):
        y = np.asanyarray(y).ravel()
        self.categories_ = np.unique(y)
        return self

    def transform(self, y):
        y = np.asanyarray(y).ravel()
        idx = {v: i for i, v in enumerate(self.categories_.tolist())}
        out = np.zeros((y.size, len(self.categories_)), dtype=float)
        for i, v in enumerate(y.tolist()):
            if v in idx:
                out[i, idx[v]] = 1.0
        return out

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class LabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(np.asanyarray(y).ravel())
        self._map = {v: i for i, v in enumerate(self.classes_.tolist())}
        return self

    def transform(self, y):
        y = np.asanyarray(y).ravel()
        return np.array([self._map[v] for v in y.tolist()])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return np.array([self.classes_[int(i)] for i in np.asanyarray(y).ravel()])


class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        X = np.asanyarray(X, dtype=float)
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            self.statistics_ = np.array([np.unique(c[~np.isnan(c)])[0] if (~np.isnan(c)).any() else self.fill_value
                                         for c in X.T])
        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        self.statistics_ = np.where(np.isnan(self.statistics_), self.fill_value, self.statistics_)
        return self

    def transform(self, X):
        X = np.asanyarray(X, dtype=float).copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(self.statistics_, idx[1])
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True):
    rng = np.random.default_rng(random_state)
    n = len(np.asanyarray(arrays[0]))
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    n_test = int(n * test_size) if isinstance(test_size, float) else int(test_size)
    te, tr = idx[:n_test], idx[n_test:]
    out = []
    for a in arrays:
        a = np.asanyarray(a)
        out += [a[tr], a[te]]
    return out


def shuffle(*arrays, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(np.asanyarray(arrays[0]))
    idx = np.arange(n)
    rng.shuffle(idx)
    return [np.asanyarray(a)[idx] for a in arrays]


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X):
        return self

    def transform(self, X):
        from itertools import combinations_with_replacement
        X = np.asanyarray(X, dtype=float)
        n, d = X.shape
        feats = [np.ones((n, 1))] if self.include_bias else []
        for deg in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(d), deg):
                feats.append(np.prod(X[:, combo], axis=1, keepdims=True))
        return np.hstack(feats)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Pipeline:
    def __init__(self, steps):
        self.steps = steps  # [(name, transformer/estimator)]

    def fit(self, X, y=None):
        Xt = X
        for _, s in self.steps[:-1]:
            Xt = s.fit_transform(Xt) if y is None else s.fit(Xt, y).transform(Xt) if hasattr(s, "transform") else s.fit(Xt, y)
        name, est = self.steps[-1]
        if y is None:
            est.fit(Xt)
        else:
            est.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = X
        for _, s in self.steps[:-1]:
            Xt = s.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    def transform(self, X):
        Xt = X
        for _, s in self.steps:
            Xt = s.transform(Xt)
        return Xt
