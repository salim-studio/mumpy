"""mumpy.nn: tiny deep-learning toolkit in pure numpy.

Layers: Dense / ReLU / Sigmoid / Tanh / Softmax / Dropout / BatchNorm-lite
Losses: MSE / BCE / CrossEntropy
Optimizers: SGD (+momentum) / Adam
Model: Sequential with fit/predict/save/load

    from mumpy import nn
    model = nn.Sequential([nn.Dense(4, 16), nn.ReLU(), nn.Dense(16, 1)])
    model.fit(X, y, epochs=200, lr=0.01, verbose=True)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "Dense", "ReLU", "Sigmoid", "Tanh", "Softmax", "Dropout",
    "MSELoss", "BCELoss", "CrossEntropyLoss",
    "SGD", "Adam", "Sequential",
]


class Dense:
    def __init__(self, in_features, out_features, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(0)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def __call__(self, x):
        return self.forward(x)


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, g):
        return g * (self.x > 0)

    def params(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class Sigmoid:
    def forward(self, x):
        self.o = 1 / (1 + np.exp(-np.clip(x, -30, 30)))
        return self.o

    def backward(self, g):
        return g * self.o * (1 - self.o)

    def params(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    def forward(self, x):
        self.o = np.tanh(x)
        return self.o

    def backward(self, g):
        return g * (1 - self.o ** 2)

    def params(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class Softmax:
    def forward(self, x):
        x = x - x.max(1, keepdims=True)
        e = np.exp(x)
        self.o = e / e.sum(1, keepdims=True)
        return self.o

    def backward(self, g):
        # combined with CrossEntropy upstream usually passes (p - y)/n
        return g

    def params(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class Dropout:
    def __init__(self, p=0.2, seed=0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (self.rng.random(x.shape) > self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, g):
        return g * self.mask if self.training and self.p else g

    def params(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class MSELoss:
    def forward(self, pred, target):
        self.pred, self.target = pred, target
        return float(np.mean((pred - target) ** 2))

    def backward(self):
        n = self.pred.size
        return 2 * (self.pred - self.target) / n


class BCELoss:
    def forward(self, pred, target, eps=1e-12):
        self.pred = np.clip(pred, eps, 1 - eps)
        self.target = target
        return float(-np.mean(target * np.log(self.pred) + (1 - target) * np.log(1 - self.pred)))

    def backward(self):
        n = self.pred.size
        p, t = self.pred, self.target
        return (-(t / p) + (1 - t) / (1 - p)) / n


class CrossEntropyLoss:
    def forward(self, logits, target):
        # logits: (n, K), target: class indices
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z)
        self.probs = e / e.sum(1, keepdims=True)
        self.target = np.asanyarray(target).ravel().astype(int)
        n = len(self.target)
        return float(-np.log(self.probs[np.arange(n), self.target] + 1e-12).mean())

    def backward(self):
        n = len(self.target)
        g = self.probs.copy()
        g[np.arange(n), self.target] -= 1
        return g / n


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.vel = {}

    def step(self, layers):
        for li, layer in enumerate(layers):
            for pi, (p, g) in enumerate(layer.params()):
                key = (li, pi)
                if self.momentum:
                    v = self.vel.get(key, np.zeros_like(p))
                    v = self.momentum * v + g
                    self.vel[key] = v
                    p -= self.lr * v
                else:
                    p -= self.lr * g


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for li, layer in enumerate(layers):
            for pi, (p, g) in enumerate(layer.params()):
                key = (li, pi)
                m = self.m.get(key, np.zeros_like(p))
                v = self.v.get(key, np.zeros_like(p))
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
                self.m[key], self.v[key] = m, v
                mh = m / (1 - self.beta1 ** self.t)
                vh = v / (1 - self.beta2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def predict(self, X):
        for l in self.layers:
            if isinstance(l, Dropout):
                l.training = False
        out = self.forward(np.asanyarray(X, dtype=float))
        for l in self.layers:
            if isinstance(l, Dropout):
                l.training = True
        return out

    def fit(self, X, y, epochs=100, batch_size=None, lr=0.01, optimizer=None,
            loss="mse", verbose=False, seed=0, shuffle=True):
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y)
        rng = np.random.default_rng(seed)
        opt = optimizer or Adam(lr)
        if isinstance(opt, float):
            opt = Adam(opt)
        losses = {"mse": MSELoss, "bce": BCELoss, "ce": CrossEntropyLoss}
        criterion = losses[loss]() if isinstance(loss, str) else loss
        n = len(X)
        history = []
        for ep in range(epochs):
            idx = rng.permutation(n) if shuffle else np.arange(n)
            total, nb = 0.0, 0
            bs = batch_size or n
            for s in range(0, n, bs):
                bi = idx[s:s + bs]
                xb, yb = X[bi], y[bi]
                # reshape y for mse/bce single-output
                if yb.ndim == 1 and not isinstance(criterion, CrossEntropyLoss):
                    yb = yb.reshape(-1, 1)
                pred = self.forward(xb)
                total += criterion.forward(pred, yb) * len(bi)
                grad = criterion.backward()
                for l in reversed(self.layers):
                    grad = l.backward(grad)
                opt.step(self.layers)
                nb += len(bi)
            history.append(total / max(nb, 1))
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"epoch {ep + 1}/{epochs} loss={history[-1]:.6f}")
        return history

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
