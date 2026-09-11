"""mumpy.random: numpy.random-compatible API over np.random.Generator (PCG64)."""
from __future__ import annotations

import numpy as np

__all__ = ["default_rng", "rand", "randn", "randint", "random", "choice",
           "shuffle", "permutation", "normal", "uniform", "seed", "Generator",
           "integers", "exponential", "poisson", "binomial", "multivariate_normal",
           "beta", "gamma", "multinomial", "standard_cauchy", "laplace"]

_global_rng = np.random.default_rng()


def default_rng(seed=None):
    return np.random.default_rng(seed)


def seed(s=None):
    global _global_rng
    _global_rng = np.random.default_rng(s)


def rand(*args):
    return _global_rng.random(args if args else None)


def random(size=None):
    return _global_rng.random(size)


def randn(*args):
    return _global_rng.standard_normal(args if args else None)


def randint(low, high=None, size=None, dtype=int):
    return _global_rng.integers(low, high, size=size, dtype=dtype, endpoint=False)


def integers(low, high=None, size=None, dtype=np.int64, endpoint=False):
    return _global_rng.integers(low, high, size=size, dtype=dtype, endpoint=endpoint)


def choice(a, size=None, replace=True, p=None):
    return _global_rng.choice(a, size=size, replace=replace, p=p)


def shuffle(x):
    return _global_rng.shuffle(x)


def permutation(x):
    return _global_rng.permutation(x)


def normal(loc=0.0, scale=1.0, size=None):
    return _global_rng.normal(loc, scale, size)


def uniform(low=0.0, high=1.0, size=None):
    return _global_rng.uniform(low, high, size)


def exponential(scale=1.0, size=None):
    return _global_rng.exponential(scale, size)


def poisson(lam=1.0, size=None):
    return _global_rng.poisson(lam, size)


def binomial(n, p, size=None):
    return _global_rng.binomial(n, p, size)


def multivariate_normal(mean, cov, size=None):
    return _global_rng.multivariate_normal(mean, cov, size)


def beta(a, b, size=None):
    return _global_rng.beta(a, b, size)


def gamma(shape, scale=1.0, size=None):
    return _global_rng.gamma(shape, scale, size)


def multinomial(n, pvals, size=None):
    return _global_rng.multinomial(n, pvals, size)


def standard_cauchy(size=None):
    return _global_rng.standard_cauchy(size)


def laplace(loc=0.0, scale=1.0, size=None):
    return _global_rng.laplace(loc, scale, size)


Generator = np.random.Generator
