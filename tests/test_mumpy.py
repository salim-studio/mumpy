"""Compatibility + correctness tests for mumpy vs numpy."""
import numpy as np
import mumpy as cp


def test_creation():
    assert np.array_equal(cp.zeros(5), np.zeros(5))
    assert np.array_equal(cp.ones((2, 3)), np.ones((2, 3)))
    assert np.array_equal(cp.arange(10), np.arange(10))
    assert np.array_equal(cp.linspace(0, 1, 5), np.linspace(0, 1, 5))
    assert np.array_equal(cp.eye(3), np.eye(3))
    assert isinstance(cp.zeros(3), np.ndarray)  # subclass -> passes isinstance


def test_ew():
    a = cp.arange(1000.0)
    b = cp.arange(1000.0)
    assert np.allclose(cp.add(a, b), np.add(a, b))
    assert np.allclose(cp.sqrt(a), np.sqrt(np.arange(1000.0)))
    assert np.allclose(cp.exp(a[:10]), np.exp(np.arange(10.0)))
    assert np.allclose(cp.fma(a, b, 1.0), np.arange(1000.0) ** 2 + 1.0)


def test_reduce():
    a = np.arange(100000.0)
    assert cp.sum(a) == np.sum(a)
    assert cp.mean(a) == np.mean(a)
    assert cp.min(a) == np.min(a)
    assert cp.max(a) == np.max(a)
    assert cp.prod(np.arange(1, 10)) == np.prod(np.arange(1, 10))


def test_linalg_fft_random():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    assert np.allclose(cp.linalg.solve(A, b), np.linalg.solve(A, b))
    assert np.allclose(cp.dot(A, b), np.dot(A, b))
    x = np.random.RandomState(0).randn(1024)
    assert np.allclose(cp.fft.fft(x), np.fft.fft(x))
    cp.random.seed(0)
    assert cp.random.rand(3).shape == (3,)


def test_large_parallel_path():
    # force the parallel (>threshold) code path
    a = cp.arange(200_000.0)
    assert np.allclose(cp.sqrt(a), np.sqrt(np.arange(200_000.0)))
    assert float(cp.sum(a)) == float(np.sum(np.arange(200_000.0)))
