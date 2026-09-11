"""Backward-compat shim: `import cumpy` still works after the rename to `mumpy`.

New code should use `import mumpy as mp`.
"""
from mumpy import *  # noqa: F401,F403
from mumpy import (  # noqa: F401
    ndarray, array, asarray, asnumpy, linalg, fft, random, io, db, frame,
    stats, preprocessing, metrics, ml, nn, utils, info, set_workers, set_threshold,
)
import mumpy as _m

__version__ = _m.__version__
