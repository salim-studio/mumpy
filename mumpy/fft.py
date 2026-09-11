"""mumpy.fft: numpy.fft-compatible but uses scipy.fft workers when available."""
from __future__ import annotations

try:
    import scipy.fft as _sf
    _HAS_SCIPY = True
except Exception:
    _sf = None
    _HAS_SCIPY = False

import numpy as _np
import numpy.fft as _nf
from numpy.fft import fftfreq, rfftfreq, fftshift, ifftshift  # noqa: F401

__all__ = ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn",
           "rfftn", "irfftn", "hfft", "ihfft",
           "fftfreq", "rfftfreq", "fftshift", "ifftshift",
           "next_fast_len", "convolve", "has_scipy"]


def has_scipy() -> bool:
    return _HAS_SCIPY


def next_fast_len(n):
    try:
        if _HAS_SCIPY:
            return _sf.next_fast_len(n)
    except Exception:
        pass
    # fallback: next pow2-ish 5-smooth
    m = 1
    while m < n:
        m *= 2
    return m


def _call(name, a, **kw):
    if _HAS_SCIPY:
        kw.setdefault("workers", -1)
        kw.setdefault("overwrite_x", False)
        try:
            return getattr(_sf, name)(a, **{k: v for k, v in kw.items()
                                            if k in ("n", "axis", "norm", "workers",
                                                     "overwrite_x", "s", "axes", "shape")})
        except TypeError:
            pass
    return getattr(_nf, name)(a, **{k: v for k, v in kw.items()
                                    if k in ("n", "axis", "norm", "s", "axes", "shape")})


def fft(a, n=None, axis=-1, norm=None, workers=-1, overwrite_x=False):
    return _call("fft", a, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x)


def ifft(a, n=None, axis=-1, norm=None, workers=-1, overwrite_x=False):
    return _call("ifft", a, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x)


def rfft(a, n=None, axis=-1, norm=None, workers=-1, overwrite_x=False):
    return _call("rfft", a, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x)


def irfft(a, n=None, axis=-1, norm=None, workers=-1, overwrite_x=False):
    return _call("irfft", a, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x)


def fft2(a, s=None, axes=(-2, -1), norm=None, workers=-1, overwrite_x=False):
    return _call("fft2", a, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x)


def ifft2(a, s=None, axes=(-2, -1), norm=None, workers=-1, overwrite_x=False):
    return _call("ifft2", a, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x)


def fftn(a, s=None, axes=None, norm=None, workers=-1, overwrite_x=False):
    return _call("fftn", a, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x)


def ifftn(a, s=None, axes=None, norm=None, workers=-1, overwrite_x=False):
    return _call("ifftn", a, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x)


def rfftn(a, s=None, axes=None, norm=None, workers=-1, overwrite_x=False):
    try:
        return _call("rfftn", a, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x)
    except AttributeError:
        return _nf.rfftn(a, s=s, axes=axes, norm=norm)


def irfftn(a, s=None, axes=None, norm=None, workers=-1, overwrite_x=False):
    try:
        return _call("irfftn", a, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x)
    except AttributeError:
        return _nf.irfftn(a, s=s, axes=axes, norm=norm)


def hfft(a, n=None, axis=-1, norm=None):
    try:
        if _HAS_SCIPY:
            return _sf.hfft(a, n=n, axis=axis, norm=norm)
    except Exception:
        pass
    return _nf.hfft(a, n=n, axis=axis, norm=norm)


def ihfft(a, n=None, axis=-1, norm=None):
    try:
        if _HAS_SCIPY:
            return _sf.ihfft(a, n=n, axis=axis, norm=norm)
    except Exception:
        pass
    return _nf.ihfft(a, n=n, axis=axis, norm=norm)


def convolve(a, b, mode="full"):
    """FFT-based convolution (fast for large signals)."""
    a = _np.asanyarray(a)
    b = _np.asanyarray(b)
    n = a.size + b.size - 1
    N = next_fast_len(n)
    FA = fft(a, n=N)
    FB = fft(b, n=N)
    full = _np.real(ifft(FA * FB))[:n]
    if mode == "full":
        return full
    if mode == "same":
        start = (n - max(a.size, b.size)) // 2
        return full[start:start + max(a.size, b.size)]
    # valid
    start = b.size - 1
    return full[start:start + (a.size - b.size + 1)]
