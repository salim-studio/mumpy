# Changelog

All notable changes to mumpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - 2026-09-11
### Added
- Renamed `cumpy` → `mumpy` (with `cumpy.py` compatibility shim).
- New modules: `io`, `db` (sqlite/SQLAlchemy/DuckDB), `frame` (DataFrame),
  `stats`, `preprocessing`, `metrics`, `ml` (regression/classification/clustering/PCA),
  `nn` (tiny deep learning), `viz`, `utils`.
- Fused ops `fms/fnma/lerp`, extended parallel ufuncs, `cho_solve/ridge_solve`,
  FFT `convolve/next_fast_len`, expanded `random` distributions.
- English README, logo + banner brand assets, CI, issue/PR templates.

## [0.1.0] - 2026-09-10
### Added
- Initial `cumpy` release: NumPy-compatible `ndarray`, parallel ufuncs,
  fused `fma`, parallel FFT, `linalg`, `random`, benchmarks and tests.
