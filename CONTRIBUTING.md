# Contributing to mumpy

Thanks for your interest! mumpy stays intentionally lean: **NumPy + stdlib core**,
everything else is an optional extra.

## Setup

```bash
git clone https://github.com/salim-studio/mumpy.git
cd mumpy
pip install -e ".[all]"   # or ".[fast]" for the minimum
python -m pytest tests/ -q
```

## Ground rules

1. **NumPy-compatible first** — same function names and semantics as NumPy where they overlap.
2. **No new hard dependencies** — put integrations behind optional extras
   (`fast`, `io`, `db`, `viz`, `all`) with a clear `ImportError` message.
3. **Tests for new behavior** — add cases to `tests/test_mumpy2.py` (or a focused new file).
4. **Docs follow code** — update `README.md` and `CHANGELOG.md` when the public API changes.

## Pull requests

- Keep PRs small and focused; one feature/fix per PR.
- Run `python -m pytest tests/ -q` and `python examples_mumpy.py` before opening.
- Use the PR template checklist.

## Reporting bugs

Open an issue with the bug template and include a minimal reproduction snippet.
