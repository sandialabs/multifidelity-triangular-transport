# MFTT Developer Guide

Developer and contributor notes are retained in
[docs/development.md](docs/development.md) for repository use but are not part
of the public Sphinx documentation site.

Install the package in editable development mode with its test, tutorial,
build, and documentation tools:

```bash
python -m pip install -e ".[dev]"
```

Build the site locally with:

```bash
PYTHONPATH=src python -m sphinx -W --keep-going -b html docs docs/_build/html
```
