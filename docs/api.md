# Public API reference

The package root is the supported import surface:

```python
import mftt
```

The generated reference below is built directly from the public classes and
functions in `mftt`. Their NumPy-style docstrings are the authoritative source
for signatures, parameters, defaults, return values, and exceptions.

```{toctree}
:maxdepth: 2

api/maps
api/configuration
api/inversion
api/diagnostics
```

The stable package version is available as `mftt.__version__`. The public
map contract is summarized in [shared operations](operations.md), including
the target/reference direction, raw-coordinate behavior, density helpers, and
prefix-conditioned sampling.
