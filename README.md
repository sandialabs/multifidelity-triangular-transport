# MFTT: Multifidelity Triangular Transport

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-005f85)](https://sandialabs.github.io/multifidelity-triangular-transport/)
[![Documentation workflow](https://github.com/sandialabs/multifidelity-triangular-transport/actions/workflows/docs.yml/badge.svg)](https://github.com/sandialabs/multifidelity-triangular-transport/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MFTT learns monotone triangular transport maps from samples of one or more
target distributions. It implements the single-fidelity, hierarchical, and
nonhierarchical constructions in the submitted manuscript *Multifidelity
Formulations for Triangular Transport*, with particular attention to settings
where high-fidelity data are scarce.

## Installation

MFTT supports Python 3.10 or newer. Install it from a repository clone:

```bash
git clone https://github.com/sandialabs/multifidelity-triangular-transport.git
cd multifidelity-triangular-transport
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The [installation guide](https://sandialabs.github.io/multifidelity-triangular-transport/installation.html)
includes Windows commands and verification instructions.

## Quick example

```python
import numpy as np

from mftt import MapParams, TriangularMap

rng = np.random.default_rng(2026)
u, v = rng.standard_normal((2, 600))
samples = np.column_stack((u, v + 0.5 * (u**2 - 1.0)))

model = TriangularMap(samples, MapParams(total_order=2))
model.train()

reference_samples = model.evaluate(samples)
recovered_samples = model.inverse(reference_samples[:32])
np.testing.assert_allclose(recovered_samples, samples[:32], atol=1e-8)
```

See the [single-fidelity guide](https://sandialabs.github.io/multifidelity-triangular-transport/single-fidelity.html)
for the full model workflow and interpretation.

## Documentation

The complete rendered documentation is available at
**[sandialabs.github.io/multifidelity-triangular-transport](https://sandialabs.github.io/multifidelity-triangular-transport/)**.

- [Mathematical notation](https://sandialabs.github.io/multifidelity-triangular-transport/notation.html)
- [Method guides](https://sandialabs.github.io/multifidelity-triangular-transport/#methods)
- [Rendered tutorials](https://sandialabs.github.io/multifidelity-triangular-transport/#tutorials)
- [Public API reference](https://sandialabs.github.io/multifidelity-triangular-transport/api.html)

## Citation

The manuscript is the authority for the mathematical definitions and method
names. See the [citation page](https://sandialabs.github.io/multifidelity-triangular-transport/citation.html)
for software authorship and the manuscript reference.

## Development

Contributor setup and local documentation build commands are in the
[developer guide](DEVELOPER_GUIDE.md).

## Authors and license

MFTT is developed by Owen Davis and Gianluca Geraci and released under the
[MIT License](LICENSE). The implemented methods are based on the referenced
manuscript by Owen Davis, Daniel Sharp, Youssef Marzouk, and Gianluca Geraci.
