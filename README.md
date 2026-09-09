# MFTT: Multifidelity Triangular Transport

MFTT learns monotone triangular transport maps from samples of one or more
target distributions. It implements the single-fidelity, hierarchical, and
non-hierarchical constructions in the submitted manuscript *Multifidelity
Formulations for Triangular Transport*.

## Install

MFTT supports Python 3.10 or newer. From a clone of the repository, create an
isolated environment and install the runtime package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Verify the installation from the repository root:

```bash
python -m pytest
```

The [installation guide](docs/installation.md) includes Windows commands,
direct GitHub installation, and the complete standard dependency list. The
standard installation is ready to run the tutorials.

## Documentation

The complete documentation is available from the [Sphinx documentation
index](docs/index.md):

- [Notation and mathematical foundations](docs/notation.md)
- [Single-fidelity guide](docs/single-fidelity.md)
- [Hierarchical multifidelity guide](docs/hierarchical.md)
- [Non-hierarchical multifidelity guide](docs/non-hierarchical.md)
- [Shared operations](docs/operations.md)
- [Public API reference](docs/api.md)
- [Single-fidelity notebook](docs/tutorials/single_fidelity_tutorial.ipynb)
- [Hierarchical notebook](docs/tutorials/hierarchical_multifidelity_tutorial.ipynb)
- [Non-hierarchical notebook](docs/tutorials/non_hierarchical_multifidelity_tutorial.ipynb)
- [Developer guide](DEVELOPER_GUIDE.md)

The manuscript is the authority for the mathematical definitions and method
names. This repository's [citation page](docs/citation.md) records the
reference and maps its Sections 3–4 to the documentation.

## Authors and license

MFTT is developed by Owen Davis and Gianluca Geraci and released under the
[MIT License](LICENSE). The implemented methods are based on the referenced
manuscript by Owen Davis, Daniel Sharp, Youssef Marzouk, and Gianluca Geraci.
