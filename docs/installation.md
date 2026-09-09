# Installation

MFTT supports Python 3.10 or newer. The standard installation includes the
library and everything needed to run the tutorials and test suite.

## Create an isolated environment

We recommend setting up a virtual environment specifically for MFTT.

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Run the installation command below after activating the environment. To leave
it later, run `deactivate`.

## Install MFTT

Clone the repository and enter its root directory:

```bash
git clone https://github.com/sandialabs/multifidelity-triangular-transport.git
cd multifidelity-triangular-transport
```

Then run the standard installation:

```bash
python -m pip install .
```

This installs MFTT and everything needed to run the tutorials and test suite;
there are no separate tutorial or test profiles to install.

| Dependency | Version |
| --- | --- |
| NumPy | `>=1.21` |
| SciPy | `>=1.8` |
| Jupyter | `>=1.0` |
| Matplotlib | `>=3.6` |
| nbconvert | `>=7.0` |
| pytest | `>=7.0` |

## Verify the installation

From the root of a repository clone, run:

```bash
python -m pytest
```

A successful run confirms that MFTT and its dependencies work in the active
environment.

## Run the tutorials

The notebooks are in `docs/tutorials/`. Launch Jupyter from the repository
root:

```bash
jupyter lab docs/tutorials
```

The hosted documentation renders the notebooks and their saved outputs for
reading. Running them locally uses the same standard installation above.
