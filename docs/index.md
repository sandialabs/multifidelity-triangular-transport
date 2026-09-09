# MFTT

:::{div} mftt-hero
**Multifidelity Triangular Transport**

Sample-trained monotone triangular maps for constructing transports with
single- and multifidelity information, especially when high-fidelity data are
scarce.
:::

::::{grid} 1 2 4 4
:gutter: 2
:class-container: mftt-actions

:::{grid-item}
```{button-ref} installation
:ref-type: doc
:color: primary
:expand:

Get Started
```
:::

:::{grid-item}
```{button-link} #tutorials
:color: primary
:outline:
:expand:

Tutorials
```
:::

:::{grid-item}
```{button-ref} api
:ref-type: doc
:color: primary
:outline:
:expand:

API Reference
```
:::

:::{grid-item}
```{button-link} https://github.com/sandialabs/multifidelity-triangular-transport
:color: primary
:outline:
:expand:

GitHub
```
:::
::::

MFTT follows the notation and method names of the submitted manuscript
*Multifidelity Formulations for Triangular Transport*. Choose the formulation
that matches how fidelity information is organized, then use its rendered
notebook for a paper-to-code walkthrough.

## Methods

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Single-fidelity
Learn one monotone triangular transport directly from samples of a target
distribution.

[{octicon}`book;0.9em` Guide](single-fidelity.md) ·
[{octicon}`play;0.9em` Tutorial](tutorials/single_fidelity_tutorial.ipynb)
:::

:::{grid-item-card} Hierarchical multifidelity
Compose stage maps across an ordered fidelity hierarchy using changing- or
fixed-reference training.

[{octicon}`book;0.9em` Guide](hierarchical.md) ·
[{octicon}`play;0.9em` Tutorial](tutorials/hierarchical_multifidelity_tutorial.ipynb)
:::

:::{grid-item-card} Nonhierarchical multifidelity
Combine multiple low-fidelity peer maps inside a coupled high-fidelity
transport.

[{octicon}`book;0.9em` Guide](non-hierarchical.md) ·
[{octicon}`play;0.9em` Tutorial](tutorials/non_hierarchical_multifidelity_tutorial.ipynb)
:::
::::

## Quick installation

Clone the repository and install MFTT into an isolated Python 3.10 or newer
environment:

```bash
git clone https://github.com/sandialabs/multifidelity-triangular-transport.git
cd multifidelity-triangular-transport
python -m pip install .
```

See the [installation guide](installation.md) for environment setup, Windows
commands, and verification instructions.

(tutorials)=
## Tutorials

The rendered notebooks pair the mathematical formulation with complete,
reproducible examples and saved outputs:

- [Single-fidelity tutorial](tutorials/single_fidelity_tutorial.ipynb)
- [Hierarchical multifidelity tutorial](tutorials/hierarchical_multifidelity_tutorial.ipynb)
- [Nonhierarchical multifidelity tutorial](tutorials/non_hierarchical_multifidelity_tutorial.ipynb)

## Learn more

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} Mathematical notation
:link: notation
:link-type: doc

Transport conventions and symbols used throughout MFTT.
:::

:::{grid-item-card} Shared operations
:link: operations
:link-type: doc

Evaluation, inversion, sampling, and density operations.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

Generated reference for the supported public interface.
:::

:::{grid-item-card} Citation
:link: citation
:link-type: doc

Software authorship and the associated manuscript.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

installation
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Foundations

notation
operations
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Methods

single-fidelity
hierarchical
non-hierarchical
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials

tutorials/single_fidelity_tutorial
tutorials/hierarchical_multifidelity_tutorial
tutorials/non_hierarchical_multifidelity_tutorial
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api
citation
```
