# MFTT Developer Guide

`mftt` is a measure transport package for single-fidelity and multifidelity monotone triangular maps.

This guide is for someone who wants to:
- use the library through the public API,
- understand how the package is organized,
- extend or refactor one map family without breaking the others,
- connect the core library to downstream research workflows.

## Start Here

For onboarding, use the method guides, tests, and rendered notebooks. The
notebooks under `tutorials/` document the mathematical behavior.

1. [Notation and mathematical foundations](notation.md)
2. [Single-fidelity guide](single-fidelity.md)
3. [Hierarchical multifidelity guide](hierarchical.md)
4. [Non-hierarchical multifidelity guide](non-hierarchical.md)
5. The tests listed in [How To Learn Through Tests](#how-to-learn-through-tests)

## Public API

The package root, [src/mftt/__init__.py](../src/mftt/__init__.py), re-exports the main user-facing classes and diagnostics.

The three core map families are:

- `TriangularMap`
  - single-fidelity transport from a target density to a reference density
- `HierarchicalTriangularMap`
  - stagewise multifidelity composition along a user-specified fidelity hierarchy
- `NonHierarchicalTriangularMap`
  - peer multifidelity transport that approximates the high-fidelity transport as a joint scaling and shifting of pre-trained low-fidelity parent maps.

The primary parameter objects are:

- `MapParams`
- `HierarchicalMapParams`
- `NonHierarchicalMapParams`
- `OptimizationParams`
- `Reference`

Construct the single-fidelity map with one dataset and approximation settings:

```python
map = TriangularMap(train_data, MapParams(total_order=2), reference=Reference.standard_normal(2))
```

The map infers its dimension from `train_data`.  Standardization is runtime
state and, when needed, is supplied as `standardization=...`; it is not part
of `MapParams`.

Across the map families, the public contract is intentionally similar. A contributor should assume these methods and behaviors should stay aligned unless there is a strong reason not to:

- `train(...)`
- `evaluate(...)`
- `inverse(...)`
- `sample(...)`
- `conditional_sample(...)`
- `conditional_pullback_logpdf(...)`
- `conditional_pullback_pdf(...)`
- `log_det(...)`
- `pullback_logpdf(...)`
- `pullback_pdf(...)`
- `pushforward_logpdf(...)`
- `pushforward_pdf(...)`
- `coefficients`

The common API is covered by [tests/test_public_api_symmetry.py](../tests/test_public_api_symmetry.py); numerical behavior is protected by direct mathematical and round-trip tests.

## Conceptual Contract

All three map families implement triangular transports from target-space coordinates to reference-space coordinates.

Shared design principles:

- target data are standardized before model evaluation and training
- evaluation is always on raw user coordinates unless a method explicitly says otherwise
- params objects own configuration, not ad hoc keyword bundles
- training APIs expose modern map-family-specific choices through a small number of explicit arguments
- maps expose density queries through pullback and pushforward helpers
- every family should remain inspectable through readable `__str__` summaries and structured coefficients

When adding a feature, first ask whether it should exist:

- in every family with a common name
- only in one family because the concept is genuinely family-specific
- in diagnostics or a downstream workflow instead of the core map class

## Package Structure

### `single_fidelity/`

This is the base transport implementation and the easiest place to understand the library’s mechanics.

- `map.py`
  - `TriangularMap`
  - standardization, training, evaluation, inversion, conditional sampling, density queries
- `params.py`
  - `MapParams`, `OptimizationParams`, `StandardizationParams`
  - shared component-parameter construction helpers
- `components.py`
  - per-component training logic and caches
- `coefficients.py`
  - structured coefficient containers
- `basis.py`
  - basis specifications and Hermite multi-index machinery
- `reference.py`
  - reference-density abstraction
- `utils.py`
  - shape validation and standardization helpers

### `hierarchical_multifidelity/`

This package composes trained single-fidelity stages over a fidelity hierarchy.

- `map.py`
  - `HierarchicalTriangularMap`
  - stage state, training methods, evaluation order, composed density queries
- `params.py`
  - `HierarchicalMapParams`
- `coefficients.py`
  - hierarchical coefficient wrapper

### `non_hierarchical_multifidelity/`

This package couples a high-fidelity map to low-fidelity parent maps in a peer NHMF design.

- `map.py`
  - `NonHierarchicalTriangularMap`
  - HF-side standardization, parent-map pretraining, coupled component construction
- `params.py`
  - `NonHierarchicalMapParams`
- `components.py`
  - coupled NHMF component training and precomputation
- `parent_terms.py`
  - parent basis evaluation and parent-term semantics
- `coefficients.py`
  - NHMF coefficient wrapper
- `expansion_pair.py`
  - shift, scale, and correction term building blocks

### `diagnostics.py`

Shared post-training diagnostics and distribution-comparison helpers used by tests and downstream workflows.

### `tests/`

Behavioral, numerical, API, and workflow coverage.

## What Lives Where

When you are looking for a change point:

- map-family behavior and orchestration:
  - `map.py`
- configuration and defaults:
  - `params.py`
- basis construction or term ordering:
  - `basis.py` or NHMF `expansion_pair.py`
- training caches and contractions:
  - `components.py`
- coefficient layout or flattening:
  - `coefficients.py`
- parent adaptation logic:
  - NHMF `parent_terms.py`

## Standardization And Precomputation

### Single-Fidelity

- standardization is created in `TriangularMap.__init__`
- standardized training data are cached on the map
- component-level precompute caches are created through `precompute_training_data(...)`

### Hierarchical Multifidelity

- each stage is a `TriangularMap`
- standardization happens inside the stage maps
- changing-reference and fixed-reference differ in how the reference and stage data are built, not in the single-fidelity internals

### Non-Hierarchical Multifidelity

- the canonical coupled coordinates are high-fidelity-standardized coordinates
- each low-fidelity parent map is pretrained on its own locally standardized data
- NHMF hot paths should avoid runtime coordinate-conversion logic
- HF-side coupled precomputation and LF-side standalone precomputation are both important performance features

## Public API Versus Internals

Treat these as public unless there is an explicit reason not to:

- the package-root exports from `mftt`
- the main map classes and params classes
- documented methods used in tutorials, tests, and downstream workflows

Treat these as internal implementation details unless you are deliberately extending internals:

- component cache shapes and contraction details
- internal helper naming in `components.py`, `parent_terms.py`, and coefficient containers
- exact `__str__` formatting beyond the documented substance

If you change a public behavior, update:

- the relevant tutorial notebook
- the corresponding tests

## How To Learn Through Tests

The tests are one of the best ways to learn the codebase. Recommended reading order:

1. [test_public_api_symmetry.py](../tests/test_public_api_symmetry.py)
   - shows the common public contract across map families
2. [test_single_fidelity_core.py](../tests/test_single_fidelity_core.py)
   - base-family behavior and core expectations
3. [test_hierarchical_multifidelity.py](../tests/test_hierarchical_multifidelity.py)
   - stage training choices and hierarchy semantics
4. [test_non_hierarchical_multifidelity.py](../tests/test_non_hierarchical_multifidelity.py)
   - parent-map behavior, corrected versus uncorrected semantics, and NHMF internals
5. [test_single_fidelity_gradients.py](../tests/test_single_fidelity_gradients.py)
   - numerical correctness of the analytic-gradient path
6. [test_single_fidelity_regression.py](../tests/test_single_fidelity_regression.py)
   - regression protection for map behavior

## Contributor Checklist

When making a change:

1. Decide whether it is public API, internal behavior, or downstream workflow behavior.
2. Check whether the same concept exists in the other map families.
3. Update docstrings and readable summaries if the change affects user understanding.
4. Add or update tests at the smallest level that protects the behavior.
5. Update the onboarding notebook or developer guide if the change affects how a new user should think about the package.

## Extension Points

Typical safe extension patterns:

- add diagnostics without changing map-family contracts
- add new downstream scenarios without touching core map internals
- improve readable summaries or tutorial material without changing model semantics
- add a family-specific parameter when the concept is genuinely local to that family

Higher-risk changes:

- changing coefficient layout
- changing basis ordering
- changing standardization semantics
- changing NHMF parent-term meaning
- changing hierarchical stage evaluation order
- changing any method that tutorials and downstream workflows call directly

For those changes, always inspect both family-level tests and workflow tests before and after the edit.
