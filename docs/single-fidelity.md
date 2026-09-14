# Single-fidelity maps

Single-fidelity transport learns one monotone triangular map from samples of
one target distribution. It is both a useful model on its own and the basic
building block used by the multifidelity methods. This page develops the
sample-based objective, monotone parameterization, and their exact
representation in `TriangularMap`.

The common target-to-reference convention, pullback density, and KR
rearrangement are introduced in the [notation guide](notation.md). Unless
stated otherwise, this page uses the default reference
$\eta=\mathcal N(0,I)$.

## Learning a map from samples

Let

$$
\mathcal X
=\{\boldsymbol x^{(j)}\}_{j=1}^{N},
\qquad \boldsymbol x^{(j)}\sim\pi,
$$

be independent target samples in $\mathbb R^d$. The ideal KR map
$\boldsymbol S$ satisfies

$$
\boldsymbol S_\sharp\pi=\eta.
$$

We denote by $\mathcal F_\triangle$ the set of
monotone lower-triangular maps:

$$
\mathcal F_\triangle
:=\left\{
\boldsymbol S:\mathbb R^d\to\mathbb R^d
\;\middle|\;
[\boldsymbol S(\boldsymbol x)]_k=S_k(\boldsymbol x_{\leq k}),\quad
\partial_kS_k(\boldsymbol x_{\leq k})>0,\quad k=1,\ldots,d
\right\}.
$$

The KR map can be characterized by minimizing the divergence from $\pi$ to
the map-induced density {footcite:p}`marzouk2016sampling`:

$$
\boldsymbol S^*
\in\arg\min_{\boldsymbol S\in\mathcal F_\triangle}
\mathcal D_{\mathrm{KL}}\!\left(
\pi\,\middle\|\,\boldsymbol S^\sharp\eta
\right).
$$

After dropping terms independent of $\boldsymbol S$, this is equivalent to

$$
\min_{\boldsymbol S\in\mathcal F_\triangle}
\mathbb E_\pi\!\left[
-\log\eta(\boldsymbol S(\boldsymbol x))
-\log\det\nabla\boldsymbol S(\boldsymbol x)
\right].
$$

Because $\pi$ is available only through $\mathcal X$, the expectation is
replaced by its sample-average approximation, which can also be understood as
maximum-likelihood estimation {footcite:p}`wang2022minimax`:

$$
\min_{\boldsymbol S\in\mathcal F_\triangle}
\frac{1}{N}\sum_{j=1}^{N}
\left[
-\log\eta\!\left(\boldsymbol S(\boldsymbol x^{(j)})\right)
-\log\det\nabla\boldsymbol S(\boldsymbol x^{(j)})
\right].
$$

The lower-triangular Jacobian gives

$$
\log\det\nabla\boldsymbol S(\boldsymbol x)
=\sum_{k=1}^d\log\partial_kS_k(\boldsymbol x_{\leq k}).
$$

### Componentwise Gaussian objective

For the standard-normal reference,
$-\log\eta(\boldsymbol z)=\tfrac12\sum_k z_k^2+C$. The training objective
therefore separates into $d$ component problems,

$$
\min_{S_k:\mathbb R^k\to\mathbb R,\;\partial_kS_k>0}
\frac{1}{N}\sum_{j=1}^{N}
\left[
\frac12 S_k^2(\boldsymbol x_{\leq k}^{(j)})
-\log\partial_kS_k(\boldsymbol x_{\leq k}^{(j)})
\right]
,\qquad k=1,\ldots,d.
$$

These component problems are mathematically independent. A concrete
coefficient parameterization is introduced next.

## Monotone component parameterization

Each component must remain increasing in its final coordinate throughout
optimization. MFTT enforces this through an integrated rectifier
parameterization {footcite:p}`baptista2024representation`. Let
$\boldsymbol f=(f_1,\ldots,f_d)$ be lower triangular and write
$\boldsymbol S=\mathcal R(\boldsymbol f)$ for the integrated-rectifier
operator. It acts componentwise as

$$
S_k(\boldsymbol x_{\leq k})
=\mathcal R_k(f_k)(\boldsymbol x_{\leq k})
=f_k(\boldsymbol x_{\leq k-1},0)
+\int_0^{x_k}g\!\left(
\partial_k f_k(\boldsymbol x_{\leq k-1},t)
\right)\,\mathrm dt.
$$

Here, $f_k:\mathbb R^k\to\mathbb R$, while
$g:\mathbb R\to(0,\infty)$ is a positive, bijective rectifier. The manuscript
uses the SoftPlus rectifier
{footcite:p}`baptista2024representation,ramgraber2025friendly`:

$$
g(u)=\operatorname{SoftPlus}(u)=\log(1+\exp(u)).
$$

Differentiating with respect to the final coordinate gives

$$
\partial_kS_k(\boldsymbol x_{\leq k})
=g\!\left(
\partial_kf_k(\boldsymbol x_{\leq k})
\right)>0,
$$

so $\mathcal R(\boldsymbol f)\in\mathcal F_\triangle$ for every admissible
$\boldsymbol f$. The implementation evaluates
$g_\epsilon(u)=\operatorname{SoftPlus}(u)+\epsilon$, where
`rectifier_epsilon` is the small positive floor $\epsilon$ used for numerical stability, and approximates
the integral with the component's Gauss--Legendre `QuadratureRule`.

The nonmonotone term uses expansion terms that are constant in $x_k$. Terms
that depend on $x_k$ contribute through the rectified derivative and its
integral. Both parts share the coefficient block exposed by
`model.coefficients.components[k - 1]`.

## Choosing component bases

The simplest configuration uses one total Hermite order for every component:

```python
params = MapParams(total_order=2)
```

Following the manuscript, MFTT restricts $f_k$ to a finite-dimensional space
$V_k^p$ and writes

$$
f_k(\boldsymbol x_{\leq k};\boldsymbol\theta_k)
=\sum_{\boldsymbol\alpha\in\Lambda_k(p)}
c_{\boldsymbol\alpha}\Phi_{\boldsymbol\alpha}(\boldsymbol x_{\leq k}),
\qquad
\boldsymbol\theta_k
=\{c_{\boldsymbol\alpha}:\boldsymbol\alpha\in\Lambda_k(p)\},
$$

where $\Phi_{\boldsymbol\alpha}$ is a tensor-product basis function
{footcite:p}`ernst2012convergence,lemaitre2010spectral`. The total-order index
set is

$$
\Lambda_k(p)
=\left\{\boldsymbol\alpha\in\mathbb N_0^k:
|\boldsymbol\alpha|_1=\sum_{r=1}^k\alpha_r\leq p\right\}.
$$

MFTT uses ordinary probabilists'
Hermite polynomials for constant and linear factors and damped Hermite
functions for factors of order two and higher
{footcite:p}`ramgraber2025friendly`.

After selecting these finite-dimensional spaces, write
$\boldsymbol S(\cdot;\boldsymbol\theta)
=\mathcal R(\boldsymbol f(\cdot;\boldsymbol\theta))$. MFTT then solves the
coefficient-level version of the sample-average problem, with optional
regularization:

$$
\min_{\boldsymbol\theta}
\frac{1}{N}\sum_{j=1}^{N}
\left[
-\log\eta\!\left(
\boldsymbol S(\boldsymbol x^{(j)};\boldsymbol\theta)
\right)
-\log\det\nabla\boldsymbol S(
\boldsymbol x^{(j)};\boldsymbol\theta)
\right]
+\lambda\lVert\boldsymbol\theta\rVert_2^2.
$$

In code, `OptimizationParams.reg_cst` is $\lambda$; its default is zero.
For the standard-normal reference, `TriangularMap.train()` solves the
separated component problems one at a time. `TriangularMap.train_aao()`
instead solves the full coefficient problem and evaluates `model.reference`
and its score directly. The all-at-once path supports a general, possibly
nonfactorized reference and is also used when a multifidelity stage has a
map-induced reference.

As an alternative to the total order construction, one can specify a custom multi-index set for each component:

```python
model = TriangularMap.from_multi_index_sets(
    train_data=X,
    multi_index_sets=[
        [(0,), (1,), (2,)],
        [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1)],
    ],
)
```

The $k$th set must contain multi-indices of length $k$. MFTT
initializes maps as the identity in standardized coordinates.

## From the equations to MFTT

The following example constructs a nonlinear two-dimensional target, trains a
single-fidelity map, and exercises its main public operations:

```python
import numpy as np

from mftt import MapParams, OptimizationParams, TriangularMap

rng = np.random.default_rng(2026)
u, v = rng.standard_normal((2, 600))
X = np.column_stack((u, v + 0.5 * (u**2 - 1.0)))

params = MapParams(
    total_order=2,
    optimization=OptimizationParams(
        optimizer="BFGS", reg_cst=1e-6, gtol=1e-6, maxiter=200
    ),
)
model = TriangularMap(X, params)
model.train()

z = model.evaluate(X[:32])
x_recovered = model.inverse(z)
x_draws = model.sample(1000, random_state=7)
log_density = model.pullback_logpdf(X[:32])

print(np.max(np.abs(x_recovered - X[:32])))
print(model.coefficients.shapes())
print(model.optimization_diagnostics())
```

`evaluate` accepts raw target coordinates and returns reference coordinates.
`inverse` reverses that transformation, `sample` draws standard-normal
reference values and inverts them, and `pullback_logpdf` evaluates the density
induced by the learned map.

| Mathematical object | Library representation |
| --- | --- |
| $\mathcal X=\{\boldsymbol x^{(j)}\}_{j=1}^N$ | `model.train_data` |
| Standardized training samples | `model.standardized_train_data` |
| $\eta$ | `model.reference`, configured with `Reference` |
| $S_k$ | `model.components[k - 1]` |
| $f_k\in V_k^p$ with basis $\{\Phi_{\boldsymbol\alpha}\}$ | `model.components[k - 1].expansion` |
| Component coefficients $\boldsymbol\theta_k$ | `model.coefficients.components[k - 1]` |
| Flattened $\boldsymbol\theta$ | `model.coeffs` or `model.coefficients.flatten()` |
| Total-order or explicit basis | `MapParams` and each component's `BasisSpec` |
| $\mathcal R_k$, SoftPlus $g$, and quadrature | Each component's `rectifier` and `QuadratureRule` |
| $\lambda$ | `OptimizationParams.reg_cst` |
| $\widehat{\boldsymbol S}^{\mathrm{SF}}$ | `model.evaluate(x)`, with `inverse(z)` for its inverse |

### Internal standardization

`TriangularMap` computes a mean and standard deviation for every coordinate of
`train_data` and trains in the standardized coordinates

$$
\widetilde{\boldsymbol x}
=\frac{\boldsymbol x-\boldsymbol\mu}{\boldsymbol\sigma}.
$$

These values are fitted map state in `model.standardization_params`.
Public methods accept and return raw target coordinates: `evaluate` applies
the standardization internally, while `inverse`, sampling, and density evaluation undo it.

### References and training paths

The default constructor creates `Reference.standard_normal(d)`. With this
reference, `train()` uses the separated component objectives and `sample()`
and `conditional_sample()` draw from the correct reference automatically.

A custom reference can be created with `Reference(logpdf_fn, score_fn)` or,
for a compatible frozen SciPy distribution, `Reference.from_scipy(...)`.
Use `train_aao()` for such a reference; analytic all-at-once training requires
`score_fn`. `pullback_pdf` and `pullback_logpdf` use the configured reference
density.

The current `sample()` and `conditional_sample()` convenience methods always
draw standard-normal reference coordinates. For a nonstandard reference,
draw $\boldsymbol z\sim\eta$ yourself and call `model.inverse(z)`. The
conditional convenience is likewise specific to the standard-normal
reference path.

### Inspection and next steps

After training:

- `components[k - 1]` exposes component $k$, its basis, rectifier, and most
  recent component optimizer result.
- `coefficients` exposes structured per-component blocks; `coeffs` is the
  flattened optimizer view.
- `optimization_diagnostics()` returns separate component summaries and the
  all-at-once summary, if that path was used.
- `log_det(x)` evaluates the raw-coordinate log Jacobian determinant.

Use held-out target samples with the functions in `mftt.diagnostics` to check
whether `evaluate` produces approximately Gaussian reference samples.
Optimizer success alone does not establish an accurate transport.

For inversion controls, prefix-conditional operations, density queries, and
diagnostic return types, see [shared operations](operations.md). Continue to
the [single-fidelity notebook](tutorials/single_fidelity_tutorial.ipynb) for a
complete executable example, or consult the
[single-fidelity API reference](api/maps/triangular-map.md) for all methods.

## References

```{footbibliography}
```
