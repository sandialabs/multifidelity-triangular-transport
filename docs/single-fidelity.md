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

Because $\pi$ is available only through samples, MFTT seeks an approximation
$\widehat{\boldsymbol S}^{\mathrm{SF}}$ by minimizing the divergence from
$\pi$ to the map-induced density:

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

For one sample, split the negative pullback log-density into its reference and
Jacobian contributions,

$$
\mathcal L_j(\boldsymbol\theta)
:=\mathcal L_j^{\mathrm{ref}}(\boldsymbol\theta)
+\mathcal L_j^{\mathrm{Jac}}(\boldsymbol\theta),
$$

where

$$
\mathcal L_j^{\mathrm{ref}}(\boldsymbol\theta)
:=-\log\eta\!\left(
\boldsymbol S(\boldsymbol x^{(j)};\boldsymbol\theta)
\right),
$$

and

$$
\mathcal L_j^{\mathrm{Jac}}(\boldsymbol\theta)
:=-\log\det\nabla\boldsymbol S(
\boldsymbol x^{(j)};\boldsymbol\theta).
$$

The lower-triangular Jacobian gives

$$
\log\det\nabla\boldsymbol S(\boldsymbol x)
=\sum_{k=1}^d\log\partial_kS_k(\boldsymbol x_{\leq k}).
$$

MFTT minimizes the sample average with optional coefficient
regularization,

$$
\min_{\boldsymbol\theta}
\frac{1}{N}\sum_{j=1}^{N}\mathcal L_j(\boldsymbol\theta)
+\lambda\lVert\boldsymbol\theta\rVert_2^2.
$$

In code, `OptimizationParams.reg_cst` is $\lambda$; its default is zero.

### Componentwise Gaussian objective

For the standard-normal reference,
$-\log\eta(\boldsymbol z)=\tfrac12\sum_k z_k^2+C$. The training objective
therefore separates into $d$ component problems,

$$
\min_{\boldsymbol\theta_k}
\frac{1}{N}\sum_{j=1}^{N}
\left[
\frac12 S_k^2(
\boldsymbol x_{\leq k}^{(j)};\boldsymbol\theta_k)
-\log\partial_kS_k(
\boldsymbol x_{\leq k}^{(j)};\boldsymbol\theta_k)
\right]
+\lambda\lVert\boldsymbol\theta_k\rVert_2^2,
\quad k=1,\ldots,d.
$$

`TriangularMap.train()` implements this standard-Gaussian objective, training
one component at a time. The problems are mathematically independent even
though the implementation visits them in component order. Componentwise
training is usually preferable because it replaces one large optimization
with several smaller ones.

`TriangularMap.train_aao()` instead optimizes the full, all-at-once objective.
It evaluates `model.reference` directly and uses its score for analytic
gradients. This is the appropriate training path for a general, possibly
nonfactorized reference, and it is also used when a multifidelity stage has a
map-induced reference. The reference must therefore provide both a log
density and a score.

## Monotone component parameterization

Each component must remain increasing in its final coordinate throughout
optimization. MFTT enforces this constraint with an integrated rectifier:

$$
S_k(\boldsymbol x_{\leq k};\boldsymbol\theta_k)=f_k(\boldsymbol x_{<k},0;\boldsymbol\theta_k)+\int_0^{x_k}g\!\left(\partial_k f_k(\boldsymbol x_{<k},t;\boldsymbol\theta_k)\right)\,\mathrm dt.
$$

Here, $f_k:\mathbb R^k\to\mathbb R$ is a finite expansion and
$g:\mathbb R\to(0,\infty)$ is a positive rectifier. Differentiating with
respect to the final coordinate gives

$$
\partial_kS_k(\boldsymbol x_{\leq k};\boldsymbol\theta_k)
=g\!\left(
\partial_kf_k(\boldsymbol x_{\leq k};\boldsymbol\theta_k)
\right)>0,
$$

so every coefficient vector represents a monotone component. In the
implementation, $f_k$ is a multivariate Hermite-function expansion, $g$ is
SoftPlus, and `rectifier_epsilon` adds a small positive floor to the
derivative. The integral is approximated by the component's Gauss--Legendre
`QuadratureRule`.

The nonmonotone term uses expansion terms that are constant in $x_k$. Terms
that depend on $x_k$ contribute through the rectified derivative and its
integral. Both parts share the coefficient block exposed by
`model.coefficients.components[k - 1]`.

The objective is convex over suitable infinite-dimensional map spaces when
the reference is log-concave. The finite integrated-rectifier
parameterization is nonlinear in its coefficients, however, so the numerical
optimization is generally nonconvex. Initialization, basis size,
regularization, and convergence diagnostics still matter.

## Choosing component bases

The simplest configuration uses one total Hermite order for every component:

```python
params = MapParams(total_order=2)
```

When this configuration is bound to $d$-dimensional data, component $k$ gets
all $k$-dimensional multi-indices $\boldsymbol\alpha$ with
$|\boldsymbol\alpha|\leq2$. Thus the number of terms grows with both $k$ and
the selected order.

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
| Hermite expansion $f_k$ | `model.components[k - 1].expansion` |
| Component coefficients $\boldsymbol\theta_k$ | `model.coefficients.components[k - 1]` |
| Flattened $\boldsymbol\theta$ | `model.coeffs` or `model.coefficients.flatten()` |
| Total-order or explicit basis | `MapParams` and each component's `BasisSpec` |
| Positive rectifier and quadrature | Each component's `rectifier` and `QuadratureRule` |
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
the standardization internally, while `inverse` and sampling undo it. Density
and `log_det` evaluations include the standardization Jacobian.

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
complete executable banana example with plots, or consult the
[single-fidelity API reference](api/maps/triangular-map.md) for all methods.
