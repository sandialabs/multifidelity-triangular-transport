# Non-hierarchical multifidelity maps

Non-hierarchical multifidelity transport uses low-fidelity models as **peer
information sources** rather than arranging them in a hierarchy. It first
learns a transport for every low-fidelity density, then uses all of those maps
inside one monotone high-fidelity transport. This page develops the
parameterization and its two-phase training procedure before connecting each
mathematical object to `NonHierarchicalTriangularMap`.

The target-to-reference convention is reviewed in the
[notation guide](notation.md). The [single-fidelity guide](single-fidelity.md)
develops the integrated-rectifier construction and componentwise parent-map
training used below. The current non-hierarchical implementation uses the
standard-normal reference $\eta=\mathcal N(0,I)$.

## Peer multifidelity data

Let $\pi_0$ be the high-fidelity density and let
$\pi_1,\ldots,\pi_M$ be lower-fidelity densities. At each level,

$$
\mathcal X_\ell
=\{\boldsymbol x_\ell^{(j)}\}_{j=1}^{N_\ell},
\qquad
\boldsymbol x_\ell^{(j)}\sim\pi_\ell.
$$

Supply the datasets as `[X_0, X_1, ..., X_M]`. The first entry is always high
fidelity, but the remaining entries are unordered peers: permuting
`X_1, ..., X_M` together with their configurations does not change the model
assumption. The datasets need not be paired and may have different sample
counts, but all must have the same coordinate dimension.

The central assumption is representational: at least some low-fidelity KR
maps contain structure useful for approximating the high-fidelity KR map
$\boldsymbol S_0$. Unlike a hierarchical method, the non-hierarchical method
does not compose maps between densities. Instead, it combines corrected and
scaled pieces of all low-fidelity maps inside each component of a single
high-fidelity transport.

The complete training flow is:

| Step | Data used | Result |
| --- | --- | --- |
| 1. Pretrain parents | Each $\mathcal X_\ell$, $\ell=1,\ldots,M$, independently | Low-fidelity maps $\widehat{\boldsymbol S}_\ell$ |
| 2. Train coupled components | $\mathcal X_0$ and all lower-fidelity datasets | Final $\widehat{\boldsymbol S}^{\mathrm{NH}}$ |

## Low-fidelity parent maps

For each peer density, first seek a monotone triangular map

$$
(\boldsymbol S_\ell)_\sharp\pi_\ell=\eta,
\qquad \ell=1,\ldots,M.
$$

Its learned component has the usual integrated-rectifier form

$$
\widehat S_{\ell,k}(\boldsymbol x_{\leq k};\boldsymbol\theta_{\ell,k}^*)=f_{\ell,k}(\boldsymbol x_{<k},0;\boldsymbol\theta_{\ell,k}^*)+\int_0^{x_k}g\!\left(\partial_k f_{\ell,k}(\boldsymbol x_{<k},t;\boldsymbol\theta_{\ell,k}^*)\right)\,\mathrm dt,
$$

where $g:\mathbb R\to(0,\infty)$ is the positive rectifier. The parameters
$\boldsymbol\theta_{\ell,k}^*$ are obtained during pretraining.

In the manuscript-aligned construction, these pretrained parameters are then
frozen. A trainable function
$c_{\ell,k}(\boldsymbol x_{\leq k};
\boldsymbol\theta^{\mathrm{corr}}_{\ell,k})$ is added to the corresponding
pre-rectifier function. In compact notation,

$$
\widetilde f_{\ell,k}
:=f_{\ell,k}(\,\cdot\,;\boldsymbol\theta_{ell,k}^*)
+c_{\ell,k}(\,\cdot\,;
\boldsymbol\theta^{\mathrm{corr}}_{\ell,k}).
$$

When the two functions share a basis, this is the manuscript expression
$f_{\ell,k}(\,\cdot\,;
\boldsymbol\theta_{ell,k}^*+
\boldsymbol\theta^{\mathrm{corr}}_{ell,k})$. MFTT allows the correction to
have its own total order, so the additive-function view is more general. The
corrected parent remains monotone because
$\partial_k\widetilde f_{\ell,k}$ is passed through $g$ before integration.

## The non-hierarchical component

For component $k$, the manuscript combines a high-fidelity shift with separate
scales for the nonmonotone and monotone contributions of every peer:

$$
S_k^{\mathrm{NH}}(\boldsymbol x_{\leq k};\boldsymbol\Theta_k)=\delta_k(\boldsymbol x_{<k},0;\boldsymbol\vartheta_k)+\sum_{\ell=1}^M\rho^{\mathrm{nm}}_{\ell,k}\widetilde f_{\ell,k}(\boldsymbol x_{<k},0)+\int_0^{x_k}g\!\biggl(\partial_k\delta_k(\boldsymbol x_{<k},t;\boldsymbol\vartheta_k)+\sum_{\ell=1}^M\rho^{\mathrm m}_{\ell,k}\partial_k\widetilde f_{\ell,k}(\boldsymbol x_{<k},t)\biggr)\,\mathrm dt.
$$

The parameter block is

$$
\boldsymbol\Theta_k
=\left(
\boldsymbol\vartheta_k,
\{\rho^{\mathrm{nm}}_{\ell,k},
\rho^{\mathrm m}_{\ell,k},
\boldsymbol\theta^{\mathrm{corr}}_{\ell,k}\}_{\ell=1}^M
\right).
$$

The roles of these terms are distinct:

- $\delta_k$ learns high-fidelity structure not supplied by the parents.
- $\rho^{\mathrm{nm}}_{\ell,k}$ scales the parent contribution at
  $x_k=0$.
- $\rho^{\mathrm m}_{\ell,k}$ scales the parent contribution to the
  pre-monotone derivative.
- $\boldsymbol\theta^{\mathrm{corr}}_{\ell,k}$ adapts pretrained parent
  structure using both high- and low-fidelity data.

The scales are unconstrained and can amplify, reverse, or suppress a peer's
contribution. Monotonicity does not require positive scales: the entire
pre-monotone combination is placed inside $g$, so

$$
\partial_k S_k^{\mathrm{NH}}(\boldsymbol x_{\leq k})
=g\!\left(
\partial_k\delta_k
+\sum_{\ell=1}^M
\rho^{\mathrm m}_{\ell,k}\partial_k\widetilde f_{\ell,k}
\right)>0.
$$

### Scale functions in MFTT

With `scale_order=0`, MFTT implements the manuscript's constant
$\rho^{\mathrm{nm}}_{\ell,k}$ and $\rho^{\mathrm m}_{\ell,k}$. The two
constants are separately trainable. With `scale_order > 0`, each scalar is
generalized to a Hermite expansion evaluated on
$\boldsymbol x_{\leq k}$; this permits a parent's influence to vary over the
input space while retaining the same final rectifier and monotonicity
guarantee.

`shift_order`, `scale_order`, and `correction_order` independently control the
three function families. An order of zero still supplies the constant and
final-coordinate linear seed required by the integrated parameterization. In
particular, `correction_order=0` does not disable corrections; use
`train(use_corrections=False)` to remove the separate correction blocks.

## Two-phase training

For any map $\boldsymbol T$, define its negative pullback log-density at
$\boldsymbol x$ as

$$
\mathcal L(\boldsymbol T,\boldsymbol x)
:=-\log\eta(\boldsymbol T(\boldsymbol x))
-\log\det\nabla\boldsymbol T(\boldsymbol x).
$$

### Phase 1: independent parent pretraining

For every $\ell=1,\ldots,M$, MFTT trains
$\widehat{\boldsymbol S}_\ell$ on $\mathcal X_\ell$ by minimizing the
empirical KL objective

$$
\mathcal J_\ell^{\mathrm{pre}}(\boldsymbol\theta_\ell)=\frac{1}{N_\ell}\sum_{j=1}^{N_\ell}\mathcal L(\widehat{\boldsymbol S}_\ell,\boldsymbol x_\ell^{(j)})+\lambda_\ell\lVert\boldsymbol\theta_\ell\rVert_2^2.
$$

The dependence of $\widehat{\boldsymbol S}_\ell$ on
$\boldsymbol\theta_\ell$ is implicit in this compact expression.

Each parent uses its own `MapParams`, including its basis and optional
`OptimizationParams.reg_cst`. The parent problems are mathematically
independent. Because $\eta$ is factorized Gaussian, each parent also trains
componentwise using the standard `TriangularMap.train` path. An
`optimization=...` argument passed to `model.train` overrides only the coupled
Phase 2 optimizer; parent pretraining still uses each parent's `MapParams`.
See the [single-fidelity training discussion](single-fidelity.md#componentwise-gaussian-objective)
for the separated Gaussian objective.

### Phase 2: coupled multifidelity training

With `use_corrections=True`, the pretrained parent coefficients are fixed and
the coupled problem optimizes the shifts, peer scales, and correction
parameters. Define

$$
w_\ell:=\frac{N_\ell}{\sum_{r=1}^M N_r},
\qquad \ell=1,\ldots,M.
$$

Using the same loss, the joint objective is

$$
\mathcal J(\boldsymbol\Theta)=\frac{\omega_0}{N_0}\sum_{j=1}^{N_0}\mathcal L(\boldsymbol S^{\mathrm{NH}},\boldsymbol x_0^{(j)})+\sum_{\ell=1}^M\frac{w_\ell}{N_\ell}\sum_{j=1}^{N_\ell}\mathcal L(\widehat{\boldsymbol S}^{\mathrm{corr}}_\ell,\boldsymbol x_\ell^{(j)})+\lambda\lVert\boldsymbol\Theta\rVert_2^2.
$$

The dependence of both maps on $\boldsymbol\Theta$ is implicit in this
compact expression.

Here, `hf_weight` is $\omega_0$ and defaults to one, while the low-fidelity
weights $w_\ell$ are computed automatically from sample counts. The
high-fidelity term trains the final map; the lower-fidelity terms keep each
corrected parent supported by its abundant source data and therefore act as a
multifidelity regularizer. `OptimizationParams.reg_cst` is $\lambda$.
Setting `regularize_parent_terms=False` excludes the parent/correction blocks
from this quadratic penalty while continuing to regularize shifts and scales.

Although the manuscript calls Phase 2 all-at-once training, a factorized
Gaussian reference makes the objective separable across triangular components.
MFTT therefore trains $k=1,\ldots,d$ in sequence. Within each component, the
shift, every peer scale, and every correction are optimized jointly against
the high- and low-fidelity terms.


## From the equations to MFTT

Assume `X_0`, `X_1`, and `X_2` are two-dimensional NumPy arrays whose rows are
samples. The parent configurations correspond to $\pi_1$ and $\pi_2$; no
`MapParams` is supplied for the high-fidelity dataset because its transport is
the coupled NH map.

```python
from mftt import (
    MapParams,
    NonHierarchicalMapParams,
    NonHierarchicalTriangularMap,
    OptimizationParams,
)

X_ell = [X_0, X_1, X_2]
optimization = OptimizationParams(
    optimizer="BFGS", reg_cst=1e-8, gtol=1e-6, maxiter=200
)
nhmf_params = NonHierarchicalMapParams(
    low_fidelity_map_params=[
        MapParams(total_order=2, optimization=optimization)
        for _ in X_ell[1:]
    ],
    shift_order=2,
    scale_order=0,       # manuscript's constant rho_nm and rho_m
    correction_order=2,
    optimization=optimization,
    hf_weight=1.0,
)

model = NonHierarchicalTriangularMap(X_ell, nhmf_params)
model.train(use_corrections=True)

z = model.evaluate(X_0[:32])
x_recovered = model.inverse(z)
x_draws = model.sample(1000)

print(len(model.low_fidelity_maps))       # 2 pretrained parents
print(model.coefficients.shapes())        # shift/scale/parent blocks
```

| Mathematical object | Library representation |
| --- | --- |
| $\mathcal X_0$ | `model.train_data[0]` and `model.standardized_train_data` |
| $\mathcal X_\ell$, $\ell\geq1$ | `model.train_data[ell]` |
| $\widehat{\boldsymbol S}_\ell$ | `model.low_fidelity_maps[ell - 1]` |
| $\boldsymbol\delta_k$ and $\boldsymbol\vartheta_k$ | `model.coefficients.components[k - 1].shift` |
| $\rho^{\mathrm{nm}}_{\ell,k}$ and $\rho^{\mathrm m}_{\ell,k}$ | `model.coefficients.components[k - 1].scales[ell - 1]` |
| $\boldsymbol\theta^{\mathrm{corr}}_{\ell,k}$ | `model.coefficients.components[k - 1].parents[ell - 1]` when corrections are enabled |
| $w_\ell$ | `model.components[k - 1].low_fidelity_weights()` |
| $\omega_0$ | `NonHierarchicalMapParams.hf_weight` |
| $\lambda$ | `NonHierarchicalMapParams.optimization.reg_cst` |
| $\widehat{\boldsymbol S}^{\mathrm{NH}}$ | `model.evaluate(x)`, with `inverse(z)` and `sample(n)` for its inverse |
| $\boldsymbol\Theta$ | `model.coefficients`, or `model.coefficients.flatten()` for the optimizer view |

### Internal standardization and inspection

The high-fidelity dataset and every low-fidelity dataset are standardized
independently. Parent pretraining occurs in each parent's local standardized
coordinates. During coupled training, the parent component functions and corrections
are evaluated on high-fidelity-standardized inputs to construct
$\boldsymbol S^{\mathrm{NH}}$, while each corrected-parent loss is evaluated
on its own parent-standardized data. Public evaluation, inversion, sampling,
and density methods accept or return raw high-fidelity coordinates and account
for the high-fidelity standardization and its Jacobian.

After training:

- `low_fidelity_maps[ell - 1]` exposes parent $\ell$ and its pretraining
  diagnostics.
- `components[k - 1]` exposes the coupled component, its source weights, and
  `scale_constant_optimization_result`.
- `coefficients.components[k - 1]` groups the trainable `shift`, `scales`, and
  `parents` blocks; `coefficients.shapes()` summarizes their structure.
- `low_fidelity_weights()` returns the same $w_\ell$ for every component,
  determined solely by parent sample counts.

### Corrections and the implementation ablation

`train(use_corrections=True)` is the manuscript-aligned default. It preserves
the pretrained coefficients in `low_fidelity_maps` and learns separate
additive correction blocks. The corrected maps are terms in the coupled model;
`low_fidelity_maps` themselves remain the frozen pretrained baselines.

`train(use_corrections=False)` removes the separate correction expansions. In
the current implementation, the coupled objective then directly fine-tunes
the pretrained parent coefficients alongside the shifts and scales. It is an
ablation of the correction parameterization, not a mode with frozen,
uncorrected parent maps.


For a complete executable banana example with pushforward, sampling,
conditional, and density diagnostics, continue to the
[non-hierarchical notebook](tutorials/non_hierarchical_multifidelity_tutorial.ipynb).
See the
[non-hierarchical API reference](api/maps/non-hierarchical-triangular-map.md)
for all methods and [shared operations](operations.md) for inversion,
conditional sampling, and density evaluation.
