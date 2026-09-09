# Notation and mathematical foundations

MFTT constructs monotone triangular maps from samples. This page fixes the
map direction and notation shared by the single-fidelity, hierarchical, and
non-hierarchical methods. The method guides develop the corresponding
parameterizations and training objectives.

## Monotone triangular maps

Let $\pi$ be a target density and $\eta$ a tractable reference density on
$\mathbb R^d$. A lower-triangular map
$\boldsymbol S:\mathbb R^d\to\mathbb R^d$ has components

$$
[\boldsymbol S(\boldsymbol x)]_k
=S_k(x_1,\ldots,x_k),
\qquad k=1,\ldots,d.
$$

Thus, component $k$ depends only on the first $k$ input coordinates. MFTT
uses maps that are strictly increasing in their final input,

$$
\partial_k S_k(\boldsymbol x_{\leq k})
:=\frac{\partial}{\partial x_k}
S_k(x_1,\ldots,x_k)>0.
$$

The Jacobian is lower triangular, so its determinant is the product of its
diagonal entries:

$$
\det\nabla\boldsymbol S(\boldsymbol x)
=\prod_{k=1}^d \partial_kS_k(\boldsymbol x_{\leq k})>0.
$$

This structure makes density evaluation and inversion substantially simpler
than for a general multivariate transformation.

## Target and reference directions

MFTT follows the **target-to-reference** convention. The ideal transport
satisfies

$$
\boldsymbol S_\sharp\pi=\eta.
$$

Equivalently, if $\boldsymbol X\sim\pi$, then
$\boldsymbol S(\boldsymbol X)\sim\eta$. In the library,
`model.evaluate(x)` applies this forward direction. The inverse direction
generates target samples: if $\boldsymbol Z\sim\eta$, then
$\boldsymbol S^{-1}(\boldsymbol Z)\sim\pi$, and `model.inverse(z)` performs
this operation.

The density induced by a map is the pullback of the reference,

$$
(\boldsymbol S^\sharp\eta)(\boldsymbol x)
=\eta(\boldsymbol S(\boldsymbol x))
\left|\det\nabla\boldsymbol S(\boldsymbol x)\right|.
$$

An exact transport therefore obeys

$$
\pi(\boldsymbol x)
=\eta(\boldsymbol S(\boldsymbol x))
\left|\det\nabla\boldsymbol S(\boldsymbol x)\right|.
$$

The library methods `pullback_pdf` and `pullback_logpdf` evaluate the learned
version of this density. See [shared operations](operations.md) for the common
forward, inverse, sampling, and density interfaces.

## The Knothe--Rosenblatt rearrangement

Under standard absolute-continuity assumptions, the monotone triangular map
coupling $\pi$ and $\eta$ is unique up to vairable ordering. It is the
Knothe--Rosenblatt (KR) rearrangement.

Triangularity reduces inversion to a sequence of scalar problems. First solve
$S_1(x_1)=z_1$ for $x_1$, then solve
$S_2(x_1,x_2)=z_2$ for $x_2$, and continue through component $d$. Strict
monotonicity in $x_k$ makes each scalar solution unique when it exists. This
same structure supports exact conditioning on a leading coordinate prefix.

## Sample and fidelity conventions

Training samples are stored as rows. A single-fidelity dataset is

$$
\mathcal X
=\{\boldsymbol x^{(j)}\}_{j=1}^{N},
\qquad \boldsymbol x^{(j)}\sim\pi,
$$

and is supplied as an array with shape `(N, d)`.

For multifidelity problems, $\ell=0,\ldots,M$ indexes the data sources:

$$
\mathcal X_\ell
=\{\boldsymbol x_\ell^{(j)}\}_{j=1}^{N_\ell},
\qquad \boldsymbol x_\ell^{(j)}\sim\pi_\ell.
$$

Fidelity $0$ is always the high-fidelity target, and fidelities
$1,\ldots,M$ are lower-fidelity sources. Multifidelity datasets are supplied
as `[X_0, ..., X_M]`; the method guides state whether the lower-fidelity
sources are ordered or treated as peers.

## Reading the method guides

| Guide | Data assumption | Learned construction |
| --- | --- | --- |
| [Single-fidelity maps](single-fidelity.md) | One sample set from one target | One KR-map approximation |
| [Hierarchical multifidelity maps](hierarchical.md) | An ordered low-to-high fidelity chain | A composition of stage maps |
| [Non-hierarchical multifidelity maps](non-hierarchical.md) | Unordered low-fidelity peers | One high-fidelity map informed by pretrained parent maps |

The single-fidelity guide is also the common background for the component
parameterization and sample-based training used inside both multifidelity
methods.

## Notation conventions

| Mathematical notation | Meaning in MFTT |
| --- | --- |
| $d$ | Input dimension |
| $k=1,\ldots,d$ | Triangular component index |
| $\boldsymbol x$ | Target-space coordinate or sample |
| $\boldsymbol z$ | Reference-space coordinate or sample |
| $\pi$ | Target density |
| $\eta$ | Reference density; standard normal by default |
| $\boldsymbol S$ | Target-to-reference triangular map |
| $\boldsymbol S_\sharp\pi$ | Pushforward of $\pi$ through $\boldsymbol S$ |
| $\boldsymbol S^\sharp\eta$ | Pullback density induced by $\boldsymbol S$ and $\eta$ |
| $\mathcal X$ or `X` | Single-fidelity training matrix with samples as rows |
| $\pi_\ell$ | Target density at fidelity $\ell$ |
| $\mathcal X_\ell$ or `X_ell[ell]` | Training matrix at fidelity $\ell$ |
| $N_\ell$ | Number of samples at fidelity $\ell$ |
| $\ell=0$ | High-fidelity source |
| $\ell>0$ | Lower-fidelity source |
