# Shared operations

All three map families share the following target/reference convention:

- `evaluate(x)` maps target coordinates to reference coordinates.
- `inverse(z)` maps reference coordinates back to target coordinates.
- `sample(n)` draws target samples by drawing from the reference and inverting.
- `pullback_pdf(x)` and `pullback_logpdf(x)` evaluate the learned target density.
- `pushforward_pdf(z, target)` and `pushforward_logpdf(z, target)` evaluate a supplied target after mapping it to reference coordinates.
- `log_det(x)` evaluates the log absolute Jacobian determinant.

`conditional_sample` and `conditional_pullback_*` operate on leading fixed
indices, for example `fixed_indices=[0, 1]`. Inputs are batches with shape
`(n_samples, d)`; a single point should be reshaped to `(1, d)`.

Inverse calls support `InverseOptions` for bracket, evaluation, and time
limits. Set `return_diagnostics=True` to receive an `InverseResult` containing
values and structured failure diagnostics instead of only an array.

The `diagnostics` functions compare mapped samples with Gaussian reference
samples using moment errors, MMD, and Forstner distance. They are evaluation
helpers, not training objectives.
