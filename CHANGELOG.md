# Changelog

## [0.2.0] - 2026-11-10

### Breaking changes

- Require Rust 1.85 or later and use the Rust 2024 edition.
- Accept generic chain slices in diagnostic functions. Existing direct calls
  with `Vec<Vec<f64>>` continue to work; explicit function pointers now take
  `&[Vec<f64>]`, and empty inputs or ambiguous `.collect()` calls may need
  type annotations.
- Return errors for invalid or undefined estimates, including negative
  truncated autocorrelation times and unrepresentable MCSE results.

### Added

- Borrowed slices and native arrays of `f32` or `f64` as diagnostic inputs,
  with calculations and results in `f64`.
- Borrowed split-chain views that preserve sample buffers and element types.
- Optional `ndarray` adapters for contiguous, column-major, transposed,
  reversed, and strided views, with chains on axis 0 and draws on axis 1.
- Optional stable-Rust SIMD autocovariance through the `simd` feature, plus
  an ESS/Rhat benchmark. The default build uses the scalar kernel.
- Five external draw sets covering 15 selected parameters, checked against
  pinned R and `posterior` references. Include source hashes, license
  attribution, reproducible generation, and documented estimator differences.

### Fixed

- Reject empty or insufficient chain inputs without panicking, and trim all
  calculations consistently to the shortest retained chain.
- Preserve variation at tiny scales and large offsets by centering and scaling
  calculations, with exact constant detection.
- Preserve very small within-chain variation for Rhat by combining locally
  scaled standard deviations without squaring their original magnitudes.
- Avoid intermediate overflow and underflow when computing autocovariance,
  Rhat, and MCSE. Restore MCSE units after taking square roots.
- Compute MCSE variance without allocating a flattened sample copy.

### Development

- Refresh dependencies and move the general-purpose `arima` dependency to
  test-only reference use.
- Test Rust 1.85 and stable across all optional-feature combinations, with
  formatting, Clippy, documentation, and external-fixture integrity checks.
- Document the public API, compatibility changes, validation corpus, and
  remaining diagnostic and performance work.

[0.2.0]: https://github.com/isms/mcmc-rs/compare/v0.1.3...v0.2.0
