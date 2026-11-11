MCMC diagnostics in Rust
====

[![Crates.io][crates-badge]][crates-url]
[![docs.rs][docs-badge]][docs-url]
[![MIT licensed][mit-badge]][mit-url]

[crates-badge]: https://img.shields.io/crates/v/mcmc.svg
[crates-url]: https://crates.io/crates/mcmc
[docs-badge]: https://docs.rs/mcmc/badge.svg
[docs-url]: https://docs.rs/mcmc
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: LICENSE

A Rust library implementing various MCMC diagnostics and utilities, such as Gelman Rubin
potential scale reduction factor (R hat), effective sample size (ESS), chain splitting,
and others.

This crate is language agnostic and intended to work with the outputs of any MCMC sampler
(e.g. Stan, PyMC, Turing.jl, etc).

Development
-----------

Requires Rust 1.85 or later and uses the Rust 2024 edition. CI tests both the
minimum supported Rust version and current stable Rust.

```sh
cargo test --all-targets
cargo test --doc
cargo test --all-targets --all-features
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
python3 scripts/validation/generate.py --check
```

Implementation
--------------

The public API accepts vectors, borrowed slices, and native arrays of `f32` or
`f64` draws, grouped by chain. Calculations and results use `f64`. The `Array1`
and `Array2` aliases remain `Vec<f64>` and `Vec<Vec<f64>>` for existing callers.
Diagnostics trim each chain to the length of the shortest chain. Splitting drops
the middle draw when that common length is odd.

```rust
let draws = [[1.0_f32, 3.0, 2.0, 4.0], [2.0, 4.0, 1.0, 3.0]];
let ess = mcmc::ess::compute_effective_sample_size(&draws).unwrap();
let halves = mcmc::utils::split_chain_views(&draws).unwrap();
```

`f64` input buffers are borrowed; `f32` inputs are widened once per retained
chain. `split_chain_views` returns borrowed halves without copying draws.
The existing consuming `split_chains` function still returns owned vectors.
Existing direct vector calls continue to work. With generic inputs, an untyped
empty input or an ambiguous `.collect()` needs a type annotation such as `Array2`.
Explicit diagnostic function pointers now take `&[Vec<f64>]` instead of `&Array2`.

Enable the optional `ndarray` feature for `mcmc::ndarray` adapters. These accept
`ndarray::ArrayView2` with **chains on axis 0 and draws on axis 1**:

```rust
let draws = ndarray::array![[1.0_f64, 3.0, 2.0, 4.0], [2.0, 4.0, 1.0, 3.0]];
let rhat = mcmc::ndarray::potential_scale_reduction_factor(draws.view()).unwrap();
let halves = mcmc::ndarray::split_chain_views(draws.view()).unwrap();
```

Contiguous `f64` rows are borrowed. Column-major, transposed, reversed, and
strided views work too; noncontiguous rows are copied in their logical order.
`f32` rows are converted to contiguous `f64` buffers.
The ndarray `split_chain_views` adapter borrows the original samples and
preserves strides and element type, including for noncontiguous inputs.

Implementations for some of these diagnostics vary slightly, so reference implementations
are based on Stan 2.24.0, and unit tests adapted from that version check matching
behavior. Rhat and split Rhat currently use the classical variance-based estimator;
rank normalization and folding are still planned. ESS is the classical estimator,
and MCSE estimates the error of the sample mean.

Invalid inputs return errors: ESS and MCSE need at least four retained draws per
chain; split ESS needs at least eight before splitting. Rhat needs two chains with
two draws each; split Rhat can split one chain with at least four draws. Diagnostics
reject non-finite retained draws and identical constant samples. Rhat also requires
positive within-chain variance. See the
[Stan reference manual](https://mc-stan.org/docs/reference-manual/analysis.html)
for the definitions and newer diagnostics.

Diagnostics center and scale retained draws internally to protect against large
offsets, overflow, and underflow. Rhat preserves each chain's local variation
before combining spreads. ESS and Rhat are independent of the original units;
MCSE is converted back to those units after computing the standard error.
Constant detection uses exact equality of retained draws rather than a fixed
absolute tolerance. Variation already rounded away in the input cannot be recovered.
If an estimate or its required variance cannot be represented, the function
returns an error instead of a non-finite, negative, or spuriously zero result.
Negative estimates of the classical truncated autocorrelation time also return
errors, even when all input draws are finite and nonconstant.

External validation
-------------------

In addition to the original two-chain Stan blocker fixture, tests use five
saved draw sets from `posterior` and ArviZ: eight schools (two independent fits),
a multivariate normal model, hierarchical radon regression, and rugby.
Three selected parameters per draw set give 15 cross-implementation cases,
each with four chains and either 100 or 500 stored posterior draws per chain.

Checked-in references were generated with R 4.5.2, `posterior` 1.7.1, and an
independent R implementation of the Stan 2.24 ESS formula. Tests compare
classical Rhat, split Rhat, ESS, split ESS, and unsplit MCSE across scalar/SIMD,
borrowed slices, `f32`, and contiguous/strided ndarray inputs. Cargo tests use
the saved files without R or network access.

Current and legacy ESS agree on the selected external cases, but differ on
some boundary inputs. `posterior::mcse_mean` also uses split ESS by default,
whereas this crate's MCSE uses unsplit ESS. Both forms and focused boundary
probes are recorded. See [fixture provenance and reproduction](test/external/README.md)
for source hashes, versions, licenses, estimator differences, and regeneration.

SIMD and benchmarks
-------------------

Enable `simd` to use the stable-Rust `wide` crate for four-lane `f64`
autocovariance dot products. It uses the compilation target's SIMD support with
portable fallbacks; the default build uses the scalar kernel. Both paths retain
`f64` arithmetic and are checked against the Stan fixtures and `arima` reference
covariances, including unaligned inputs and incomplete vector lanes. Floating
point summation order differs, so results need not be bit-for-bit identical.

```sh
cargo bench --bench diagnostics
cargo bench --bench diagnostics --features simd
```

The benchmark uses four synthetic chains at 256, 1,024, and 4,096 draws each,
reporting median time per call over seven 100 ms batches. Autocovariance still
does quadratic work; an FFT implementation is a separate future improvement
for long chains. SIMD accelerates the dot products, not array conversion or
the sequential variance estimator.

Roadmap
-------

**Next: diagnostic coverage and reliability**

- [ ] Add rank-normalized and folded split Rhat with reference tests from Stan or
      the `posterior` package, while preserving the classical API.
- [ ] Add bulk and tail ESS, then quantile-specific MCSE, with reference fixtures.
- [ ] Replace or deprecate the public test-oriented CSV helper with a fallible
      reader that reports I/O and parse errors and validates column counts.
- [ ] Add a worked example showing sampler output, warmup removal, and diagnostics.

**Later: API and performance**

- [ ] Benchmark FFT autocovariance against the direct scalar/SIMD kernels, and
      choose a crossover threshold using realistic chain lengths.
- [ ] Benchmark splitting and adapter conversion costs before further allocation changes.
- [ ] Evaluate named multi-parameter chains against concrete caller needs before
      committing to a container type.
- [ ] Consider thinning as an optional storage utility, with a validated nonzero
      stride and documentation of the statistical tradeoff.

References
----------

  [1]: Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
       Convergence of Iterative Simulations.
       _Journal of Computational and Graphical Statistics_, 7(4), 1998.

  [2]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
       Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.

  [3]: Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, Paul-Christian
       Burkner. Rank-normalization, folding, and localization: An improved R-hat
       for assessing convergence of MCMC, 2019. Retrieved from
       [https://arxiv.org/abs/1903.08008](https://arxiv.org/abs/1903.08008).

  [4]: Geyer, Charles J. Introduction to Markov Chain Monte Carlo.
       _Handbook of Markov Chain Monte Carlo_, edited by Steve Brooks, Andrew Gelman,
       Galin L. Jones, and Xiao-Li Meng. Chapman; Hall/CRC. 2011.

Acknowledgements
----------------

_Thanks to [Ivan Ukhov](https://github.com/IvanUkhov) for generously providing
the `mcmc` namespace on Cargo._
