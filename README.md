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

- [ ] Benchmark FFT autocovariance against the direct kernels, and
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
