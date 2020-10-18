MCMC diangostics in Rust
====

[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]
[![Actions Status][build-badge]][build-url]

[crates-badge]: https://img.shields.io/crates/v/mcmc.svg
[crates-url]: https://crates.io/crates/mmcmc
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: LICENSE
[build-badge]: https://github.com/isms/mcmc-rs/workflows/Rust/badge.svg
[build-url]: https://github.com/isms/mcmc-rs/actions

A Rust library implementing various MCMC diagnostics and utilities, such as Gelman Rubin
potential scale reduction factor (R hat), effective sample size (ESS), chain splitting,
and others.

This crate is language agnostic and intended to work with the outputs of any MCMC sampler
(e.g. Stan, PyMC3, Turing.jl, etc).

Implementation
--------------

Currently we expect plain vectors of `f64` floating point numbers, but this may be
worth generalizing to `f32`s as well (see roadmap below).

Implementations for some of these diagnostics vary slightly, so reference implementations
are based on [Stan](https://github.com/stan-dev/stan), and unit tests are adapted from the
Stan codebase to ensure matching behavior.

Roadmap
-------

**Diagnostics**

- [X] Potential scale reduction factor
- [X] Split potential scale reduction factor
- [X] Effective sample size
- [X] Monte Carlo Standard Error

**Utilities**

- [X] Split chains as recommended in Vehtari, et al 2019
- [ ] Thinning

**Data structures**

- [ ] Introduce `Num` type to generalize our implementations to work for `f32` or `f64`.
- [ ] Would it be helpful to have some kind of struct that can represent
      one or more sample chains with a parameter name?

**Performance**

- [ ] Remove unnecessary copying or allocation

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
       [http://arxiv.org/abs/1903.08008]().

  [4]: Geyer, Charles J. Introduction to Markov Chain Monte Carlo.
       _Handbook of Markov Chain Monte Carlo_, edited by Steve Brooks, Andrew Gelman,
       Galin L. Jones, and Xiao-Li Meng. Chapman; Hall/CRC. 2011.

Acknowledgements
----------------

_Thanks to [Ivan Ukhov](https://github.com/IvanUkhov) for generously providing
the `mcmc` namespace on Cargo._
