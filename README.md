mcmc
====

A Rust library implementing various MCMC diagnostics and utilities, such as Gelman Rubin
potential scale reduction factor ("R̂" or "R hat"), effective sample size, chain splitting,
and others.

Implementation
--------------

Currently we expect plain vectors of `f64` floating point numbers.

Implementations for some of these diagnostics vary slightly, so reference implementations
are based on [Stan](https://github.com/stan-dev/stan), and unit tests are adapted to ensure
matching behavior.

Roadmap
-------

**Diagnostics**

- [X] Potential scale reduction factor
- [X] Split potential scale reduction factor
- [ ] Effective sample size
- [ ] Autocovariance

**Utilities**

- [X] Split chains as recommended in Vehtari, et al 2019
- [ ] Thinning

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


Acknowledgements
----------------

_Thanks to [Ivan Ukhov](https://github.com/IvanUkhov) for generously providing
the `mcmc` namespace on Cargo._