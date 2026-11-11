# External validation fixtures

These are saved posterior draws from external projects. No MCMC sampler is run
by this repository's test suite. The original `../stan/blocker.*.csv` fixtures
remain covered by the existing Stan-reference tests.

| Fixture | Source model | Chains | Draws per chain | Selected variables/elements |
| --- | --- | ---: | ---: | --- |
| `posterior_eight_schools` | Hierarchical eight schools, Stan | 4 | 100 | `mu`, `tau`, `theta[1]` |
| `posterior_multi_normal` | 3D multivariate normal mean/covariance, Stan | 4 | 100 | `mu[1]`, `mu[2]`, `Sigma[1,2]` |
| `arviz_centered_eight` | Centered eight schools, PyMC 5.20.0 | 4 | 500 | `mu`, `tau`, first `theta` |
| `arviz_radon` | Hierarchical Minnesota radon regression, PyMC3 3.9.2 | 4 | 500 | first `g`, `b`, `sigma_a` |
| `arviz_rugby` | Six Nations rugby model, PyMC 5.10.4+7.g34d2a5d9 | 4 | 500 | `home`, first `atts`, `sd_def` |

This adds five draw sets across four models, including two different eight-schools
fits. Coverage includes hierarchical scale parameters, multivariate covariance
parameters, and a centered fit that ArviZ documents as difficult for NUTS. It
is a regression corpus, not an exhaustive validation of all posterior geometries.
A dedicated Student-t/Cauchy target and larger models remain useful extensions.
The two `posterior` examples are intentionally short demonstration/test samples.

The variable selections were fixed before comparing diagnostic results. All
stored posterior iterations of each selected variable are retained in original
chain order, without additional thinning, warmup removal, resampling, or shuffling.
CSV chain and draw indices start at one. Variable aliases and source element
indices (zero-based) appear in `sources.json`. Values are serialized to 17
significant digits for lossless binary64 round trips.

## Sources and licensing

- [`posterior` example draws](https://github.com/stan-dev/posterior/blob/540cc15f7f3a81913f2a02a0ac04700e0d352a20/R/example_draws.R), from `R/sysdata.rda` in the pinned 1.7.1 source archive: BSD 3-Clause.
- [ArviZ examples](https://github.com/arviz-devs/arviz_example_data/tree/e56f5b5d7ae6e605f763661934ced3ac55a6f490): centered eight schools, Apache 2.0.
- Oriol Abril Pla, [radon hierarchical model inferencedata](https://doi.org/10.6084/m9.figshare.12711695.v1): Apache 2.0.
- Oriol Abril Pla, [Rugby base model](https://doi.org/10.6084/m9.figshare.25293310.v2): CC BY 4.0.

`NOTICE` describes our extraction changes and attribution. The upstream license
texts are retained in `licenses/`. `sources.json` pins source URLs, upstream
commits/file IDs, input SHA-256 hashes, reference versions, and variable selections.
`manifest.json` records conversion/sampler metadata and hashes of generated files
and generation inputs. The original NetCDF/R package archives are cached outside
this fixture directory and are not committed.

## Reference calculations

References were actually generated with **R 4.5.2**, **posterior 1.7.1**, and
**matrixStats 1.5.0**. They do not call the Rust library.

`expected.csv` has one row for each of the 15 selected scalar variables:

- `rhat` / `split_rhat`: `posterior::rhat_basic(x, split = FALSE/TRUE)`.
- `legacy_ess` / `legacy_split_ess`: an R port of the truncation and variance
  formulas in [Stan 2.24.0's ESS implementation](https://github.com/stan-dev/stan/blob/v2.24.0/src/stan/analyze/mcmc/compute_effective_sample_size.hpp), using independently calculated biased autocovariances from `stats::acf`.
- `legacy_mcse`: `sd(as.vector(x)) / sqrt(legacy_ess)`.
- `posterior_ess` / `posterior_split_ess`: `posterior::ess_basic(x, split = FALSE/TRUE)`.
- `posterior_mcse_unsplit`: `sd(as.vector(x)) / sqrt(posterior_ess)`, matching this crate's unsplit definition.
- `posterior_mcse_mean`: the actual default `posterior::mcse_mean(x)`, which uses split-chain ESS.

R matrices use **iterations x chains**. Rust slices and the ndarray adapters use
**chains x draws**. The generation and test code explicitly preserve that mapping.
The legacy R port follows the estimator for finite fixture variables; it does not
try to reproduce every historical Eigen constant-detection tolerance.

For these external selections, legacy and current R ESS values agree to floating
point precision (maximum relative difference below `2e-15`). Rust assertions allow
relative error `1e-9` plus absolute error `1e-12`; adapters rounding source values
to `f32` allow relative error `1e-5`. Every selected parameter is checked through
owned vectors, borrowed slices, `f32` inputs, and, with its feature enabled,
contiguous and column-major/strided ndarray views. Running with `simd` repeats
the same reference assertions through the SIMD kernel.

### Estimator differences

The selected external cases agree; that does not make the estimators universally
identical. `semantics.csv` contains two separate **synthetic probes**, generated
by the same R script and tested explicitly:

- `negative_tau`: `[-1,-1,-1,1,-1,0]` has legacy truncated autocorrelation time
  `-4/35` and historical ESS `-52.5`. This crate returns an error; current
  `posterior` floors the time and returns the capped positive ESS.
- `positive_tail_cutoff`: two short, separated nonconstant chains retain positive
  autocorrelation pairs to the last permitted lag. Stan 2.24 and current
  `posterior` handle the final lag/bias term differently. This crate preserves
  the legacy result, and both reference values are recorded.

Default `posterior::mcse_mean` can differ for a second reason: it uses split ESS.
Comparisons against this crate therefore use `posterior_mcse_unsplit`. The default
reference is retained to make that distinction visible.

## Running and regenerating

Ordinary verification needs only Rust. Hash verification needs Python 3.11+ with
no third-party packages, R, or network access:

```sh
cargo test --test external
cargo test --test external --all-features
python3 scripts/validation/generate.py --check
```

Regeneration requires downloads and the reference/conversion tools. A conda-forge
environment recipe pins the relevant versions:

```sh
micromamba create -y -p /tmp/mcmc-validation-env -f scripts/validation/environment.yml
micromamba run -p /tmp/mcmc-validation-env python scripts/validation/generate.py \
  --rscript /tmp/mcmc-validation-env/bin/Rscript
```

The generator installs the checksum-pinned `posterior` 1.7.1 source archive in its
private cache library; the conda `posterior` package supplies its dependencies.
It rejects a different R or matrixStats version. Existing source downloads are
reused only after SHA-256 verification. The default cache is
`target/external-validation`; `--cache PATH` selects another location.

With R 4.5.2 and the necessary R dependencies already installed, the pinned Python
conversion dependencies can alternatively be supplied by `uv`:

```sh
uv run scripts/validation/generate.py --rscript /path/to/Rscript
```

To compare a fresh regeneration without overwriting the checked-in snapshots:

```sh
micromamba run -p /tmp/mcmc-validation-env python scripts/validation/generate.py \
  --rscript /tmp/mcmc-validation-env/bin/Rscript --output /tmp/mcmc-fixture-reproduction
diff -ru --exclude=README.md --exclude=sources.json test/external /tmp/mcmc-fixture-reproduction
```

The generated directory omits the human-maintained `README.md` and `sources.json`;
compare the CSVs, `manifest.json`, `NOTICE`, and `licenses/`. Generation is
deterministic: no sampling, random seeds, generation-time timestamps, or Rust-derived expected
values are involved. When changing generation inputs, regenerate the manifest,
review the diagnostic changes, and rerun all Rust feature configurations.
