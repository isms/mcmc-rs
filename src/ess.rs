use crate::utils::{mean, sample_variance};
use crate::{Array1, Array2};
use anyhow::{anyhow, Error, Result};
use arima::acf;

pub fn compute_effective_sample_size(chains: &Array2) -> Result<f64, Error> {
    let num_chains = chains.len();
    let num_draws = chains.iter().map(|c| c.len()).min().unwrap();

    if num_draws < 4 {
        return Err(anyhow!("Must have at least 4 samples to compute ESS"));
    }

    let mut curr = chains[0][0];
    let mut prev = chains[0][0];
    let mut all_same = true;
    for c in 0..chains.len() {
        for i in 0..chains[0].len() {
            curr = chains[c][i];
            if !curr.is_finite() {
                return Err(anyhow!("All values must be finite to compute ESS"));
            }
            // the only way all_same can stay true the whole way through is if
            // every single element of all the chains is the same
            all_same &= curr == prev;
            prev = curr;
        }
    }
    if all_same {
        let msg = format!("No ESS when elements are all constant (value={})", curr);
        return Err(anyhow!(msg));
    }

    // check if any element is NaN
    // TODO: check

    let mut chain_acov: Array2 = Vec::new();
    let mut chain_mean: Array1 = Vec::new();
    let mut chain_var: Array1 = Vec::new();
    for chain in chains.iter() {
        let acov = acf::acf(&chain, None, true).unwrap();
        chain_mean.push(mean(&chain)?);
        chain_var.push(acov[0] * num_draws as f64 / (num_draws as f64 - 1.0));
        chain_acov.push(acov);
    }

    let mean_var = mean(&chain_var)?;
    let mut var_plus = mean_var * (num_draws as f64 - 1.0) / num_draws as f64;
    if num_chains > 1 {
        var_plus += sample_variance(&chain_mean)?;
    }

    let mut rho_hat_s: Array1 = vec![0.0; num_draws];
    let mut acov_s: Array1 = vec![0.0; num_chains];
    for c in 0..num_chains {
        acov_s[c] = chain_acov[c][1]
    }
    let mut rho_hat_even = 1.0;
    rho_hat_s[0] = rho_hat_even;
    let mut rho_hat_odd = 1.0 - (mean_var - mean(&acov_s)?) / var_plus;
    rho_hat_s[1] = rho_hat_odd;

    // Convert raw autocovariance estimators into Geyer's initial
    // positive sequence. Loop only until num_draws - 4 to
    // leave the last pair of autocorrelations as a bias term that
    // reduces variance in the case of antithetical chains.
    let mut s = 1;
    while s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0.0 {
        for c in 0..num_chains {
            acov_s[c] = chain_acov[c][s + 1];
        }
        rho_hat_even = 1.0 - (mean_var - mean(&acov_s)?) / var_plus;
        for c in 0..num_chains {
            acov_s[c] = chain_acov[c][s + 2];
        }
        rho_hat_odd = 1.0 - (mean_var - mean(&acov_s)?) / var_plus;
        if (rho_hat_even + rho_hat_odd) >= 0.0 {
            rho_hat_s[s + 1] = rho_hat_even;
            rho_hat_s[s + 2] = rho_hat_odd;
        }
        s += 2;
    }

    let max_s = s;
    // this is used in the improved estimate, which reduces variance
    // in antithetic case -- see tau_hat below
    if rho_hat_even > 0.0 {
        rho_hat_s[max_s + 1] = rho_hat_even;
    }

    // Convert Geyer's initial positive sequence into an initial
    // monotone sequence
    for s in (1..=(max_s - 3)).step_by(2) {
        if (rho_hat_s[s + 1] + rho_hat_s[s + 2]) > (rho_hat_s[s - 1] + rho_hat_s[s]) {
            rho_hat_s[s + 1] = (rho_hat_s[s - 1] + rho_hat_s[s]) / 2.0;
            rho_hat_s[s + 2] = rho_hat_s[s + 1];
        }
    }

    let num_total_draws = num_chains as f64 * num_draws as f64;
    // Geyer's truncated estimator for the asymptotic variance
    // Improved estimate reduces variance in antithetic case
    let tau_hat: f64 =
        -1.0 + 2.0 * rho_hat_s.iter().take(max_s).sum::<f64>() + rho_hat_s[max_s + 1];
    let option1: f64 = num_total_draws / tau_hat;
    let option2: f64 = num_total_draws * num_total_draws.log10();
    Ok(option1.min(option2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::read_csv;
    use std::path::PathBuf;

    #[test]
    fn test_identical_autocovariance_in_arima_library_and_stan() {
        let arr = vec![
            0.747858687681513,
            0.290118161168511,
            -0.66263075102762,
            -0.00794439358648058,
            0.612494029879686,
            1.15915333101436,
            0.844402455747637,
            -0.493298834393585,
            0.140306938408938,
            -0.207331367372662,
            0.344322796977632,
            -0.216755313401662,
            -0.704730639551491,
            -0.262457923752462,
            0.338587814578015,
            0.79334841402936,
            -0.495245866959037,
            -0.736378128523917,
            -1.10220108378805,
            2.37069694852591,
        ];
        let stan_acov = vec![
            0.6269672577,
            -0.0113804234,
            -0.1668563930,
            -0.2086591087,
            0.1016590536,
            0.1767212413,
            -0.0059714922,
            -0.1489622883,
            -0.0996503101,
            0.0996094900,
            0.0450098619,
            -0.0109203038,
            -0.2154921627,
            -0.0374684937,
            0.1274360411,
            0.1121981758,
            0.0073812983,
            -0.1254719533,
            -0.0208019612,
            0.0681360996,
        ];
        let arima_acf_cov = acf::acf(&arr, None, true).unwrap();

        for i in 0..arr.len() {
            assert_abs_diff_eq!(arima_acf_cov[i], stan_acov[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_compute_effective_sample_size_one_chain() {
        // Based on the unit test in Stan 2.2.4 but with more digits of precision
        // https://github.com/stan-dev/stan/blob/v2.24.0/src/test/unit/analyze/mcmc/compute_effective_sample_size_test.cpp#L22-L57
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let samples1 = read_csv(&d.join("test/stan/blocker.1.csv"), 41, 1000);

        let expected_ess = vec![
            284.77189783,
            105.68133852,
            668.69085592,
            569.40248945,
            523.29194917,
            403.39642868,
            432.34846958,
            441.28796989,
            209.86506314,
            472.82764779,
            451.13261546,
            429.32700879,
            375.41770775,
            507.37609173,
            222.90641272,
            218.27768923,
            316.07200543,
            489.08398336,
            404.05662679,
            379.35140759,
            232.84915591,
            445.68359658,
            675.56238444,
            362.88182976,
            720.20116516,
            426.74354119,
            376.69233682,
            509.39946501,
            247.15051215,
            440.42603897,
            160.53246711,
            411.10864659,
            419.39514647,
            411.98813366,
            425.52858473,
            420.61546436,
            336.49516091,
            131.94624221,
            461.60551174,
            469.62779507,
            479.45824312,
            611.19593264,
            483.30090212,
            584.61443630,
            500.26381682,
            453.11227606,
            646.06673456,
            72.18775005,
        ];

        for (i, expected) in expected_ess.iter().enumerate() {
            let chains = vec![samples1[i + 4].clone()];
            let actual = compute_effective_sample_size(&chains).unwrap();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }
    }
}
