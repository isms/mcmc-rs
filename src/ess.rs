use crate::utils::{flatten, mean, sample_variance, split_chains};
use crate::{Array1, Array2};
use anyhow::{anyhow, Error, Result};
use arima::acf;

/// Computes the effective sample size (ESS) for the specified
/// parameter across all kept samples.  The value returned is the
/// minimum of ESS and the number_total_draws * log10(number_total_draws).
/// When the number of total draws N is odd, the (N+1)/2th draw is ignored.
///
/// Chains are trimmed from the back to match the
/// length of the shortest chain.  Note that the effective sample size
/// can not be estimated with fewer than four draws.
///
/// See more details in Stan reference manual section
/// ["Effective Sample Size"](http://mc-stan.org/users/documentation)
///
/// Based on reference implementation in Stan v2.4.0 at
/// [https://github.com/stan-dev/stan/blob/v2.24.0/src/stan/analyze/mcmc/compute_effective_sample_size.hpp#L32-L138]()
///
///
/// # Arguments
/// * `chains` - Reference to a vector of chains, each of which is a vector of samples for
///              the same parameter
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
            all_same &= (curr - prev).abs() < 1e-10;
            prev = curr;
        }
    }
    if all_same {
        let msg = format!("No ESS when elements are all constant (value={})", curr);
        return Err(anyhow!(msg));
    }

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
    let mut s = 1;
    while max_s >= 3 && s <= (max_s - 3) {
        if (rho_hat_s[s + 1] + rho_hat_s[s + 2]) > (rho_hat_s[s - 1] + rho_hat_s[s]) {
            rho_hat_s[s + 1] = (rho_hat_s[s - 1] + rho_hat_s[s]) / 2.0;
            rho_hat_s[s + 2] = rho_hat_s[s + 1];
        };
        s += 2;
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

/// Computes the split effective sample size (ESS) for the specified
/// parameter across all kept samples.  The value returned is the
/// minimum of ESS and the number_total_draws * log10(number_total_draws).
/// When the number of total draws N is odd, the (N+1)/2th draw is ignored.
///
/// Chains are trimmed from the back to match the
/// length of the shortest chain.  Note that the effective sample size
/// can not be estimated with fewer than four draws.
///
/// See more details in Stan reference manual section
/// ["Effective Sample Size"](http://mc-stan.org/users/documentation)
///
/// Based on reference implementation in Stan v2.4.0 at
/// [https://github.com/stan-dev/stan/blob/v2.24.0/src/stan/analyze/mcmc/compute_effective_sample_size.hpp#L185-L199]()
///
///
/// # Arguments
/// * `chains` - Reference to a vector of chains, each of which is a vector of samples for
///              the same parameter
pub fn compute_split_effective_sample_size(chains: &Array2) -> Result<f64, Error> {
    let num_draws = chains.iter().map(|c| c.len()).min().unwrap();
    // trim chains to the length of the shortest chain
    let mut trimmed = Vec::new();
    for chain in chains.iter() {
        trimmed.push(chain[..num_draws].to_vec());
    }
    let split = split_chains(trimmed)?;
    compute_effective_sample_size(&split)
}

/// Computes the Monte Carlo Standard Error (MCSE) for the specified parameter
/// across all samples, which is the standard deviation of the samples over the
/// square root of effective sample size.
///
/// See the Stan reference manual section
/// ["Estimation of MCMC Standard Error"](https://mc-stan.org/docs/2_24/reference-manual/effective-sample-size-section.html#estimation-of-mcmc-standard-error)
///
///
/// # Arguments
/// * `chains` - Reference to a vector of chains, each of which is a vector of samples for
///              the same parameter
pub fn compute_estimated_mcse(chains: &Array2) -> Result<f64, Error> {
    let ess = compute_effective_sample_size(&chains)?;
    let var = sample_variance(&flatten(chains))?;
    Ok((var / ess).sqrt())
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

    #[test]
    fn test_compute_effective_sample_size_two_chains() {
        // Based on the unit test in Stan 2.2.4 but with more digits of precision
        // https://github.com/stan-dev/stan/blob/v2.24.0/src/test/unit/analyze/mcmc/compute_effective_sample_size_test.cpp#L22-L57
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let samples1 = read_csv(&d.join("test/stan/blocker.1.csv"), 41, 1000);
        let samples2 = read_csv(&d.join("test/stan/blocker.2.csv"), 41, 1000);

        let expected_ess = vec![
            467.36757686,
            138.62780027,
            1171.62891355,
            543.89301136,
            519.89670767,
            590.53267759,
            764.75729757,
            690.21936104,
            326.21745260,
            505.50985231,
            356.44510650,
            590.14928533,
            655.71371952,
            480.72769500,
            178.74587968,
            184.87140679,
            643.85564048,
            472.13048627,
            563.84825583,
            584.74450883,
            449.13707437,
            400.23475140,
            339.21683773,
            680.60538752,
            1410.38271694,
            836.01702508,
            871.38979093,
            952.26509331,
            620.94420986,
            869.97895746,
            235.16790031,
            788.52022938,
            911.34806602,
            234.22761856,
            909.20881398,
            748.70965886,
            722.36225578,
            196.76168649,
            945.74138475,
            768.79701460,
            725.52731616,
            1078.46726260,
            471.56987828,
            956.35673474,
            498.19497759,
            582.66324514,
            696.85069050,
            99.78353935,
        ];

        for (i, expected) in expected_ess.iter().enumerate() {
            let chains = vec![samples1[i + 4].clone(), samples2[i + 4].clone()];
            let actual = compute_effective_sample_size(&chains).unwrap();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_compute_split_effective_sample_size_two_chains() {
        // Based on the unit test in Stan 2.2.4 but with more digits of precision
        // https://github.com/stan-dev/stan/blob/v2.24.0/src/test/unit/analyze/mcmc/compute_effective_sample_size_test.cpp#L22-L57
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let samples1 = read_csv(&d.join("test/stan/blocker.1.csv"), 41, 1000);
        let samples2 = read_csv(&d.join("test/stan/blocker.2.csv"), 41, 1000);

        let expected_ess = vec![
            467.84472286,
            134.49757091,
            1189.59121923,
            569.19341812,
            525.00159997,
            572.69157167,
            763.91010048,
            710.97717906,
            338.29803319,
            493.34818866,
            333.49289697,
            588.28304375,
            665.62041018,
            504.26271137,
            187.04932436,
            156.91316803,
            650.01816166,
            501.45489247,
            570.16074452,
            550.36645240,
            446.21946848,
            408.21801438,
            364.20430683,
            678.69938531,
            1419.23404653,
            841.74191739,
            881.92328583,
            960.42014222,
            610.92148539,
            917.64184496,
            239.59903291,
            773.72649323,
            921.33231871,
            227.34002818,
            900.81898633,
            748.47755057,
            727.36524051,
            184.94880796,
            948.42542442,
            776.03021619,
            735.27919044,
            1077.17739932,
            475.25192235,
            955.28139954,
            503.04549546,
            591.91289033,
            715.96959077,
            95.59380790,
        ];

        for (i, expected) in expected_ess.iter().enumerate() {
            let chains = vec![samples1[i + 4].clone(), samples2[i + 4].clone()];
            let actual = compute_split_effective_sample_size(&chains).unwrap();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }
    }

    #[test]
    pub fn compute_effective_sample_size_minimum_n() {
        let chains = vec![vec![1.0, 2.0, 3.0]];
        let ess = compute_effective_sample_size(&chains);
        assert!(ess.is_err());
    }

    #[test]
    pub fn compute_effective_sample_size_sufficient_n() {
        let chains = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let ess = compute_effective_sample_size(&chains);
        assert!(ess.unwrap().is_finite());
    }

    #[test]
    pub fn compute_effective_sample_size_nan() {
        let chains = vec![vec![1.0, f64::NAN, 3.0, 4.0]];
        let ess = compute_effective_sample_size(&chains);
        assert!(ess.is_err());
    }

    #[test]
    pub fn compute_effective_sample_size_constant() {
        let chains = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let ess = compute_effective_sample_size(&chains);
        assert!(ess.is_err());
    }

    #[test]
    fn test_compute_estimated_mcse() {
        // Based on the unit test in Stan 2.2.4 but with more digits of precision
        // https://github.com/stan-dev/stan/blob/v2.24.0/src/test/unit/analyze/mcmc/compute_effective_sample_size_test.cpp#L22-L57
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let samples1 = read_csv(&d.join("test/stan/blocker.1.csv"), 41, 1000);
        let samples2 = read_csv(&d.join("test/stan/blocker.2.csv"), 41, 1000);

        let expected_mcse = vec![
            1.041454110e+00,
            3.791888876e-02,
            2.173376810e-02,
            1.825876681e-02,
            2.661215900e-03,
            1.131246947e-03,
            1.260798781e-02,
            1.030700714e-02,
            1.228143969e-02,
            3.330029841e-03,
            5.353227092e-03,
            1.308588008e-02,
            4.700032366e-03,
            5.257861092e-03,
            7.533851160e-03,
            2.758236978e-03,
            4.345012004e-03,
            5.841727439e-03,
            1.771073621e-02,
            1.037211580e-02,
            6.046724542e-03,
            6.605926256e-03,
            7.575775682e-03,
            1.190997112e-02,
            1.602859734e-02,
            7.008613253e-03,
            7.249334314e-03,
            5.329946992e-03,
            3.879811372e-03,
            4.748270142e-03,
            4.865599426e-03,
            2.880021654e-03,
            5.057902504e-03,
            4.800369415e-03,
            7.453771374e-03,
            4.140658457e-03,
            3.925703715e-03,
            5.498448282e-03,
            3.515675895e-03,
            4.387941995e-03,
            5.155243445e-03,
            1.318791554e-02,
            3.738973852e-03,
            4.325514463e-03,
            4.724583423e-03,
            4.468024552e-03,
            7.140312463e-03,
            3.651782874e-03,
            5.773674797e-03,
            5.189233437e-03,
            6.343078722e-03,
            4.972475627e-03,
        ];
        for (i, expected) in expected_mcse.iter().enumerate() {
            let chains = vec![samples1[i].clone(), samples2[i].clone()];
            let actual = compute_estimated_mcse(&chains).unwrap();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }
    }
}
