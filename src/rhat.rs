use crate::utils::{mean, sample_variance, split_chains};
use crate::{Array1, Array2};
use anyhow::{Error, Result};

/// Computes the potential scale reduction (Rhat) for the specified
/// parameter across all kept samples.  Chains are trimmed from the
/// back to match the length of the shortest chain.
///
/// See more details in Stan reference manual section
/// ["Potential Scale Reduction"](https://mc-stan.org/docs/2_24/reference-manual/notation-for-samples-chains-and-draws.html#potential-scale-reduction).
///
/// Based on reference implementation in Stan v2.24.0 at
/// [https://github.com/stan-dev/stan/blob/v2.24.0/src/stan/analyze/mcmc/compute_potential_scale_reduction.hpp]()
pub fn potential_scale_reduction_factor(chains: &Array2) -> Result<f64, Error> {
    let m = chains.len();
    let n = chains.iter().map(|c| c.len()).min().unwrap();
    let mut split_chain_mean: Array1 = Vec::new();
    let mut split_chain_var: Array1 = Vec::new();

    for chain in chains.iter().take(m) {
        let chain_mean = mean(chain)?;
        split_chain_mean.push(chain_mean);
        let chain_var = sample_variance(chain)?;
        split_chain_var.push(chain_var);
    }

    let n = n as f64;
    let var_between = n * sample_variance(&split_chain_mean)?;
    let var_within = mean(&split_chain_var)?;
    let result = ((var_between / var_within + n - 1.0) / n).sqrt();

    Ok(result)
}

/// Computes the split potential scale reduction (Rhat) for the
/// specified parameter across all kept samples.  When the number of
/// total draws N is odd, the (N+1)/2th draw is ignored.
///
/// Chains are trimmed from the back to match the
/// length of the shortest chain.  Argument size will be broadcast to
/// same length as draws.
///
/// See more details in Stan reference manual section
/// ["Potential Scale Reduction"](https://mc-stan.org/docs/2_24/reference-manual/notation-for-samples-chains-and-draws.html#potential-scale-reduction)
///
/// Based on reference implementation in Stan v2.24.0 at
/// [https://github.com/stan-dev/stan/blob/v2.24.0/src/stan/analyze/mcmc/compute_potential_scale_reduction.hpp]()
pub fn split_potential_scale_reduction_factor(chains: &Array2) -> Result<f64, Error> {
    let num_draws = chains.iter().map(|c| c.len()).min().unwrap();
    // trim chains to the length of the shortest chain
    let mut trimmed = Vec::new();
    for chain in chains.iter() {
        trimmed.push(chain[..num_draws].to_vec());
    }
    let split = split_chains(trimmed)?;
    potential_scale_reduction_factor(&split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::read_csv;
    use std::path::PathBuf;

    #[test]
    fn test_split_chains() {
        // Make sure the we Err on empty or minimum 0 length chains
        let a: Array1 = vec![1.0];
        let b: Array1 = vec![];
        let c: Array1 = vec![];
        let chains = vec![a, b, c];
        assert!(split_chains(chains).is_err());

        let a: Array1 = vec![];
        let b: Array1 = vec![];
        let chains = vec![a, b];
        assert!(split_chains(chains).is_err());

        // Regular split with even numbers
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let chains = vec![a, b];
        let split = split_chains(chains).unwrap();
        assert_eq!(split[0], vec![1.0, 2.0]);
        assert_eq!(split[1], vec![3.0, 4.0]);
        assert_eq!(split[2], vec![5.0, 6.0]);
        assert_eq!(split[3], vec![7.0, 8.0]);

        // Make sure the middle value gets dropped per the Stan reference implementation
        let a = vec![1.0, 2.0, 3.0, 4.0, 4.5];
        let b = vec![5.0, 6.0, 7.0, 8.0, 8.5];
        let chains = vec![a, b];
        let split = split_chains(chains).unwrap();
        assert_eq!(split[0], vec![1.0, 2.0]);
        assert_eq!(split[1], vec![4.0, 4.5]);
        assert_eq!(split[2], vec![5.0, 6.0]);
        assert_eq!(split[3], vec![8.0, 8.5]);
    }

    #[test]
    fn test_stan_blocker_unit_test_potential_scale_reduction_factor() {
        // Based on the unit test in Stan 2.2.4 but using slightly more precision:
        // https://github.com/stan-dev/stan/blob/v2.24.0/src/test/unit/analyze/mcmc/compute_potential_scale_reduction_test.cpp#L63-L99
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let samples1 = read_csv(&d.join("test/stan/blocker.1.csv"), 41, 1000);
        let samples2 = read_csv(&d.join("test/stan/blocker.2.csv"), 41, 1000);

        let expected_rhats = vec![
            1.000417, 1.000359, 0.999546, 1.000466, 1.001193, 1.000887, 1.000175, 1.000190,
            1.002262, 0.999539, 0.999603, 0.999511, 1.002374, 1.005145, 1.005657, 0.999572,
            1.000986, 1.008535, 1.000799, 0.999605, 1.000602, 1.000457, 1.010228, 0.999600,
            1.001100, 0.999672, 0.999734, 0.999579, 1.002418, 1.002131, 1.002444, 0.999978,
            0.999686, 1.000791, 0.999546, 1.000902, 1.001362, 1.002881, 1.000360, 0.999889,
            1.000768, 0.999972, 1.001942, 0.999718, 1.002574, 1.001089, 1.000042, 0.999555,
        ];
        for (i, expected) in expected_rhats.iter().enumerate() {
            let chains = vec![samples1[i + 4].clone(), samples2[i + 4].clone()];
            let actual = potential_scale_reduction_factor(&chains).unwrap();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_stan_blocker_unit_test_split_potential_scale_reduction_factor() {
        // Based on the unit test in Stan 2.2.4 but using slightly more precision:
        // https://github.com/stan-dev/stan/blob/v2.24.0/src/test/unit/analyze/mcmc/compute_potential_scale_reduction_test.cpp#L135-L175
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let samples1 = read_csv(&d.join("test/stan/blocker.1.csv"), 41, 1000);
        let samples2 = read_csv(&d.join("test/stan/blocker.2.csv"), 41, 1000);

        let expected_rhats = vec![
            1.00718209, 1.00472781, 0.99920319, 1.00060574, 1.00378194, 1.01031069, 1.00173146,
            1.00449845, 1.00110520, 1.00336914, 1.00546003, 1.00105054, 1.00557523, 1.00462913,
            1.00534461, 1.01243951, 1.00174291, 1.00718051, 1.00186144, 1.00554010, 1.00436048,
            1.00146549, 1.01016783, 1.00161542, 1.00143164, 1.00058020, 0.99922069, 1.00012079,
            1.01028435, 1.00100481, 1.00304822, 1.00435219, 1.00054786, 1.00246262, 1.00446672,
            1.00479686, 1.00209188, 1.01159003, 1.00201738, 1.00076562, 1.00209813, 1.00262278,
            1.00308325, 1.00196623, 1.00246300, 1.00084883, 1.00047332, 1.00735293,
        ];
        for (i, expected) in expected_rhats.iter().enumerate() {
            let chains = vec![samples1[i + 4].clone(), samples2[i + 4].clone()];
            let actual = split_potential_scale_reduction_factor(&chains).unwrap();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
        }
    }
}
