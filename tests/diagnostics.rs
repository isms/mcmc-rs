use approx::assert_abs_diff_eq;
use mcmc::Array2;
use mcmc::ess::{
    compute_effective_sample_size, compute_estimated_mcse, compute_split_effective_sample_size,
};
use mcmc::rhat::{potential_scale_reduction_factor, split_potential_scale_reduction_factor};
use mcmc::utils::split_chains;

type Diagnostic = fn(&[Vec<f64>]) -> anyhow::Result<f64>;

const DIAGNOSTICS: [(&str, Diagnostic); 5] = [
    ("ESS", compute_effective_sample_size),
    ("split ESS", compute_split_effective_sample_size),
    ("MCSE", compute_estimated_mcse),
    ("Rhat", potential_scale_reduction_factor),
    ("split Rhat", split_potential_scale_reduction_factor),
];

fn draws() -> Array2 {
    vec![
        vec![1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0],
        vec![2.0, 4.0, 1.0, 6.0, 3.0, 8.0, 5.0, 7.0],
    ]
}

#[test]
fn diagnostics_reject_empty_inputs_without_panicking() {
    for (name, diagnostic) in DIAGNOSTICS {
        for chains in [vec![], vec![vec![]], vec![vec![1.0; 8], vec![]]] {
            assert!(diagnostic(&chains).is_err(), "{name}: {chains:?}");
        }
    }
    assert!(split_chains(vec![]).is_err());
}

#[test]
fn diagnostics_use_only_the_common_prefix_in_either_chain_order() {
    for n in [8, 9] {
        let mut trimmed = draws();
        if n == 9 {
            trimmed.iter_mut().for_each(|chain| chain.push(0.0));
        }
        for longer in 0..2 {
            let mut unequal = trimmed.clone();
            unequal[longer].extend([1e10, f64::NAN, f64::INFINITY]);
            for (name, diagnostic) in DIAGNOSTICS {
                let expected = diagnostic(&trimmed).unwrap();
                let actual = diagnostic(&unequal).unwrap();
                assert!(actual.is_finite(), "{name}");
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
            assert_eq!(
                split_chains(unequal).unwrap(),
                split_chains(trimmed.clone()).unwrap()
            );
        }
    }
}

#[test]
fn diagnostics_reject_non_finite_retained_draws() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut chains = draws();
        chains[1][2] = value;
        for (name, diagnostic) in DIAGNOSTICS {
            assert!(diagnostic(&chains).is_err(), "{name}: {value}");
        }
    }
}

#[test]
fn diagnostics_enforce_minimum_chain_and_draw_counts() {
    assert!(split_chains(vec![vec![1.0]]).is_err());
    assert_eq!(
        split_chains(vec![vec![1.0, 2.0]]).unwrap(),
        vec![vec![1.0], vec![2.0]]
    );
    assert!(potential_scale_reduction_factor(&[vec![1.0, 2.0]]).is_err());
    assert!(potential_scale_reduction_factor(&[vec![1.0], vec![2.0]]).is_err());
    assert!(potential_scale_reduction_factor(&[vec![1.0, 2.0], vec![2.0, 3.0]]).is_ok());
    for n in 2..8 {
        let chains: Array2 = vec![(0..n).map(f64::from).collect()];
        assert!(compute_split_effective_sample_size(&chains).is_err());
    }
    assert!(compute_split_effective_sample_size(&[draws()[0].clone()]).is_ok());
    assert!(split_potential_scale_reduction_factor(&[vec![1.0, 2.0, 3.0]]).is_err());
    assert!(split_potential_scale_reduction_factor(&[vec![1.0, 2.0, 3.0, 4.0]]).is_ok());
}

#[test]
fn diagnostics_reject_identical_constant_draws() {
    let chains = vec![vec![1.0; 8], vec![1.0; 8]];
    for (name, diagnostic) in DIAGNOSTICS {
        assert!(diagnostic(&chains).is_err(), "{name}");
    }
}

#[test]
fn rhat_rejects_zero_within_chain_variance() {
    let chains = vec![vec![1.0; 8], vec![2.0; 8]];
    assert!(potential_scale_reduction_factor(&chains).is_err());
    assert!(split_potential_scale_reduction_factor(&chains).is_err());
}
