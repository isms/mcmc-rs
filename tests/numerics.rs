use approx::assert_relative_eq;
use mcmc::ess::{
    compute_effective_sample_size, compute_estimated_mcse, compute_split_effective_sample_size,
};
use mcmc::rhat::{potential_scale_reduction_factor, split_potential_scale_reduction_factor};

const DRAWS: [[f64; 8]; 2] = [
    [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0],
    [2.0, 4.0, 1.0, 6.0, 3.0, 8.0, 5.0, 7.0],
];

fn check(chains: &[[f64; 8]; 2], units: f64) {
    // Computed using exact rational means, sample variances, and lag
    // covariances in the Stan 2.24 formulas, before the final sqrt/log.
    let actual = [
        compute_effective_sample_size(chains).unwrap(),
        compute_split_effective_sample_size(chains).unwrap(),
        potential_scale_reduction_factor(chains).unwrap(),
        split_potential_scale_reduction_factor(chains).unwrap(),
        compute_estimated_mcse(chains).unwrap() / units.abs(),
    ];
    let expected = [
        12.218181818181819,
        8.0,
        0.9354143466934853,
        1.2398524277321037,
        0.67700320038633,
    ];
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && actual > 0.0);
        assert_relative_eq!(actual, expected, max_relative = 1e-12);
    }
}

#[test]
fn diagnostics_are_invariant_to_tiny_and_large_units() {
    for scale in [1e-300, 1e-200, 1e-12, 1.0, 1e150, 1e300, f64::MAX / 8.0] {
        for sign in [-1.0, 1.0] {
            let units = scale * sign;
            let chains = DRAWS.map(|chain| chain.map(|draw| draw * units));
            check(&chains, units);
        }
    }
}

#[test]
fn diagnostics_preserve_variation_at_large_offsets() {
    // These increments are exactly representable at the corresponding offsets.
    for (offset, units) in [
        (2.0_f64.powi(50), 0.25),
        (2.0_f64.powi(100), 2.0_f64.powi(48)),
        (2.0_f64.powi(1020), 2.0_f64.powi(968)),
    ] {
        let chains = DRAWS.map(|chain| chain.map(|draw| offset + draw * units));
        check(&chains, units);
    }
}

#[test]
fn diagnostics_handle_the_full_finite_f64_span() {
    let chains = [[
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
    ]; 2];
    assert_relative_eq!(
        compute_effective_sample_size(&chains).unwrap(),
        8.0,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        compute_split_effective_sample_size(&chains).unwrap(),
        8.0,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        potential_scale_reduction_factor(&chains).unwrap(),
        (7.0_f64 / 8.0).sqrt(),
        max_relative = 1e-12
    );
    assert_relative_eq!(
        split_potential_scale_reduction_factor(&chains).unwrap(),
        (3.0_f64 / 4.0).sqrt(),
        max_relative = 1e-12
    );
    let mcse = compute_estimated_mcse(&chains).unwrap();
    assert!(mcse.is_finite() && mcse > 0.0);
    assert_relative_eq!(
        mcse / f64::MAX,
        (2.0_f64 / 15.0).sqrt(),
        max_relative = 1e-12
    );
}

#[test]
fn dimensionless_diagnostics_work_for_subnormal_samples() {
    let unit = f64::from_bits(1);
    let chains = DRAWS.map(|chain| chain.map(|draw| draw * unit));
    assert_relative_eq!(
        compute_effective_sample_size(&chains).unwrap(),
        12.218181818181819,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        potential_scale_reduction_factor(&chains).unwrap(),
        (7.0_f64 / 8.0).sqrt(),
        max_relative = 1e-12
    );
}

#[test]
fn mcse_reports_unrepresentable_positive_results() {
    let chains = [[0.0, 0.0, 0.0, f64::from_bits(1)]];
    assert!(compute_effective_sample_size(&chains).unwrap().is_finite());
    assert!(compute_estimated_mcse(&chains).is_err());
}

#[test]
fn ess_rejects_negative_truncated_autocorrelation_time() {
    // The classical truncated estimator gives tau = -4/35 for this chain.
    // Previously this escaped as ESS = -52.5 and MCSE = NaN.
    let chains = [[-1.0, -1.0, -1.0, 1.0, -1.0, 0.0]];
    assert!(compute_effective_sample_size(&chains).is_err());
    assert!(compute_estimated_mcse(&chains).is_err());
}

#[test]
fn f32_subnormal_samples_keep_their_variation() {
    let unit = f32::from_bits(1);
    let chains = DRAWS.map(|chain| chain.map(|draw| draw as f32 * unit));
    assert_relative_eq!(
        compute_effective_sample_size(&chains).unwrap(),
        12.218181818181819,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        potential_scale_reduction_factor(&chains).unwrap(),
        (7.0_f64 / 8.0).sqrt(),
        max_relative = 1e-12
    );
    #[cfg(feature = "ndarray")]
    {
        let view = ndarray::ArrayView2::from(&chains);
        assert_relative_eq!(
            mcmc::ndarray::compute_effective_sample_size(view).unwrap(),
            12.218181818181819,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            mcmc::ndarray::compute_estimated_mcse(view).unwrap() / f64::from(unit),
            0.67700320038633,
            max_relative = 1e-12
        );
    }
}

#[test]
fn rhat_preserves_tiny_within_chain_variation_between_separated_chains() {
    for amplitude in [1e-100, 1e-200, 1e-300] {
        let chains = [
            [
                -amplitude, amplitude, -amplitude, amplitude, -amplitude, amplitude, -amplitude,
                amplitude,
            ],
            [1.0; 8],
        ];
        let rhat = potential_scale_reduction_factor(&chains).unwrap();
        assert!(rhat.is_finite());
        // W = (4/7) * amplitude^2, B/n = 1/2. The additive 7/8
        // term is below f64 precision at these amplitudes.
        assert_relative_eq!(
            rhat * amplitude,
            (7.0_f64 / 8.0).sqrt(),
            max_relative = 1e-12
        );
        let split = split_potential_scale_reduction_factor(&chains).unwrap();
        assert_relative_eq!(
            split * amplitude,
            (1.0_f64 / 2.0).sqrt(),
            max_relative = 1e-12
        );
    }
}

#[test]
fn rhat_reports_an_unrepresentable_ratio() {
    let tiny = f64::from_bits(1);
    let chains = [[-tiny, tiny, -tiny, tiny], [1.0; 4]];
    assert!(potential_scale_reduction_factor(&chains).is_err());
}

#[test]
fn constant_samples_are_rejected_at_every_scale() {
    for value in [0.0, f64::from_bits(1), 1e-300, 1.0, 1e300, f64::MAX] {
        let chains = [[value; 8]; 2];
        assert!(compute_effective_sample_size(&chains).is_err());
        assert!(compute_split_effective_sample_size(&chains).is_err());
        assert!(compute_estimated_mcse(&chains).is_err());
        assert!(potential_scale_reduction_factor(&chains).is_err());
        assert!(split_potential_scale_reduction_factor(&chains).is_err());
    }
}

#[test]
fn split_diagnostics_exclude_the_odd_middle_draw_from_normalization() {
    let chains = [
        [1.0, 3.0, 2.0, 5.0, f64::MAX, 4.0, 7.0, 6.0, 8.0],
        [2.0, 4.0, 1.0, 6.0, f64::NAN, 3.0, 8.0, 5.0, 7.0],
    ];
    assert_relative_eq!(
        compute_split_effective_sample_size(&chains).unwrap(),
        8.0,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        split_potential_scale_reduction_factor(&chains).unwrap(),
        1.2398524277321037,
        max_relative = 1e-12
    );
}
