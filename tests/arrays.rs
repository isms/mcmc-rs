use approx::assert_abs_diff_eq;
use mcmc::ess::{
    compute_effective_sample_size, compute_estimated_mcse, compute_split_effective_sample_size,
};
use mcmc::rhat::{potential_scale_reduction_factor, split_potential_scale_reduction_factor};
use mcmc::utils::split_chain_views;

const DRAWS: [[f64; 9]; 2] = [
    [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0, 0.0],
    [2.0, 4.0, 1.0, 6.0, 3.0, 8.0, 5.0, 7.0, 0.0],
];

fn expected() -> [f64; 5] {
    let chains = DRAWS.map(Vec::from).to_vec();
    [
        compute_effective_sample_size(&chains).unwrap(),
        compute_split_effective_sample_size(&chains).unwrap(),
        compute_estimated_mcse(&chains).unwrap(),
        potential_scale_reduction_factor(&chains).unwrap(),
        split_potential_scale_reduction_factor(&chains).unwrap(),
    ]
}

#[test]
fn native_arrays_and_borrowed_slices_match_vector_inputs() {
    let slices: Vec<&[f64]> = DRAWS.iter().map(|chain| &chain[..]).collect();
    let actual = [
        compute_effective_sample_size(&DRAWS).unwrap(),
        compute_split_effective_sample_size(&slices).unwrap(),
        compute_estimated_mcse(&DRAWS[..]).unwrap(),
        potential_scale_reduction_factor(&slices[..]).unwrap(),
        split_potential_scale_reduction_factor(&DRAWS).unwrap(),
    ];
    for (actual, expected) in actual.into_iter().zip(expected()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn f32_inputs_produce_f64_diagnostics() {
    let chains = DRAWS.map(|chain| chain.map(|draw| draw as f32));
    let actual: [f64; 5] = [
        compute_effective_sample_size(&chains).unwrap(),
        compute_split_effective_sample_size(&chains).unwrap(),
        compute_estimated_mcse(&chains).unwrap(),
        potential_scale_reduction_factor(&chains).unwrap(),
        split_potential_scale_reduction_factor(&chains).unwrap(),
    ];
    for (actual, expected) in actual.into_iter().zip(expected()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn split_views_borrow_trimmed_halves_and_drop_the_middle_draw() {
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let b = [6.0, 7.0, 8.0, 9.0, 10.0, f32::NAN];
    let chains: [&[f32]; 2] = [&a, &b];
    let split = split_chain_views(&chains).unwrap();
    assert_eq!(split, [&a[..2], &a[3..], &b[..2], &b[3..5]]);
    assert!(std::ptr::eq(split[0].as_ptr(), a.as_ptr()));
    assert!(std::ptr::eq(split[3].as_ptr(), b[3..].as_ptr()));
    let empty: [&[f64]; 0] = [];
    assert!(split_chain_views(&empty).is_err());
    assert!(split_chain_views(&[[1.0]]).is_err());
}

#[test]
fn f32_inputs_validate_only_the_retained_prefix() {
    let mut longer = DRAWS[0].map(|x| x as f32).to_vec();
    longer.extend([f32::NAN, f32::INFINITY]);
    let shorter = DRAWS[1].map(|x| x as f32);
    let chains: [&[f32]; 2] = [&longer, &shorter];
    assert_abs_diff_eq!(
        compute_effective_sample_size(&chains).unwrap(),
        expected()[0],
        epsilon = 1e-12
    );
    longer[0] = f32::NAN;
    let chains: [&[f32]; 2] = [&longer, &shorter];
    assert!(compute_effective_sample_size(&chains).is_err());
    assert!(potential_scale_reduction_factor(&chains).is_err());
}

#[cfg(feature = "ndarray")]
mod ndarray_tests {
    use super::*;
    use ndarray::{Array2, ArrayView2, ShapeBuilder, s};

    fn check(view: ArrayView2<'_, f64>, expected: [f64; 5]) {
        let actual = [
            mcmc::ndarray::compute_effective_sample_size(view).unwrap(),
            mcmc::ndarray::compute_split_effective_sample_size(view).unwrap(),
            mcmc::ndarray::compute_estimated_mcse(view).unwrap(),
            mcmc::ndarray::potential_scale_reduction_factor(view).unwrap(),
            mcmc::ndarray::split_potential_scale_reduction_factor(view).unwrap(),
        ];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn contiguous_fortran_transposed_and_strided_arrays_match_vectors() {
        let rows = ArrayView2::from(&DRAWS);
        check(rows, expected());
        let fortran = Array2::from_shape_fn((2, 9).f(), |(chain, draw)| DRAWS[chain][draw]);
        check(fortran.view(), expected());
        let transposed = Array2::from_shape_fn((9, 2), |(draw, chain)| DRAWS[chain][draw]);
        check(transposed.t(), expected());
        let padded = Array2::from_shape_fn((2, 18), |(chain, draw)| {
            if draw % 2 == 0 {
                DRAWS[chain][draw / 2]
            } else {
                f64::NAN
            }
        });
        check(padded.slice(s![.., ..;2]), expected());
        let reversed: Vec<Vec<f64>> = DRAWS
            .iter()
            .map(|c| c.iter().rev().copied().collect())
            .collect();
        let reverse_expected = [
            compute_effective_sample_size(&reversed).unwrap(),
            compute_split_effective_sample_size(&reversed).unwrap(),
            compute_estimated_mcse(&reversed).unwrap(),
            potential_scale_reduction_factor(&reversed).unwrap(),
            split_potential_scale_reduction_factor(&reversed).unwrap(),
        ];
        check(rows.slice(s![.., ..;-1]), reverse_expected);
    }

    #[test]
    fn f32_ndarray_inputs_match_native_arrays() {
        let chains = Array2::from_shape_fn((2, 9), |(chain, draw)| DRAWS[chain][draw] as f32);
        let actual: [f64; 5] = [
            mcmc::ndarray::compute_effective_sample_size(chains.view()).unwrap(),
            mcmc::ndarray::compute_split_effective_sample_size(chains.view()).unwrap(),
            mcmc::ndarray::compute_estimated_mcse(chains.view()).unwrap(),
            mcmc::ndarray::potential_scale_reduction_factor(chains.view()).unwrap(),
            mcmc::ndarray::split_potential_scale_reduction_factor(chains.view()).unwrap(),
        ];
        for (actual, expected) in actual.into_iter().zip(expected()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn split_ndarray_views_preserve_strides_and_borrow_samples() {
        let rows = ArrayView2::from(&DRAWS);
        let reversed = rows.slice(s![.., ..;-1]);
        let split = mcmc::ndarray::split_chain_views(reversed).unwrap();
        assert_eq!(split.len(), 4);
        assert_eq!(split[0].to_vec(), [0.0, 8.0, 6.0, 7.0]);
        assert_eq!(split[1].to_vec(), [5.0, 2.0, 3.0, 1.0]);
        assert_eq!(split[2].to_vec(), [0.0, 7.0, 5.0, 8.0]);
        assert_eq!(split[3].to_vec(), [6.0, 1.0, 4.0, 2.0]);
        assert!(std::ptr::eq(split[0].as_ptr(), &DRAWS[0][8]));
        assert!(std::ptr::eq(split[3].as_ptr(), &DRAWS[1][3]));
        for shape in [(0, 9), (2, 0), (2, 1)] {
            let empty = Array2::<f64>::zeros(shape);
            assert!(mcmc::ndarray::split_chain_views(empty.view()).is_err());
        }
    }

    #[test]
    fn invalid_ndarray_shapes_and_values_return_errors() {
        for shape in [(0, 9), (2, 0), (2, 1)] {
            let empty = Array2::<f64>::zeros(shape);
            assert!(mcmc::ndarray::compute_effective_sample_size(empty.view()).is_err());
            assert!(mcmc::ndarray::compute_split_effective_sample_size(empty.view()).is_err());
            assert!(mcmc::ndarray::compute_estimated_mcse(empty.view()).is_err());
            assert!(mcmc::ndarray::potential_scale_reduction_factor(empty.view()).is_err());
            assert!(mcmc::ndarray::split_potential_scale_reduction_factor(empty.view()).is_err());
        }
        let mut chains = Array2::from_shape_fn((2, 9), |(chain, draw)| DRAWS[chain][draw]);
        chains[[0, 2]] = f64::INFINITY;
        assert!(mcmc::ndarray::compute_effective_sample_size(chains.view()).is_err());
        assert!(mcmc::ndarray::potential_scale_reduction_factor(chains.view()).is_err());
    }
}
