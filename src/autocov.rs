/// Biased autocovariance at each lag, matching the Stan/arima normalization.
/// The caller validates at least four finite draws before reaching this kernel.
pub(crate) fn autocovariance(draws: impl ExactSizeIterator<Item = f64> + Clone) -> Vec<f64> {
    let n = draws.len();
    let mean = draws.clone().sum::<f64>() / n as f64;
    // Scale before multiplication: summing raw products can overflow even
    // when the normalized covariance is representable.
    let scale = (n as f64).sqrt();
    let centered: Vec<f64> = draws.map(|draw| (draw - mean) / scale).collect();
    (0..n)
        .map(|lag| dot(&centered[..n - lag], &centered[lag..]))
        .collect()
}

#[cfg(not(feature = "simd"))]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(feature = "simd")]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    use wide::f64x4;

    let mut sum = f64x4::ZERO;
    let mut chunks_a = a.chunks_exact(4);
    let mut chunks_b = b.chunks_exact(4);
    for (a, b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let a = f64x4::new([a[0], a[1], a[2], a[3]]);
        let b = f64x4::new([b[0], b[1], b[2], b[3]]);
        sum += a * b;
    }
    let tail: f64 = chunks_a
        .remainder()
        .iter()
        .zip(chunks_b.remainder())
        .map(|(a, b)| a * b)
        .sum();
    sum.reduce_add() + tail
}

#[cfg(test)]
mod tests {
    use super::autocovariance;
    use approx::assert_abs_diff_eq;
    use approx::assert_relative_eq;

    #[test]
    fn covariance_does_not_overflow_before_normalization() {
        let draws = [1e154, -1e154, 1e154, -1e154];
        let expected = arima::acf::acf(&draws, None, true).unwrap();
        for (actual, expected) in autocovariance(draws.iter().copied())
            .into_iter()
            .zip(expected)
        {
            assert!(actual.is_finite());
            assert_relative_eq!(actual, expected, max_relative = 1e-12);
        }
    }

    #[test]
    fn covariance_matches_arima_for_lane_boundaries_and_unaligned_inputs() {
        for n in (4..74).chain([128, 257, 1025]) {
            // Offset the start and vary the tail length to exercise unaligned
            // vector loads and every possible scalar remainder.
            let samples: Vec<f64> = (0..n + 1)
                .map(|i| (i as f64 * 1.37).sin() + (i as f64 * 0.19).cos())
                .collect();
            let draws = &samples[1..];
            let expected = arima::acf::acf(draws, None, true).unwrap();
            let actual = autocovariance(draws.iter().copied());
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn covariance_handles_constant_antithetic_and_offset_samples() {
        for draws in [
            vec![3.0; 33],
            (0..65)
                .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
                .collect(),
            (0..37).map(|i| 1e6 + f64::from(i) * 0.001).collect(),
        ] {
            let expected = arima::acf::acf(&draws, None, true).unwrap();
            let actual = autocovariance(draws.iter().copied());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
        }
    }
}
