use crate::Array2;
use anyhow::{Error, Result, anyhow};
use average::Mean;
use average::Variance;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

/// Compute the arithmetic mean of an array.
pub(crate) fn mean(arr: &[f64]) -> Result<f64, Error> {
    if arr.is_empty() {
        return Err(anyhow!("Can't take mean of empty array"));
    }
    let mean = arr.iter().collect::<Mean>();
    Ok(mean.mean())
}

/// Compute the sample variance of an array using Bessel's correction.
pub(crate) fn sample_variance(arr: &[f64]) -> Result<f64, Error> {
    if arr.len() < 2 {
        return Err(anyhow!("Sample variance requires at least two values"));
    }
    let var: Variance = arr.iter().collect();
    Ok(var.sample_variance())
}

/// Find the common prefix length, rejecting an empty collection of chains.
pub(crate) fn shortest_chain_len<T, C: AsRef<[T]>>(chains: &[C]) -> Result<usize> {
    chains
        .iter()
        .map(|chain| chain.as_ref().len())
        .min()
        .ok_or_else(|| anyhow!("Must provide at least one chain"))
}

/// Splits each chain into two chains of equal length.  When the
/// number of total draws N is odd, the (N+1)/2th draw is ignored.
///
/// See more details in Stan reference manual section
/// ["Effective Sample Size"](http://mc-stan.org/users/documentation).
///
/// Chains are trimmed from the back to match the shortest chain.
/// Each chain must contain at least two draws.
pub fn split_chains(chains: Array2) -> Result<Array2, Error> {
    Ok(split_chain_views(&chains)?
        .into_iter()
        .map(<[f64]>::to_vec)
        .collect())
}

/// Split chains into borrowed halves without copying the draws.
///
/// Accepts vectors, slices, and fixed-size arrays of any element type. Chains
/// are trimmed to the shortest length, which must be at least two. An odd
/// length drops the middle draw. Only the vector of slice references is allocated.
///
/// ```
/// let chains = [[1.0_f32, 2.0, 3.0, 4.0, 5.0]];
/// let halves = mcmc::utils::split_chain_views(&chains)?;
/// assert_eq!(halves, [&chains[0][..2], &chains[0][3..]]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn split_chain_views<T, C: AsRef<[T]>>(chains: &[C]) -> Result<Vec<&[T]>, Error> {
    let num_draws = shortest_chain_len(chains)?;
    if num_draws < 2 {
        return Err(anyhow!("Must have at least two samples to split"));
    }
    let (half, offset) = if num_draws % 2 == 0 {
        (num_draws / 2, 0)
    } else {
        ((num_draws - 1) / 2, 1)
    };
    let mut split_draws = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        let chain = chain.as_ref();
        split_draws.push(&chain[..half]);
        split_draws.push(&chain[(half + offset)..num_draws]);
    }
    Ok(split_draws)
}

/// Simplified CSV reader for testing purposes only; does not actually implement
/// parsing for headers, quotation, or other more advanced features. Assumes
/// that all values aside from the commas will be numeric.
///
/// # Arguments
/// * `skip_rows` - Number of rows to skip before numeric values. For example,
///   if there is a header row you can pass in the value `1`.
/// * `n_rows` - Number of rows to read in. Use if you only want a certain
///   subset of rows or if there are improper rows after the numeric
///   rows (e.g. in Stan sample files there are commented rows at the end).
pub fn read_csv(path: &PathBuf, skip_rows: usize, n_rows: usize) -> Array2 {
    let mut result: Array2 = Vec::new();
    let f = File::open(path).unwrap();
    let f = BufReader::new(f);
    for line in f.lines().skip(skip_rows).take(n_rows).flatten() {
        for (idx, value) in line.split(',').enumerate() {
            if idx >= result.len() {
                result.push(Vec::new())
            }
            result[idx].push(value.parse::<f64>().unwrap());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array1;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_stats() {
        // Test our basic stats functions using numbers computed with numpy.
        let arr = vec![
            2.13829088,
            -1.06214379,
            -0.79265699,
            -0.21300888,
            -1.07155142,
            -0.50425317,
            0.95708854,
            -1.23854172,
            1.37124938,
            1.17658286,
        ];
        let empty: Array1 = vec![];
        assert_abs_diff_eq!(
            sample_variance(&arr).unwrap(),
            1.492596054209826,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(mean(&arr).unwrap(), 0.07610557018217139, epsilon = 1e-6);

        assert!(sample_variance(&empty).is_err());
        assert!(mean(&empty).is_err());
    }

    #[test]
    fn test_split_empty_chains() {
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
    }

    #[test]
    fn test_split_even_chains() {
        // Regular split with even numbers
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let chains = vec![a, b];
        let split = split_chains(chains).unwrap();
        assert_eq!(split[0], vec![1.0, 2.0]);
        assert_eq!(split[1], vec![3.0, 4.0]);
        assert_eq!(split[2], vec![5.0, 6.0]);
        assert_eq!(split[3], vec![7.0, 8.0]);
    }

    #[test]
    fn test_split_odd_chains() {
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
}
