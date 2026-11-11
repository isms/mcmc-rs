//! Optional adapters for [`::ndarray::ArrayView2`].
//!
//! Enable the `ndarray` Cargo feature. Axis 0 contains chains and axis 1 contains
//! draws. Contiguous `f64` rows are borrowed; strided rows and `f32` rows are
//! copied into contiguous `f64` buffers. All diagnostics return `f64`.
//!
//! ```
//! use ndarray::array;
//! let draws = array![[1.0_f32, 3.0, 2.0, 4.0], [2.0, 4.0, 1.0, 3.0]];
//! let rhat = mcmc::ndarray::potential_scale_reduction_factor(draws.view())?;
//! assert!(rhat.is_finite());
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Sample, ess, rhat};
use ::ndarray::{ArrayView1, ArrayView2, s};
use anyhow::{Result, anyhow};
use std::borrow::Cow;

fn prepare_rows<T: Sample>(chains: ArrayView2<'_, T>) -> Vec<Cow<'_, [f64]>> {
    chains
        .into_outer_iter()
        .map(|row| match row.to_slice() {
            Some(slice) => T::as_f64_slice(slice),
            None => Cow::Owned(row.iter().map(|&sample| sample.into()).collect()),
        })
        .collect()
}

/// Split each chain into two borrowed views, preserving its strides.
///
/// Requires at least one chain and two draws. For odd lengths, the middle
/// draw is dropped. Only the vector of views is allocated; samples are borrowed
/// even for noncontiguous inputs. The element type is preserved.
pub fn split_chain_views<T>(chains: ArrayView2<'_, T>) -> Result<Vec<ArrayView1<'_, T>>> {
    if chains.nrows() == 0 || chains.ncols() < 2 {
        return Err(anyhow!(
            "Splitting requires at least one chain with two draws"
        ));
    }
    let half = chains.ncols() / 2;
    let second_start = half + chains.ncols() % 2;
    let mut split = Vec::with_capacity(chains.nrows() * 2);
    for row in chains.into_outer_iter() {
        split.push(row.slice_move(s![..half]));
        split.push(row.slice_move(s![second_start..]));
    }
    Ok(split)
}

/// Compute classical ESS. See [`ess::compute_effective_sample_size`].
pub fn compute_effective_sample_size<T: Sample>(chains: ArrayView2<'_, T>) -> Result<f64> {
    ess::compute_effective_sample_size_impl(&prepare_rows(chains))
}

/// Compute split ESS. See [`ess::compute_split_effective_sample_size`].
pub fn compute_split_effective_sample_size<T: Sample>(chains: ArrayView2<'_, T>) -> Result<f64> {
    ess::compute_split_effective_sample_size_impl(&prepare_rows(chains))
}

/// Compute MCSE of the mean. See [`ess::compute_estimated_mcse`].
pub fn compute_estimated_mcse<T: Sample>(chains: ArrayView2<'_, T>) -> Result<f64> {
    ess::compute_estimated_mcse_impl(&prepare_rows(chains))
}

/// Compute classical Rhat. See [`rhat::potential_scale_reduction_factor`].
pub fn potential_scale_reduction_factor<T: Sample>(chains: ArrayView2<'_, T>) -> Result<f64> {
    rhat::potential_scale_reduction_factor_impl(&prepare_rows(chains))
}

/// Compute split Rhat. See [`rhat::split_potential_scale_reduction_factor`].
pub fn split_potential_scale_reduction_factor<T: Sample>(chains: ArrayView2<'_, T>) -> Result<f64> {
    rhat::split_potential_scale_reduction_factor_impl(&prepare_rows(chains))
}
