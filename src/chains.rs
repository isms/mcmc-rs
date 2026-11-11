use crate::utils::shortest_chain_len;
use anyhow::Result;
use std::borrow::Cow;

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A sample stored as `f32` or `f64`.
///
/// Diagnostics accumulate and return `f64`. `f64` sample buffers are borrowed;
/// `f32` buffers are converted to `f64` once per retained chain.
/// This trait is sealed to these two types.
pub trait Sample: private::Sealed + Copy + Into<f64> + 'static {
    #[doc(hidden)]
    fn as_f64_slice(samples: &[Self]) -> Cow<'_, [f64]>;
}

impl Sample for f64 {
    fn as_f64_slice(samples: &[Self]) -> Cow<'_, [f64]> {
        Cow::Borrowed(samples)
    }
}

impl Sample for f32 {
    fn as_f64_slice(samples: &[Self]) -> Cow<'_, [f64]> {
        Cow::Owned(samples.iter().map(|&sample| f64::from(sample)).collect())
    }
}

pub(crate) fn prepare_chains<T: Sample, C: AsRef<[T]>>(
    chains: &[C],
) -> Result<Vec<Cow<'_, [f64]>>> {
    let n = shortest_chain_len(chains)?;
    Ok(chains
        .iter()
        .map(|chain| T::as_f64_slice(&chain.as_ref()[..n]))
        .collect())
}
