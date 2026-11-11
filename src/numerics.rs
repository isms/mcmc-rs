use anyhow::{Result, anyhow};

/// A common affine transform for all retained draws of one parameter.
/// Centering before scaling preserves representable variation at large offsets;
/// bounded coordinates keep squared deviations away from overflow/underflow.
#[derive(Clone, Copy)]
pub(crate) struct DrawScale {
    pub(crate) center: f64,
    pub(crate) units: f64,
}

impl DrawScale {
    pub(crate) fn new<C: AsRef<[f64]>>(chains: &[C], n: usize) -> Result<Self> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for chain in chains {
            for &draw in &chain.as_ref()[..n] {
                if !draw.is_finite() {
                    return Err(anyhow!("All retained draws must be finite"));
                }
                min = min.min(draw);
                max = max.max(draw);
            }
        }
        if min == max {
            return Err(anyhow!(
                "Diagnostics are undefined for identical constant draws"
            ));
        }

        let scale = Self::from_bounds(min, max);
        if !scale.units.is_finite() || scale.units <= 0.0 {
            return Err(anyhow!("The retained draw range cannot be normalized"));
        }
        Ok(scale)
    }

    /// Local scaling for a nonempty chain already checked for finite draws.
    /// A constant chain has zero units and normalized values equal to zero.
    pub(crate) fn for_chain(draws: &[f64]) -> Self {
        let min = draws.iter().copied().fold(f64::INFINITY, f64::min);
        let max = draws.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::from_bounds(min, max)
    }

    fn from_bounds(min: f64, max: f64) -> Self {
        let span = max - min;
        let center = if span.is_finite() {
            min + span / 2.0
        } else {
            // Opposite finite extremes can have an infinite difference.
            min / 2.0 + max / 2.0
        };
        let units = (max - center).max(center - min);
        Self { center, units }
    }

    #[inline]
    pub(crate) fn normalize(self, draw: f64) -> f64 {
        if self.units == 0.0 {
            0.0
        } else {
            (draw - self.center) / self.units
        }
    }
}
