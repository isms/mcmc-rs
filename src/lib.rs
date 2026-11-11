//! A Rust library implementing various MCMC diagnostics and utilities, such as Gelman Rubin
//! potential scale reduction factor (R hat), effective sample size, chain splitting,
//! and others.
//!
//! This crate is language agnostic and intended to work with the outputs of any MCMC sampler
//! (e.g. Stan, PyMC3, Turing.jl, etc.)

mod chains;
pub use chains::Sample;

/// Adapters for two-dimensional ndarray views with chains on axis 0 and draws on axis 1.
#[cfg(feature = "ndarray")]
pub mod ndarray;

/// Effective Sample Size (ESS)
pub mod ess;
/// Gelman-Rubin potential scale reduction (Rhat), including split Rhat
pub mod rhat;
/// Convenience utilities like chain splitting and certain helper functions
/// intended mostly for internal use to avoid external dependencies (e.g.
/// summary statistics and lightweight CSV reading)
pub mod utils;

/// One-dimensional vector of numeric values
pub type Array1 = Vec<f64>;
/// Two dimensional vector of vectors of numeric values
pub type Array2 = Vec<Array1>;
