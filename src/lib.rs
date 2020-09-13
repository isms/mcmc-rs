#[macro_use]
extern crate approx;

pub mod ess;
pub mod rhat;
pub mod utils;

pub type Array1 = Vec<f64>;
pub type Array2 = Vec<Array1>;
