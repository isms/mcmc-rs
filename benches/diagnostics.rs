//! Run with `cargo bench --bench diagnostics`, then repeat with `--features simd`.
use mcmc::ess::compute_effective_sample_size;
use mcmc::rhat::potential_scale_reduction_factor;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn measure(mut diagnostic: impl FnMut() -> f64) -> f64 {
    let mut durations = Vec::with_capacity(7);
    for _ in 0..7 {
        let start = Instant::now();
        let mut iterations = 0;
        while start.elapsed() < Duration::from_millis(100) {
            black_box(diagnostic());
            iterations += 1;
        }
        durations.push(start.elapsed().as_secs_f64() / f64::from(iterations));
    }
    durations.sort_by(f64::total_cmp);
    durations[durations.len() / 2] * 1e6
}

fn main() {
    if cfg!(debug_assertions) || std::env::args().any(|arg| arg == "--test") {
        return;
    }
    println!("Four chains; median microseconds per call over seven 100 ms batches");
    println!("SIMD enabled: {}", cfg!(feature = "simd"));
    for n in [256, 1024, 4096] {
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|chain| {
                (0..n)
                    .map(|i| {
                        let x = f64::from(i * 17 + chain * 7);
                        x.sin() + (x * 0.13).cos()
                    })
                    .collect()
            })
            .collect();
        let ess = measure(|| compute_effective_sample_size(black_box(&chains)).unwrap());
        let rhat = measure(|| potential_scale_reduction_factor(black_box(&chains)).unwrap());
        println!("draws/chain={n:4} ESS={ess:10.2} us Rhat={rhat:8.2} us");
    }
}
