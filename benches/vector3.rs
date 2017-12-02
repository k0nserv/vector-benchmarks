extern crate vector_benchmarks;
#[macro_use]
extern crate bencher;
#[macro_use]
extern crate stdsimd;

use bencher::Bencher;
use std::env;
use stdsimd::vendor;
use stdsimd::simd::f32x4;

use vector_benchmarks::{Vector3, dot_f32_sse, dot_sse};

const DEFAULT_NUM_ITERATIONS: u64 = 1_000_000_00;

fn num_iterations() -> u64 {
    match env::var("NUM_ITERATIONS") {
        Ok(iterations) => iterations.parse::<u64>().unwrap_or(DEFAULT_NUM_ITERATIONS),
        Err(_) => DEFAULT_NUM_ITERATIONS,
    }
}

fn bench_f32(b: &mut Bencher) {
    b.iter(|| {
               let a: Vector3<f32> = Vector3::new(23.2, 39.1, 21.0);
               let b: Vector3<f32> = Vector3::new(-5.2, 0.1, 13.4);

               (0..num_iterations()).fold(0.0, |acc, i| acc + a.dot(&b));
           });
}

fn bench_f32_sse(b: &mut Bencher) {
    b.iter(|| {
               let a = f32x4::new(23.2, 39.1, 21.0, 0.0);
               let b = f32x4::new(-5.2, 0.1, 13.4, 0.0);

               (0..num_iterations()).fold(0.0, |acc, i| unsafe { acc + dot_sse(a, b) });
           });
}

fn bench_f32_sse_inline(b: &mut Bencher) {
    b.iter(|| {
        let a = f32x4::new(23.2, 39.1, 21.0, 0.0);
        let b = f32x4::new(-5.2, 0.1, 13.4, 0.0);

        (0..num_iterations()).fold(0.0, |acc, i| unsafe {
            acc + vendor::_mm_dp_ps(a, b, 0x71).extract(0)
        });
    });
}

fn bench_f64(b: &mut Bencher) {
    b.iter(|| {
               let a: Vector3<f64> = Vector3::new(23.2, 39.1, 21.0);
               let b: Vector3<f64> = Vector3::new(-5.2, 0.1, 13.4);

               (0..num_iterations()).fold(0.0, |acc, i| acc + a.dot(&b));
           });
}

benchmark_group!(benches,
                 bench_f32,
                 bench_f64,
                 bench_f32_sse,
                 bench_f32_sse_inline);
benchmark_main!(benches);
