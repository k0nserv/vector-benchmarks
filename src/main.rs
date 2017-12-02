#![feature(cfg_target_feature, target_feature)]
#[macro_use]
extern crate stdsimd;
extern crate vector_benchmarks;
use stdsimd::vendor;
use stdsimd::simd::f32x4;

use vector_benchmarks::{Vector3, dot_f32_sse, dot_sse};
const DEFAULT_NUM_ITERATIONS: u64 = 1_000_000_000_0;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+sse4.1"]
fn main() {
    let a = f32x4::new(23.2, 39.1, 21.0, 0.0);
    let b = f32x4::new(-5.2, 0.1, 13.4, 0.0);
    let v1: Vector3<f32> = Vector3::new(23.2, 39.1, 21.0);
    let v2: Vector3<f32> = Vector3::new(-5.2, 0.1, 13.4);

    let mut x = 0.0;

    for _ in 0..DEFAULT_NUM_ITERATIONS {
        x += unsafe { dot_sse(a, b) };
        // unsafe { x += vendor::_mm_dp_ps(a, b, 0x71).extract(0) }
        // x += v1.dot_copy(v2);
        x += v1.dot(&v2);
    }

    println!("Done {}", x);
}
