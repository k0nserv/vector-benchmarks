extern crate vector_benchmarks;
#[macro_use]
extern crate bencher;
#[macro_use]
extern crate stdsimd;
extern crate rand;
extern crate num;

use bencher::{Bencher, black_box};
use std::env;
use stdsimd::vendor;
use stdsimd::simd::f32x4;
use num::Num;

use rand::{Rng, Rand};

use vector_benchmarks::{Vector3, dot_f32_sse, dot_sse};

const DEFAULT_NUM_CALCULATIONS: u64 = 1_000_0;

fn num_calculations() -> u64 {
    match env::var("NUM_CALCULATIONS") {
        Ok(iterations) => iterations.parse::<u64>().unwrap_or(DEFAULT_NUM_CALCULATIONS),
        Err(_) => DEFAULT_NUM_CALCULATIONS,
    }
}

fn build_vector_data<T: Rand + Copy + Num>(count: usize) -> Vec<(Vector3<T>, Vector3<T>)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(count);

    for _ in 0..count {
        let tuple = rng.gen::<(T, T, T, T, T, T)>();
        data.push((Vector3::new(tuple.0, tuple.1, tuple.2),
                   Vector3::new(tuple.3, tuple.4, tuple.4)));
    }

    data
}

fn build_simd_vec_data(count: usize) -> Vec<(f32x4, f32x4)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(count);

    for _ in 0..count {
        let tuple = rng.gen::<(f32, f32, f32, f32, f32, f32)>();
        data.push((f32x4::new(tuple.0, tuple.1, tuple.2, 0.0),
                   f32x4::new(tuple.3, tuple.4, tuple.4, 0.0)));
    }

    data
}

fn bench_f32(b: &mut Bencher) {
    let data = build_vector_data::<f32>(num_calculations() as usize);
    b.iter(|| {
               let d = black_box(&data);

               d.iter().fold(0.0, |acc, &(v1, v2)| acc + v1.dot(&v2))
           });
}

fn bench_f32_sse(b: &mut Bencher) {
    let data = build_simd_vec_data(num_calculations() as usize);
    b.iter(|| {
               let d = black_box(&data);

               d.iter().fold(0.0, |acc, &(v1, v2)| unsafe { acc + dot_sse(v1, v2) })
           });
}

fn bench_f32_sse_inline(b: &mut Bencher) {
    let data = build_simd_vec_data(num_calculations() as usize);
    b.iter(|| {
        let d = black_box(&data);

        d.iter().fold(0.0,
                      |acc, &(v1, v2)| unsafe { acc + vendor::_mm_dp_ps(v1, v2, 0x71).extract(0) })
    });
}

fn bench_f64(b: &mut Bencher) {
    let data = build_vector_data::<f64>(num_calculations() as usize);
    b.iter(|| {
               let d = black_box(&data);

               d.iter().fold(0.0, |acc, &(v1, v2)| acc + v1.dot(&v2))
           });
}

benchmark_group!(benches,
                 bench_f32,
                 bench_f64,
                 bench_f32_sse_inline,
                 bench_f32_sse);
benchmark_main!(benches);
