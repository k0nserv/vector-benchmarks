# Vector benchmarks

**Note: As show by [Cameron Hart](https://twitter.com/bitshifternz) these results are [not correct](https://bitshifter.github.io/blog/2017/12/04/rust-bench-simd/)**

This branch contains an attempt to fool the optimizer.
Use `RUSTFLAGS="-C target-cpu=native -C target-feature=+sse4.1 --emit asm" cargo bench --no-run` to get the compiled output

This project contains research exploring SIMD instructions in Rust, specifically SSE, to speed up the computation of vector dot products for use in 3D graphics. From the benchmarks demonstrated here we can conclude that Rust's inability to inline functions that use SIMD makes the option drastically slower than the naive implementation. Notably even when inlining the SIMD version it is not faster than the naive option.

Result on Macbook Pro 13"(mid 2012) with Intel Core i5 2.5Ghz

```
running 4 tests
test bench_f32            ... bench:         315 ns/iter (+/- 36)
test bench_f32_sse        ... bench: 218,383,192 ns/iter (+/- 6,526,505)
test bench_f32_sse_inline ... bench:         312 ns/iter (+/- 58)
test bench_f64            ... bench:         315 ns/iter (+/- 90)
```

+ `bench_f32` uses `Vector<f32>`.
+ `bench_f32_sse` uses `f32` SIMD instructions without inlining.
+ `bench_f32_sse_inlin` uses `f32` SIMD instructions that have been manually inlined.
+ `bench_f64` uses `Vector<f64>`.
