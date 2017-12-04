#![feature(cfg_target_feature, target_feature)]

extern crate num;
#[macro_use]
extern crate stdsimd;

use num::Num;
use stdsimd::vendor;
use stdsimd::simd::f32x4;

pub const EPSILON: f64 = 1e-9;

macro_rules! assert_eq_within_bound {
    ($x:expr, $y: expr, $bound: expr) => (
        assert!(
            $x >= $y - $bound && $x <= $y + $bound,
            "{} is not equal to {} within bound {}",
            $x, $y, $bound
        );
    );
}

macro_rules! assert_eq_vector3 {
    ($x:expr, $y: expr, $bound: expr) => (
        assert_eq_within_bound!($x.x, $y.x, $bound);
        assert_eq_within_bound!($x.y, $y.y, $bound);
        assert_eq_within_bound!($x.z, $y.z, $bound);
    );
}

#[derive(Copy, Clone)]
pub struct Vector3<T: Num + Copy> {
    x: T,
    y: T,
    z: T,
}

impl<T: Num + Copy> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x: x, y: y, z: z }
    }

    pub fn dot(&self, other: &Vector3<T>) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+sse4.1"]
#[inline(always)]
pub unsafe fn dot_sse(a: f32x4, b: f32x4) -> f32 {
    vendor::_mm_dp_ps(a, b, 0x71).extract(0)
}

pub fn dot_f32_sse(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let a = f32x4::new(a.x, a.y, a.z, 0.0);
    let b = f32x4::new(b.x, b.y, b.z, 0.0);
    unsafe { dot_sse(a, b) }
}

#[no_mangle]
pub type Vector = Vector3<f32>;

#[cfg(test)]
mod tests {
    use super::{Vector3, EPSILON, dot_f32_sse};

    #[test]
    fn test_f32_dot() {
        let vec1: Vector3<f32> = Vector3::new(3.52, 8.23, 29.0);
        let vec2: Vector3<f32> = Vector3::new(0.0, 1.3, -3.23);

        assert_eq_within_bound!(vec1.dot(&vec2), -82.971, (EPSILON as f32));
        assert_eq_within_bound!(vec2.dot(&vec1), -82.971, (EPSILON as f32));
    }

    #[test]
    fn test_f32_dot_sse() {
        let vec1: Vector3<f32> = Vector3::new(3.52, 8.23, 29.0);
        let vec2: Vector3<f32> = Vector3::new(0.0, 1.3, -3.23);

        assert_eq_within_bound!(dot_f32_sse(&vec1, &vec2), -82.971, (EPSILON as f32));
        assert_eq_within_bound!(dot_f32_sse(&vec2, &vec1), -82.971, (EPSILON as f32));
    }

    #[test]
    fn test_f64_dot() {
        let vec1: Vector3<f64> = Vector3::new(3.52, 8.23, 29.0);
        let vec2: Vector3<f64> = Vector3::new(0.0, 1.3, -3.23);

        assert_eq_within_bound!(vec1.dot(&vec2), -82.971, EPSILON);
        assert_eq_within_bound!(vec2.dot(&vec1), -82.971, EPSILON);
    }
}
