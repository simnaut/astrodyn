//! Approximate-equality helpers for unit and Tier-2 tests.
//!
//! Gated behind the `test-utils` Cargo feature so downstream crates
//! (e.g., [`astrodyn_dynamics`](https://docs.rs/astrodyn_dynamics)) can reuse
//! the same tolerance helpers without the production build paying for
//! them. Mirrors the comparison helpers used in JEOD's verification
//! sims under `models/utils/math/verif/`.

use crate::types::{DMat3, DVec3};

/// Returns `true` if `|a - b| < tol`.
pub fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Returns `true` if the L2 norm of `a - b` is less than `tol`.
pub fn approx_eq_vec3(a: DVec3, b: DVec3, tol: f64) -> bool {
    (a - b).length() < tol
}

/// Returns `true` if every element of `a - b` is within `tol`
/// (componentwise infinity norm).
pub fn approx_eq_mat3(a: &DMat3, b: &DMat3, tol: f64) -> bool {
    for c in 0..3 {
        for r in 0..3 {
            if (a.col(c)[r] - b.col(c)[r]).abs() > tol {
                return false;
            }
        }
    }
    true
}
