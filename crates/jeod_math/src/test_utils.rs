use crate::types::{DMat3, DVec3};

pub fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

pub fn approx_eq_vec3(a: DVec3, b: DVec3, tol: f64) -> bool {
    (a - b).length() < tol
}

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
