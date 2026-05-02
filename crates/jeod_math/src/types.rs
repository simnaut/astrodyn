//! Vector / matrix helpers ported from
//! [`models/utils/math/include/vector3_inline.hh`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/math/include/vector3_inline.hh)
//! and
//! [`models/utils/math/include/matrix3x3_inline.hh`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/math/include/matrix3x3_inline.hh)
//! in JEOD v5.4.0.
//!
//! JEOD source is row-major while `glam` is column-major; the
//! [`mat3_from_rows`] helper lets a port read top-to-bottom against the
//! C++ source and still produce a correct `glam::DMat3`. The other
//! helpers wrap the inline `Vector3::*` / `Matrix3x3::*` operators so
//! kernel code can name them at the JEOD convention.

pub use glam::{DMat3, DQuat, DVec3};

/// Construct a `DMat3` from three row vectors.
///
/// glam stores matrices column-major, but JEOD formulae are typically expressed
/// in row-major order (T\[row\]\[col\]).  This helper lets us write the matrix the
/// way it appears in JEOD source and still get a correct glam representation.
pub fn mat3_from_rows(r0: DVec3, r1: DVec3, r2: DVec3) -> DMat3 {
    DMat3::from_cols(
        DVec3::new(r0.x, r1.x, r2.x),
        DVec3::new(r0.y, r1.y, r2.y),
        DVec3::new(r0.z, r1.z, r2.z),
    )
}

// ── JEOD Vector3 operations ─────────────────────────────────────────────
// Port of `models/utils/math/include/vector3_inline.hh`

/// `prod = tmat * vec`
///
/// Port of JEOD `Vector3::transform(tmat, vec, prod)`.
#[inline]
pub fn vector3_transform(tmat: &DMat3, vec: DVec3) -> DVec3 {
    *tmat * vec
}

/// `prod = tmat^T * vec`
///
/// Port of JEOD `Vector3::transform_transpose(tmat, vec, prod)`.
#[inline]
pub fn vector3_transform_transpose(tmat: &DMat3, vec: DVec3) -> DVec3 {
    tmat.transpose() * vec
}

// ── JEOD Matrix3x3 operations ───────────────────────────────────────────
// Port of `models/utils/math/include/matrix3x3_inline.hh`

/// `prod = trans * mat * trans^T`
///
/// Port of JEOD `Matrix3x3::transform_matrix(trans, mat, prod)`.
/// Similarity transformation: transforms `mat` from the "trans" frame.
#[inline]
pub fn matrix3x3_transform_matrix(trans: &DMat3, mat: &DMat3) -> DMat3 {
    *trans * *mat * trans.transpose()
}

/// `prod = trans^T * mat * trans`
///
/// Port of JEOD `Matrix3x3::transpose_transform_matrix(trans, mat, prod)`.
/// Inverse similarity transformation: transforms `mat` out of the "trans" frame.
#[inline]
pub fn matrix3x3_transpose_transform_matrix(trans: &DMat3, mat: &DMat3) -> DMat3 {
    trans.transpose() * *mat * *trans
}

/// `prod = mat_left^T * mat_right^T`
///
/// Port of JEOD `Matrix3x3::product_transpose_transpose(mat_left, mat_right, prod)`.
#[inline]
pub fn matrix3x3_product_transpose_transpose(mat_left: &DMat3, mat_right: &DMat3) -> DMat3 {
    mat_left.transpose() * mat_right.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat3_from_rows_identity() {
        let m = mat3_from_rows(
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(m, DMat3::IDENTITY);
    }

    #[test]
    fn mat3_from_rows_element_access() {
        // Build a matrix where T[i][j] = 10*i + j  (row i, col j)
        let m = mat3_from_rows(
            DVec3::new(0.0, 1.0, 2.0),
            DVec3::new(10.0, 11.0, 12.0),
            DVec3::new(20.0, 21.0, 22.0),
        );
        // glam: col(j)[i] == T[i][j]
        assert_eq!(m.col(0)[0], 0.0);
        assert_eq!(m.col(1)[0], 1.0);
        assert_eq!(m.col(2)[0], 2.0);
        assert_eq!(m.col(0)[1], 10.0);
        assert_eq!(m.col(1)[1], 11.0);
        assert_eq!(m.col(2)[1], 12.0);
        assert_eq!(m.col(0)[2], 20.0);
        assert_eq!(m.col(1)[2], 21.0);
        assert_eq!(m.col(2)[2], 22.0);
    }
}
