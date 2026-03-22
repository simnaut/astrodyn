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
