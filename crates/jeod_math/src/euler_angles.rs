//! Euler angle decomposition and composition.
//!
//! Faithful port of JEOD `models/utils/orientation/src/euler_angles.cc`.
//! Supports all 12 Euler rotation sequences (6 Tait-Bryan / aerodynamic,
//! 6 proper Euler / astronomical).
//!
//! Two parallel surfaces are exposed:
//! * the historical bare-`f64` entry points (radians, deprecated), and
//! * the typed sibling APIs that accept/return `uom::si::f64::Angle` and
//!   hand out a [`NormalizedQuat`] witness for the quaternion builder so
//!   downstream code can bind the unit-norm precondition at the type level.
//!
//! The two surfaces share their numeric implementation, so the typed
//! variants are bit-identical to the bare-`f64` ones at equal inputs.

use crate::quaternion::JeodQuat;
use crate::types::DMat3;
use jeod_quantities::quat::{LeftTransform, NormalizedQuat, ScalarFirst};
use std::f64::consts::{FRAC_PI_2, PI};
use uom::si::angle::radian;
use uom::si::f64::Angle;

// ---------------------------------------------------------------------------
// EulerSequence enum
// ---------------------------------------------------------------------------

/// The 12 possible Euler rotation sequences.
///
/// Tait-Bryan (aerodynamic) sequences use three distinct axes.
/// Proper Euler (astronomical) sequences repeat the first axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum EulerSequence {
    // Tait-Bryan (aerodynamic)
    XYZ = 0,
    XZY = 1,
    YZX = 2,
    YXZ = 3,
    ZXY = 4,
    ZYX = 5,
    // Proper Euler (astronomical)
    XYX = 6,
    XZX = 7,
    YZY = 8,
    YXY = 9,
    ZXZ = 10,
    ZYZ = 11,
}

/// List of all 12 sequences, useful for exhaustive iteration in tests.
pub const ALL_SEQUENCES: [EulerSequence; 12] = [
    EulerSequence::XYZ,
    EulerSequence::XZY,
    EulerSequence::YZX,
    EulerSequence::YXZ,
    EulerSequence::ZXY,
    EulerSequence::ZYX,
    EulerSequence::XYX,
    EulerSequence::XZX,
    EulerSequence::YZY,
    EulerSequence::YXY,
    EulerSequence::ZXZ,
    EulerSequence::ZYZ,
];

// ---------------------------------------------------------------------------
// EulerInfo (private lookup table)
// ---------------------------------------------------------------------------

/// Per-sequence metadata, ported from `euler_angles.cc:50-110`.
struct EulerInfo {
    /// Rotation axes in order (X=0, Y=1, Z=2).
    indices: [usize; 3],
    /// For aero: first axis index. For astro: omitted axis index.
    alternate_x: usize,
    /// For aero: third axis index. For astro: omitted axis index.
    alternate_z: usize,
    /// Whether the effective 3-distinct-axis sequence is an even permutation of XYZ.
    is_even_permutation: bool,
    /// True for Tait-Bryan (aerodynamic), false for proper Euler (astronomical).
    is_aerodynamics_sequence: bool,
}

/// Table from `euler_angles.cc:96-110`, indexed by `EulerSequence as usize`.
static EULER_INFO: [EulerInfo; 12] = [
    //  seq       indices      alt_x alt_z  even   aero
    EulerInfo {
        indices: [0, 1, 2],
        alternate_x: 0,
        alternate_z: 2,
        is_even_permutation: true,
        is_aerodynamics_sequence: true,
    }, // XYZ
    EulerInfo {
        indices: [0, 2, 1],
        alternate_x: 0,
        alternate_z: 1,
        is_even_permutation: false,
        is_aerodynamics_sequence: true,
    }, // XZY
    EulerInfo {
        indices: [1, 2, 0],
        alternate_x: 1,
        alternate_z: 0,
        is_even_permutation: true,
        is_aerodynamics_sequence: true,
    }, // YZX
    EulerInfo {
        indices: [1, 0, 2],
        alternate_x: 1,
        alternate_z: 2,
        is_even_permutation: false,
        is_aerodynamics_sequence: true,
    }, // YXZ
    EulerInfo {
        indices: [2, 0, 1],
        alternate_x: 2,
        alternate_z: 1,
        is_even_permutation: true,
        is_aerodynamics_sequence: true,
    }, // ZXY
    EulerInfo {
        indices: [2, 1, 0],
        alternate_x: 2,
        alternate_z: 0,
        is_even_permutation: false,
        is_aerodynamics_sequence: true,
    }, // ZYX
    EulerInfo {
        indices: [0, 1, 0],
        alternate_x: 2,
        alternate_z: 2,
        is_even_permutation: true,
        is_aerodynamics_sequence: false,
    }, // XYX
    EulerInfo {
        indices: [0, 2, 0],
        alternate_x: 1,
        alternate_z: 1,
        is_even_permutation: false,
        is_aerodynamics_sequence: false,
    }, // XZX
    EulerInfo {
        indices: [1, 2, 1],
        alternate_x: 0,
        alternate_z: 0,
        is_even_permutation: true,
        is_aerodynamics_sequence: false,
    }, // YZY
    EulerInfo {
        indices: [1, 0, 1],
        alternate_x: 2,
        alternate_z: 2,
        is_even_permutation: false,
        is_aerodynamics_sequence: false,
    }, // YXY
    EulerInfo {
        indices: [2, 0, 2],
        alternate_x: 1,
        alternate_z: 1,
        is_even_permutation: true,
        is_aerodynamics_sequence: false,
    }, // ZXZ
    EulerInfo {
        indices: [2, 1, 2],
        alternate_x: 0,
        alternate_z: 0,
        is_even_permutation: false,
        is_aerodynamics_sequence: false,
    }, // ZYZ
];

/// Threshold for gimbal lock detection.
/// From `euler_angles.cc:43`.
const GIMBAL_LOCK_THRESHOLD: f64 = 1e-13;

// ---------------------------------------------------------------------------
// Matrix element access helper
// ---------------------------------------------------------------------------

/// Access `trans[row][col]` in JEOD row-major convention.
/// glam `DMat3` is column-major, so `trans.col(col)[row]`.
#[inline]
fn t(trans: &DMat3, row: usize, col: usize) -> f64 {
    trans.col(col)[row]
}

// ---------------------------------------------------------------------------
// Shared implementations (private). Both the bare-`f64` and typed public
// entry points dispatch to these helpers so the two surfaces remain
// bit-identical at equal numeric inputs.
// ---------------------------------------------------------------------------

fn compute_quaternion_from_euler_angles_impl(
    angles: [f64; 3],
    sequence: EulerSequence,
) -> JeodQuat {
    let info = &EULER_INFO[sequence as usize];
    let axes = &info.indices;

    let mut q = [JeodQuat::identity(); 3];

    for ii in 0..3 {
        let htheta = 0.5 * angles[ii];
        let cosht = htheta.cos();
        let sinht = htheta.sin();

        // JEOD left-transformation: scalar = cos(theta/2),
        // vector[axis] = -sin(theta/2), others zero.
        let mut data = [0.0_f64; 4];
        data[0] = cosht;
        data[1 + axes[ii]] = -sinht; // data[1..4] maps to vector[0..3]
        q[ii] = JeodQuat::from_array(data);
    }

    // Reverse-order product: q[2] * q[1], then that * q[0].
    let q21 = q[2].multiply(&q[1]);
    let mut result = q21.multiply(&q[0]);
    result.normalize();
    result
}

fn compute_matrix_from_euler_angles_impl(angles: [f64; 3], sequence: EulerSequence) -> DMat3 {
    let info = &EULER_INFO[sequence as usize];
    let axes = &info.indices;

    let mut m = [[[0.0_f64; 3]; 3]; 3];

    for ii in 0..3 {
        let sin_theta = angles[ii].sin();
        let cos_theta = angles[ii].cos();
        match axes[ii] {
            0 => {
                // Rotation about X
                m[ii][0][0] = 1.0;
                m[ii][1][1] = cos_theta;
                m[ii][1][2] = sin_theta;
                m[ii][2][1] = -sin_theta;
                m[ii][2][2] = cos_theta;
            }
            1 => {
                // Rotation about Y
                m[ii][1][1] = 1.0;
                m[ii][0][0] = cos_theta;
                m[ii][0][2] = -sin_theta;
                m[ii][2][0] = sin_theta;
                m[ii][2][2] = cos_theta;
            }
            2 => {
                // Rotation about Z
                m[ii][2][2] = 1.0;
                m[ii][0][0] = cos_theta;
                m[ii][0][1] = sin_theta;
                m[ii][1][0] = -sin_theta;
                m[ii][1][1] = cos_theta;
            }
            _ => unreachable!(),
        }
    }

    // Convert raw arrays to DMat3 and multiply: m[2] * m[1] * m[0].
    let to_dmat3 = |arr: &[[f64; 3]; 3]| -> DMat3 {
        crate::types::mat3_from_rows(
            glam::DVec3::new(arr[0][0], arr[0][1], arr[0][2]),
            glam::DVec3::new(arr[1][0], arr[1][1], arr[1][2]),
            glam::DVec3::new(arr[2][0], arr[2][1], arr[2][2]),
        )
    };

    let m0 = to_dmat3(&m[0]);
    let m1 = to_dmat3(&m[1]);
    let m2 = to_dmat3(&m[2]);

    // JEOD Matrix3x3::product(A, B, C) computes C = A * B (row-major multiply).
    // m21 = m[2] * m[1], then trans = m21 * m[0].
    let m21 = m2 * m1;
    m21 * m0
}

fn compute_euler_angles_from_matrix_impl(trans: &DMat3, sequence: EulerSequence) -> [f64; 3] {
    let info = &EULER_INFO[sequence as usize];

    // Extract theta_val: trans[info.indices[2]][info.indices[0]]
    let mut theta_val = t(trans, info.indices[2], info.indices[0]);

    // Extract sin/cos of phi and psi (scaled by cos(theta) or sin(theta)).
    let mut sin_phi = t(trans, info.indices[2], info.indices[1]);
    let mut cos_phi = t(trans, info.indices[2], info.alternate_z);
    let mut sin_psi = t(trans, info.indices[1], info.indices[0]);
    let mut cos_psi = t(trans, info.alternate_x, info.indices[0]);

    // Compute alternative theta values for near-singular detection.
    let alt_theta_val1 = (sin_phi * sin_phi + cos_phi * cos_phi).sqrt();
    let alt_theta_val2 = (sin_psi * sin_psi + cos_psi * cos_psi).sqrt();
    let alt_theta_val = 0.5 * (alt_theta_val1 + alt_theta_val2);

    // Negate theta_val for odd permutation aerodynamics sequences.
    if info.is_aerodynamics_sequence && !info.is_even_permutation {
        theta_val = -theta_val;
    }

    // Compute theta.
    let theta;
    if alt_theta_val < theta_val.abs() {
        let alt_theta = alt_theta_val.asin();
        if info.is_aerodynamics_sequence {
            if theta_val < 0.0 {
                theta = -FRAC_PI_2 + alt_theta;
            } else {
                theta = FRAC_PI_2 - alt_theta;
            }
        } else if theta_val < 0.0 {
            theta = PI - alt_theta;
        } else {
            theta = alt_theta;
        }
    } else if info.is_aerodynamics_sequence {
        theta = theta_val.asin();
    } else {
        theta = theta_val.acos();
    }

    // Compute phi and psi.
    let phi;
    let psi;

    if alt_theta_val > GIMBAL_LOCK_THRESHOLD {
        // Not gimbal-locked: correct signs and use atan2.

        if info.is_aerodynamics_sequence {
            // Sine values have wrong sign for even permutation aero sequences.
            if info.is_even_permutation {
                sin_phi = -sin_phi;
                sin_psi = -sin_psi;
            }
        } else {
            // A cosine term has the wrong sign in the case of astro sequences.
            // The term with the wrong sign is cos_phi for even permutations but
            // cos_psi for odd permutations.
            if info.is_even_permutation {
                cos_phi = -cos_phi;
            } else {
                cos_psi = -cos_psi;
            }
        }

        // Compute phi and psi.
        phi = sin_phi.atan2(cos_phi);
        psi = sin_psi.atan2(cos_psi);
    } else {
        // Gimbal lock: use alternative matrix elements.
        let gl_sin_phi = t(trans, info.indices[1], info.alternate_z);
        let gl_cos_phi = t(trans, info.indices[1], info.indices[1]);

        // Sine value has wrong sign for odd sequences.
        let corrected_sin_phi = if !info.is_even_permutation {
            -gl_sin_phi
        } else {
            gl_sin_phi
        };

        phi = corrected_sin_phi.atan2(gl_cos_phi);
        psi = 0.0;
    }

    [phi, theta, psi]
}

// ---------------------------------------------------------------------------
// Public API — bare-`f64` surface (deprecated, removed in Phase 10 of #101).
// ---------------------------------------------------------------------------

/// Build a left-transformation quaternion from Euler angles (radians).
///
/// Port of `euler_angles.cc:121-153`. Constructs three simple quaternions
/// (one per rotation axis) and multiplies in reverse order:
/// `result = q[2] * q[1] * q[0]`, then normalizes.
///
/// **Deprecated:** use [`compute_quaternion_from_euler_angles_typed`], which
/// accepts `uom` `Angle`s and returns a [`NormalizedQuat`] witness.
#[doc(hidden)]
#[deprecated(
    since = "0.2.0-phase-3",
    note = "use compute_quaternion_from_euler_angles_typed; f64 variant removed in Phase 10"
)]
pub fn compute_quaternion_from_euler_angles(angles: [f64; 3], sequence: EulerSequence) -> JeodQuat {
    compute_quaternion_from_euler_angles_impl(angles, sequence)
}

/// Build a left-transformation matrix from Euler angles (radians).
///
/// Port of `euler_angles.cc:164-218`. Constructs three elementary rotation
/// matrices and multiplies in reverse order: `result = m[2] * m[1] * m[0]`.
///
/// **Deprecated:** use [`compute_matrix_from_euler_angles_typed`], which
/// accepts `uom` `Angle`s.
#[doc(hidden)]
#[deprecated(
    since = "0.2.0-phase-3",
    note = "use compute_matrix_from_euler_angles_typed; f64 variant removed in Phase 10"
)]
pub fn compute_matrix_from_euler_angles(angles: [f64; 3], sequence: EulerSequence) -> DMat3 {
    compute_matrix_from_euler_angles_impl(angles, sequence)
}

/// Extract Euler angles (radians) from a left-transformation matrix.
///
/// Port of `euler_angles.cc:297-464`. Returns `[phi, theta, psi]` in radians.
///
/// **Deprecated:** use [`compute_euler_angles_from_matrix_typed`], which
/// returns `uom` `Angle`s.
#[doc(hidden)]
#[deprecated(
    since = "0.2.0-phase-3",
    note = "use compute_euler_angles_from_matrix_typed; f64 variant removed in Phase 10"
)]
pub fn compute_euler_angles_from_matrix(trans: &DMat3, sequence: EulerSequence) -> [f64; 3] {
    compute_euler_angles_from_matrix_impl(trans, sequence)
}

// ---------------------------------------------------------------------------
// Public API — typed surface (uom `Angle`s + `NormalizedQuat` witness).
// ---------------------------------------------------------------------------

/// Typed sibling of [`compute_quaternion_from_euler_angles`].
///
/// Accepts a triple of `uom::si::f64::Angle` values and returns a
/// [`NormalizedQuat`] witness of the resulting left-transformation
/// quaternion. The inner [`JeodQuat`] is bit-identical to the value
/// returned by the bare-`f64` variant when the same angles are supplied in
/// radians, so both surfaces agree on every representable input.
///
/// The underlying `compute_quaternion_from_euler_angles_impl` applies
/// JEOD's Padé-based `normalize()` before returning, so the witness is
/// constructed via [`NormalizedQuat::new`]: the norm is checked against
/// `NormalizedQuat::DEFAULT_TOLERANCE` (1e-12), which is comfortably
/// wider than floating-point round-off on a post-normalization quaternion.
/// Panics only if that tolerance is violated — i.e. if the underlying
/// normalization routine is itself broken.
pub fn compute_quaternion_from_euler_angles_typed(
    angles: [Angle; 3],
    sequence: EulerSequence,
) -> NormalizedQuat<ScalarFirst, LeftTransform> {
    let radians = [
        angles[0].get::<radian>(),
        angles[1].get::<radian>(),
        angles[2].get::<radian>(),
    ];
    let q = compute_quaternion_from_euler_angles_impl(radians, sequence);
    NormalizedQuat::new(q)
        .expect("compute_quaternion_from_euler_angles_impl returns a normalized quaternion")
}

/// Typed sibling of [`compute_matrix_from_euler_angles`].
///
/// Accepts a triple of `uom::si::f64::Angle` values and returns the raw
/// `DMat3` transformation matrix. Bit-identical to the bare-`f64` variant
/// given the same angles (in radians).
pub fn compute_matrix_from_euler_angles_typed(
    angles: [Angle; 3],
    sequence: EulerSequence,
) -> DMat3 {
    let radians = [
        angles[0].get::<radian>(),
        angles[1].get::<radian>(),
        angles[2].get::<radian>(),
    ];
    compute_matrix_from_euler_angles_impl(radians, sequence)
}

/// Typed sibling of [`compute_euler_angles_from_matrix`].
///
/// Returns a triple of `uom::si::f64::Angle` values carrying the radian
/// results of the JEOD extraction algorithm. Bit-identical numerically to
/// the bare-`f64` variant.
pub fn compute_euler_angles_from_matrix_typed(trans: DMat3, sequence: EulerSequence) -> [Angle; 3] {
    let [phi, theta, psi] = compute_euler_angles_from_matrix_impl(&trans, sequence);
    [
        Angle::new::<radian>(phi),
        Angle::new::<radian>(theta),
        Angle::new::<radian>(psi),
    ]
}

/// Build a left-transformation matrix from a witnessed unit-norm quaternion.
///
/// Preferred entry point for new callers: gates the unit-norm precondition
/// at the type level instead of relying on callers remembering to
/// renormalize after integration. Delegates to
/// [`NormalizedQuat::left_quat_to_transformation`], which applies the JEOD
/// `quaternion_to_matrix.cc` formula on the witnessed payload.
pub fn quaternion_to_matrix_normalized(q: NormalizedQuat<ScalarFirst, LeftTransform>) -> DMat3 {
    q.left_quat_to_transformation()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    // Existing tests intentionally exercise the bare-`f64` public surface so
    // the deprecated and typed entry points stay covered together through
    // the Phase 10 removal.
    #![allow(deprecated)]

    use super::*;
    use crate::test_utils::{approx_eq_f64, approx_eq_mat3};
    use glam::DVec3;
    use uom::si::angle::radian;

    const TOL: f64 = 1e-14;

    /// Helper: compare two angle triples element-wise.
    fn approx_eq_angles(a: &[f64; 3], b: &[f64; 3], tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    // -------------------------------------------------------------------
    // 1. Round-trip for all 12 sequences
    // -------------------------------------------------------------------
    #[test]
    fn round_trip_all_sequences() {
        let test_angles_deg = [30.0_f64, 45.0, 60.0];
        let test_angles_rad = [
            test_angles_deg[0].to_radians(),
            test_angles_deg[1].to_radians(),
            test_angles_deg[2].to_radians(),
        ];

        for &seq in &ALL_SEQUENCES {
            let mat = compute_matrix_from_euler_angles(test_angles_rad, seq);
            let extracted = compute_euler_angles_from_matrix(&mat, seq);
            let mat2 = compute_matrix_from_euler_angles(extracted, seq);

            assert!(
                approx_eq_mat3(&mat, &mat2, TOL),
                "Round-trip matrix mismatch for {:?}:\n  original angles = {:?}\n  extracted = {:?}\n  max element diff = {:e}",
                seq,
                test_angles_rad,
                extracted,
                max_mat_diff(&mat, &mat2),
            );

            // For aero sequences the angles themselves should round-trip
            // (astro sequences may wrap differently, so only check matrix).
            if EULER_INFO[seq as usize].is_aerodynamics_sequence {
                assert!(
                    approx_eq_angles(&test_angles_rad, &extracted, TOL),
                    "Round-trip angle mismatch for {:?}:\n  input  = {:?}\n  output = {:?}",
                    seq,
                    test_angles_rad,
                    extracted,
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // 2. Known angles: XYZ with 30-45-60 degrees
    // -------------------------------------------------------------------
    #[test]
    fn known_angles_xyz() {
        let angles = [
            30.0_f64.to_radians(),
            45.0_f64.to_radians(),
            60.0_f64.to_radians(),
        ];
        let mat = compute_matrix_from_euler_angles(angles, EulerSequence::XYZ);
        let extracted = compute_euler_angles_from_matrix(&mat, EulerSequence::XYZ);

        assert!(
            approx_eq_angles(&angles, &extracted, TOL),
            "XYZ known angles mismatch:\n  input  = {:?}\n  output = {:?}",
            angles,
            extracted,
        );
    }

    // -------------------------------------------------------------------
    // 3. Gimbal lock handling: XYZ at theta = 90 degrees
    // -------------------------------------------------------------------
    #[test]
    fn gimbal_lock_xyz() {
        let angles = [0.3, FRAC_PI_2, 0.0]; // phi arbitrary, theta = 90 deg, psi = 0
        let mat = compute_matrix_from_euler_angles(angles, EulerSequence::XYZ);
        let extracted = compute_euler_angles_from_matrix(&mat, EulerSequence::XYZ);

        // Should not produce NaN.
        for (i, val) in extracted.iter().enumerate() {
            assert!(!val.is_nan(), "Gimbal lock XYZ: angle[{}] is NaN", i);
        }

        // Reconstructed matrix should match.
        let mat2 = compute_matrix_from_euler_angles(extracted, EulerSequence::XYZ);
        assert!(
            approx_eq_mat3(&mat, &mat2, 1e-12),
            "Gimbal lock XYZ: matrix mismatch\n  extracted = {:?}\n  max diff = {:e}",
            extracted,
            max_mat_diff(&mat, &mat2),
        );
    }

    #[test]
    fn gimbal_lock_negative_theta() {
        // XYZ at theta = -90 degrees.
        let angles = [0.7, -FRAC_PI_2, 0.0];
        let mat = compute_matrix_from_euler_angles(angles, EulerSequence::XYZ);
        let extracted = compute_euler_angles_from_matrix(&mat, EulerSequence::XYZ);

        for (i, val) in extracted.iter().enumerate() {
            assert!(!val.is_nan(), "Gimbal lock -90 XYZ: angle[{}] is NaN", i);
        }

        let mat2 = compute_matrix_from_euler_angles(extracted, EulerSequence::XYZ);
        assert!(
            approx_eq_mat3(&mat, &mat2, 1e-12),
            "Gimbal lock -90 XYZ: matrix mismatch\n  max diff = {:e}",
            max_mat_diff(&mat, &mat2),
        );
    }

    #[test]
    fn gimbal_lock_astro_zxz() {
        // Astronomical sequence: ZXZ at theta = 0 (cos(theta) = 1, gimbal lock).
        let angles = [0.5, 0.0, 0.0];
        let mat = compute_matrix_from_euler_angles(angles, EulerSequence::ZXZ);
        let extracted = compute_euler_angles_from_matrix(&mat, EulerSequence::ZXZ);

        for (i, val) in extracted.iter().enumerate() {
            assert!(!val.is_nan(), "Gimbal lock ZXZ: angle[{}] is NaN", i);
        }

        let mat2 = compute_matrix_from_euler_angles(extracted, EulerSequence::ZXZ);
        assert!(
            approx_eq_mat3(&mat, &mat2, 1e-12),
            "Gimbal lock ZXZ: matrix mismatch\n  max diff = {:e}",
            max_mat_diff(&mat, &mat2),
        );
    }

    // -------------------------------------------------------------------
    // 4. Identity matrix -> [0, 0, 0] for all sequences
    // -------------------------------------------------------------------
    #[test]
    fn identity_matrix_all_sequences() {
        let identity = DMat3::IDENTITY;
        for &seq in &ALL_SEQUENCES {
            let angles = compute_euler_angles_from_matrix(&identity, seq);
            for (i, val) in angles.iter().enumerate() {
                assert!(
                    approx_eq_f64(*val, 0.0, 1e-13),
                    "Identity -> {:?}: angle[{}] = {:e}, expected 0.0",
                    seq,
                    i,
                    val,
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // 5. Matrix-quaternion consistency
    // -------------------------------------------------------------------
    #[test]
    fn matrix_quaternion_consistency() {
        let test_angles = [
            [0.3, 0.5, 0.7],
            [1.0, -0.4, 2.1],
            [0.1, 0.2, 0.3],
            [-0.5, 0.8, -1.2],
        ];

        for &seq in &ALL_SEQUENCES {
            for angles in &test_angles {
                let mat_direct = compute_matrix_from_euler_angles(*angles, seq);
                let quat = compute_quaternion_from_euler_angles(*angles, seq);
                let mat_from_quat = quat.left_quat_to_transformation();

                assert!(
                    approx_eq_mat3(&mat_direct, &mat_from_quat, TOL),
                    "Matrix/quat mismatch for {:?}, angles={:?}\n  max diff = {:e}",
                    seq,
                    angles,
                    max_mat_diff(&mat_direct, &mat_from_quat),
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // 6. Additional: multiple non-trivial angle sets, all sequences
    // -------------------------------------------------------------------
    #[test]
    fn round_trip_varied_angles() {
        let angle_sets: [[f64; 3]; 5] = [
            [0.1, 0.2, 0.3],
            [-0.5, 0.8, -1.2],
            [1.0, -0.3, 2.5],
            [0.01, 0.01, 0.01], // small angles
            [2.0, -1.0, 3.0],   // large angles
        ];

        for &seq in &ALL_SEQUENCES {
            for angles in &angle_sets {
                let mat = compute_matrix_from_euler_angles(*angles, seq);
                let extracted = compute_euler_angles_from_matrix(&mat, seq);
                let mat2 = compute_matrix_from_euler_angles(extracted, seq);

                assert!(
                    approx_eq_mat3(&mat, &mat2, 1e-13),
                    "Varied round-trip failed for {:?}, angles={:?}\n  extracted={:?}\n  max diff = {:e}",
                    seq,
                    angles,
                    extracted,
                    max_mat_diff(&mat, &mat2),
                );
            }
        }
    }

    /// Max absolute element-wise difference between two matrices (for diagnostics).
    fn max_mat_diff(a: &DMat3, b: &DMat3) -> f64 {
        let mut max_d = 0.0_f64;
        for c in 0..3 {
            for r in 0..3 {
                let d = (a.col(c)[r] - b.col(c)[r]).abs();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        max_d
    }

    // -------------------------------------------------------------------
    // JEOD euler_derived_state_ut.cc test vector — 1e-12 rad tolerance
    // -------------------------------------------------------------------

    /// Validates Euler angle extraction against the exact test vector from
    /// JEOD `euler_derived_state_ut.cc`. The matrix corresponds to a
    /// Roll-Pitch-Yaw (XYZ) decomposition of [30°, 45°, 60°].
    ///
    /// The JEOD source has one unique test vector (repeated across three
    /// test blocks). We test both ref→body and body→ref directions, and
    /// all 6 Tait-Bryan sequences on the same matrix.
    #[test]
    fn jeod_euler_test_vector_1e12() {
        // Exact matrix from euler_derived_state_ut.cc (row-major → glam column-major)
        let t = DMat3::from_cols(
            DVec3::new(
                0.353_553_390_593_273_8,
                -0.612_372_435_695_794_6,
                0.707_106_781_186_547_5,
            ),
            DVec3::new(
                0.926_776_695_296_636_9,
                0.126_826_484_044_322_3,
                -0.353_553_390_593_273_7,
            ),
            DVec3::new(
                0.126_826_484_044_322,
                0.780_330_085_889_910_6,
                0.612_372_435_695_794_6,
            ),
        );

        // Expected ref_body angles for Roll-Pitch-Yaw (XYZ): [30°, 45°, 60°]
        let expected_rad = [
            30.0_f64.to_radians(),
            45.0_f64.to_radians(),
            60.0_f64.to_radians(),
        ];

        let extracted = compute_euler_angles_from_matrix(&t, EulerSequence::XYZ);

        for i in 0..3 {
            let err = (extracted[i] - expected_rad[i]).abs();
            assert!(
                err < 1e-12,
                "JEOD XYZ angle[{i}] error {err:.4e} rad exceeds 1e-12\n  \
                 expected: {:.15}, got: {:.15}",
                expected_rad[i],
                extracted[i],
            );
        }

        // Verify body→ref direction: transpose the matrix and extract
        let t_transpose = t.transpose();
        let body_ref_expected_deg: [f64; 3] = [
            -51.876_568_255_402_19,
            7.286_245_187_115_636,
            -69.118_790_319_646_11,
        ];
        let body_ref_expected_rad = [
            body_ref_expected_deg[0].to_radians(),
            body_ref_expected_deg[1].to_radians(),
            body_ref_expected_deg[2].to_radians(),
        ];
        let body_ref_extracted = compute_euler_angles_from_matrix(&t_transpose, EulerSequence::XYZ);

        for i in 0..3 {
            let err = (body_ref_extracted[i] - body_ref_expected_rad[i]).abs();
            assert!(
                err < 1e-12,
                "JEOD XYZ body_ref angle[{i}] error {err:.4e} rad exceeds 1e-12\n  \
                 expected: {:.15}, got: {:.15}",
                body_ref_expected_rad[i],
                body_ref_extracted[i],
            );
        }

        // Also verify all 6 Tait-Bryan sequences via matrix round-trip at 1e-14
        for &seq in &[
            EulerSequence::XYZ,
            EulerSequence::XZY,
            EulerSequence::YZX,
            EulerSequence::YXZ,
            EulerSequence::ZXY,
            EulerSequence::ZYX,
        ] {
            let angles = compute_euler_angles_from_matrix(&t, seq);
            let reconstructed = compute_matrix_from_euler_angles(angles, seq);
            assert!(
                approx_eq_mat3(&t, &reconstructed, 1e-14),
                "JEOD matrix round-trip for {seq:?} exceeds 1e-14: diff = {:.4e}",
                max_mat_diff(&t, &reconstructed),
            );
        }
    }

    // -------------------------------------------------------------------
    // Typed API (uom `Angle` + NormalizedQuat witness)
    // -------------------------------------------------------------------

    /// Build a `[Angle; 3]` from three radian values without cluttering
    /// the assertion tests.
    fn angles_rad(phi: f64, theta: f64, psi: f64) -> [Angle; 3] {
        [
            Angle::new::<radian>(phi),
            Angle::new::<radian>(theta),
            Angle::new::<radian>(psi),
        ]
    }

    /// A non-trivial angle triple per `EulerSequence`. All chosen to stay
    /// away from 0°, ±90°, ±180° so the extraction algorithm's gimbal-lock
    /// branch is avoided and the raw radian angles round-trip cleanly.
    fn non_trivial_angles_for(seq: EulerSequence) -> [f64; 3] {
        if EULER_INFO[seq as usize].is_aerodynamics_sequence {
            // Aero sequences fully round-trip angles for non-singular theta.
            [0.3_f64, 0.4, 0.5]
        } else {
            // Astro (proper-Euler) sequences have theta in (0, pi). Pick
            // angles well clear of both ends.
            [0.3_f64, 0.9, 0.5]
        }
    }

    #[test]
    fn typed_roundtrip_all_sequences() {
        for &seq in &ALL_SEQUENCES {
            let raw = non_trivial_angles_for(seq);
            let typed_in = angles_rad(raw[0], raw[1], raw[2]);

            let mat = compute_matrix_from_euler_angles_typed(typed_in, seq);
            let extracted = compute_euler_angles_from_matrix_typed(mat, seq);

            // Aero sequences recover the exact input angles; astro sequences
            // are allowed to drift as long as the matrix reconstructs.
            if EULER_INFO[seq as usize].is_aerodynamics_sequence {
                for i in 0..3 {
                    let err = (extracted[i] - typed_in[i]).get::<radian>().abs();
                    assert!(
                        err < 1e-13,
                        "typed roundtrip {seq:?}: angle[{i}] err = {err:.4e} rad"
                    );
                }
            }

            let mat2 = compute_matrix_from_euler_angles_typed(extracted, seq);
            assert!(
                approx_eq_mat3(&mat, &mat2, 1e-13),
                "typed roundtrip matrix mismatch for {seq:?}: diff = {:.4e}",
                {
                    let mut d = 0.0_f64;
                    for c in 0..3 {
                        for r in 0..3 {
                            d = d.max((mat.col(c)[r] - mat2.col(c)[r]).abs());
                        }
                    }
                    d
                }
            );
        }
    }

    #[test]
    fn typed_matrix_bit_identical_to_f64() {
        // The typed path must produce exactly the same DMat3 as the
        // bare-`f64` path for every sequence, given equal numeric inputs.
        for &seq in &ALL_SEQUENCES {
            let raw = non_trivial_angles_for(seq);
            let typed = angles_rad(raw[0], raw[1], raw[2]);

            let mat_typed = compute_matrix_from_euler_angles_typed(typed, seq);
            let mat_f64 = compute_matrix_from_euler_angles(
                [
                    typed[0].get::<radian>(),
                    typed[1].get::<radian>(),
                    typed[2].get::<radian>(),
                ],
                seq,
            );

            for c in 0..3 {
                for r in 0..3 {
                    assert_eq!(
                        mat_typed.col(c)[r].to_bits(),
                        mat_f64.col(c)[r].to_bits(),
                        "{seq:?}: matrix element [{r}][{c}] differs bit-wise",
                    );
                }
            }
        }
    }

    #[test]
    fn typed_angles_bit_identical_to_f64() {
        // Extract both via typed and f64 from the same matrix and check
        // the resulting angle triples are bit-identical.
        for &seq in &ALL_SEQUENCES {
            let raw = non_trivial_angles_for(seq);
            let mat = compute_matrix_from_euler_angles(raw, seq);

            let from_typed = compute_euler_angles_from_matrix_typed(mat, seq);
            let from_f64 = compute_euler_angles_from_matrix(&mat, seq);

            for i in 0..3 {
                assert_eq!(
                    from_typed[i].get::<radian>().to_bits(),
                    from_f64[i].to_bits(),
                    "{seq:?}: angle[{i}] differs bit-wise",
                );
            }
        }
    }

    #[test]
    fn typed_quaternion_bit_identical_to_f64() {
        // The typed quaternion builder must return a NormalizedQuat whose
        // inner JeodQuat is bit-identical to the f64 variant's output.
        for &seq in &ALL_SEQUENCES {
            let raw = non_trivial_angles_for(seq);
            let typed = angles_rad(raw[0], raw[1], raw[2]);

            let q_typed = compute_quaternion_from_euler_angles_typed(typed, seq);
            let q_f64 = compute_quaternion_from_euler_angles(raw, seq);

            let inner = q_typed.inner();
            for i in 0..4 {
                assert_eq!(
                    inner.data[i].to_bits(),
                    q_f64.data[i].to_bits(),
                    "{seq:?}: quaternion data[{i}] differs bit-wise",
                );
            }

            // The NormalizedQuat witness implies unit norm; sanity-check
            // this rather than leave it implicit.
            assert!((inner.norm() - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn quaternion_to_matrix_normalized_matches_f64_matrix() {
        // The new matrix-conversion entry point gated on NormalizedQuat
        // must agree with the bare-`f64` matrix constructor.
        for &seq in &ALL_SEQUENCES {
            let raw = non_trivial_angles_for(seq);
            let typed = angles_rad(raw[0], raw[1], raw[2]);

            let q = compute_quaternion_from_euler_angles_typed(typed, seq);
            let mat_from_quat = quaternion_to_matrix_normalized(q);
            let mat_direct = compute_matrix_from_euler_angles_typed(typed, seq);

            assert!(
                approx_eq_mat3(&mat_from_quat, &mat_direct, 1e-14),
                "{seq:?}: NormalizedQuat-gated matrix disagrees with typed Euler->matrix",
            );
        }
    }
}
