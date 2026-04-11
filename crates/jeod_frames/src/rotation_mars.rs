//! Mars rotation model (IAU Pathfinder orientation).
//!
//! Computes the inertial-to-Mars-body-fixed transformation matrix at a given
//! TDB epoch. Based on Konopliv et al. (2006, 2011) JPL Pathfinder model.
//!
//! Port of JEOD `models/environment/RNP/RNPMars/`.

use glam::DMat3;
use std::f64::consts::PI;

// ── Unit conversion constants ──
const DEG2RAD: f64 = PI / 180.0;
const MAS2RAD: f64 = DEG2RAD / (1000.0 * 3600.0);
const SECONDS_PER_DAY: f64 = 86400.0;

// ── Nutation constants (from data_rnp_mars.cc) ──
/// Obliquity at J2000 (rad)
const I_AT_J2000: f64 = 25.189_398_458_5 * DEG2RAD;
/// Obliquity rate (rad/day)
const I_DOT: f64 = -0.000_000_005 * DEG2RAD;
/// Mean orbital motion of Mars (rad/day)
const MEAN_MOTION: f64 = 0.524_039_588_28 * DEG2RAD;
/// Mean anomaly at J2000 (rad)
const MEAN_ANOMALY_J2000: f64 = 19.3564 * DEG2RAD;
/// q angle at J2000 (rad), from Konopliv 2006
const Q_ANGLE_J2000: f64 = 142.00 * DEG2RAD;

/// Nutation-in-obliquity amplitudes (milliarcseconds, converted to radians).
/// Index 0 is the constant term; indices 1-9 are Fourier cosine amplitudes.
const I_M: [f64; 10] = [
    -1.4 * MAS2RAD,
    -0.4 * MAS2RAD,
    0.0,
    0.0,
    -49.1 * MAS2RAD,
    515.7 * MAS2RAD,
    112.8 * MAS2RAD,
    19.2 * MAS2RAD,
    3.0 * MAS2RAD,
    0.4 * MAS2RAD,
];

/// Nutation-in-longitude amplitudes (milliarcseconds, converted to radians).
/// Index 0 unused; indices 1-9 are Fourier sine amplitudes.
const PSI_M: [f64; 10] = [
    0.0,
    -632.6 * MAS2RAD,
    -44.2 * MAS2RAD,
    -4.0 * MAS2RAD,
    -104.5 * MAS2RAD,
    1097.0 * MAS2RAD,
    240.1 * MAS2RAD,
    40.9 * MAS2RAD,
    6.5 * MAS2RAD,
    1.0 * MAS2RAD,
];

// ── Precession constants (from data_rnp_mars.cc) ──
/// Precession angle at J2000 (rad)
const PSI_AT_J2000: f64 = 81.968_367_126_7 * DEG2RAD;
/// Precession rate (rad/day)
const PSI_DOT: f64 = -0.000_005_756 * DEG2RAD;
/// Node angle N (rad)
const NODE_N: f64 = 3.379_191_83 * DEG2RAD;
/// Inclination angle J (rad)
const INCL_J: f64 = 24.676_826_69 * DEG2RAD;

// ── Rotation constants (from data_rnp_mars.cc) ──
/// Prime meridian angle at J2000 (rad)
const PHI_AT_J2000: f64 = 133.384_62 * DEG2RAD;
/// Mars rotational velocity (rad/day)
const PLANET_ROT_VEL: f64 = 350.891_985_303 * DEG2RAD;

/// Precomputed NJ matrix (rotation_z(-N) * rotation_x(-J)).
/// Computed once; constant throughout the simulation.
fn nj_matrix() -> DMat3 {
    let cos_n = NODE_N.cos();
    let sin_n = NODE_N.sin();
    let cos_j = INCL_J.cos();
    let sin_j = INCL_J.sin();

    // N_mat: rotation about z-axis by -N (Rz(ψ) convention)
    let n_mat = DMat3::from_cols_array(&[cos_n, sin_n, 0.0, -sin_n, cos_n, 0.0, 0.0, 0.0, 1.0]);
    // J_mat: rotation about x-axis by -J (Rx(ε) convention)
    let j_mat = DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cos_j, sin_j, 0.0, -sin_j, cos_j]);
    n_mat * j_mat
}

/// Nutation outputs needed by precession and rotation.
struct NutationResult {
    /// Nutation in longitude (rad).
    psi_nut: f64,
    /// Obliquity angle including nutation (rad).
    obliquity: f64,
    /// Nutation rotation matrix (Rx by obliquity).
    matrix: DMat3,
}

/// Compute Mars nutation at a given epoch.
///
/// Port of JEOD `NutationMars::update`.
fn compute_nutation(time_days: f64) -> NutationResult {
    let mut psi_nut = 0.0;
    let mut obliquity_nut = I_M[0]; // constant term

    for ii in 1..=9usize {
        let (alpha_m, theta_m) = if ii <= 3 {
            (ii as f64 * MEAN_MOTION, ii as f64 * MEAN_ANOMALY_J2000)
        } else {
            let k = (ii - 3) as f64;
            (k * MEAN_MOTION, k * MEAN_ANOMALY_J2000 + Q_ANGLE_J2000)
        };

        let arg = alpha_m * time_days + theta_m;
        psi_nut += PSI_M[ii] * arg.sin();
        obliquity_nut += I_M[ii] * arg.cos();
    }

    let obliquity = I_AT_J2000 + I_DOT * time_days + obliquity_nut;

    // Nutation matrix: rotation about x-axis by obliquity
    let cos_i = obliquity.cos();
    let sin_i = obliquity.sin();
    let matrix = DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cos_i, sin_i, 0.0, -sin_i, cos_i]);

    NutationResult {
        psi_nut,
        obliquity,
        matrix,
    }
}

/// Compute Mars precession at a given epoch.
///
/// Depends on nutation-in-longitude from the nutation computation.
///
/// Port of JEOD `PrecessionMars::update`.
fn compute_precession(time_days: f64, psi_nut: f64) -> DMat3 {
    let psi_precess = PSI_AT_J2000 + PSI_DOT * time_days + psi_nut;

    let cos_psi = psi_precess.cos();
    let sin_psi = psi_precess.sin();

    // P: rotation about z-axis by psi_precess
    let p_mat =
        DMat3::from_cols_array(&[cos_psi, sin_psi, 0.0, -sin_psi, cos_psi, 0.0, 0.0, 0.0, 1.0]);

    // Final precession = NJ_constant * P
    nj_matrix() * p_mat
}

/// Compute Mars spin rotation at a given epoch.
///
/// Depends on nutation-in-longitude and obliquity from the nutation computation.
///
/// Port of JEOD `RotationMars::update` (full RNP mode).
fn compute_rotation(time_days: f64, psi_nut: f64, obliquity: f64) -> DMat3 {
    // Konopliv 2006, pg 41: Phi(t) = Phi_o + Phi_dot*t - psi_nut*cos(I)
    let mut phi_spin = PHI_AT_J2000 + PLANET_ROT_VEL * time_days - psi_nut * obliquity.cos();

    // Normalize to [0, 2π) for numerical stability with large angles
    phi_spin = phi_spin.rem_euclid(2.0 * PI);

    let cos_phi = phi_spin.cos();
    let sin_phi = phi_spin.sin();

    DMat3::from_cols_array(&[cos_phi, sin_phi, 0.0, -sin_phi, cos_phi, 0.0, 0.0, 0.0, 1.0])
}

/// Compute the full Mars inertial-to-body-fixed transformation matrix.
///
/// # Arguments
/// * `tt_seconds_since_j2000` — TT seconds elapsed since J2000.0 epoch.
///   JEOD's `RNPMars::update_rnp()` reads `time_tt.seconds` (TT, not TDB).
///   Typically obtained from `(tt_tjt - 11544.5) * 86400.0`.
///
/// # Returns
/// 3x3 rotation matrix mapping vectors from the Mars-centered inertial frame
/// (ICRF) to the Mars body-fixed frame. Composition: (P × N × R)^T, matching
/// JEOD's `T_parent_this` convention from `planet_rnp.cc:propagate_rnp()`.
pub fn compute_mars_rotation(tt_seconds_since_j2000: f64) -> DMat3 {
    let time_days = tt_seconds_since_j2000 / SECONDS_PER_DAY;

    // 1. Nutation (must be first — precession and rotation depend on it)
    let nut = compute_nutation(time_days);

    // 2. Precession (depends on psi_nut)
    let p = compute_precession(time_days, nut.psi_nut);

    // 3. Rotation (depends on psi_nut and obliquity)
    let r = compute_rotation(time_days, nut.psi_nut, nut.obliquity);

    // 4. Compose: inertial_to_body = (P × N × R)^T
    // JEOD's propagate_rnp() transposes each component: T = R^T × N^T × P^T.
    (p * nut.matrix * r).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mars_rotation_at_j2000() {
        let mat = compute_mars_rotation(0.0);
        // Should be a valid rotation matrix: orthogonal, det = 1
        let det = mat.determinant();
        assert!(
            (det - 1.0).abs() < 1e-12,
            "determinant should be 1, got {det}"
        );
        // M * M^T should be identity
        let mt = mat.transpose();
        let product = mat * mt;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = product.col(j)[i];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "M*M^T[{i}][{j}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn mars_rotation_changes_with_time() {
        let mat0 = compute_mars_rotation(0.0);
        let mat1 = compute_mars_rotation(3600.0); // 1 hour later
                                                  // Matrices should differ (Mars rotates ~14.6°/hr)
        let diff = (mat1.col(0) - mat0.col(0)).length();
        assert!(
            diff > 0.01,
            "rotation should change over 1 hour, diff={diff}"
        );
    }

    #[test]
    fn mars_sidereal_day() {
        // Mars sidereal day ≈ 88642.66 seconds (24h 37m 22.66s)
        // After one sidereal day, the rotation matrix should be nearly identical
        let sidereal_day_s = 360.0 / (PLANET_ROT_VEL / DEG2RAD) * SECONDS_PER_DAY;
        let mat0 = compute_mars_rotation(0.0);
        let mat1 = compute_mars_rotation(sidereal_day_s);
        // The spin component repeats, but precession/nutation shift slightly
        // Check that the matrices are close but not identical
        let diff = (mat1.col(0) - mat0.col(0)).length();
        assert!(
            diff < 0.001,
            "after one sidereal day, rotation should nearly repeat, diff={diff}"
        );
    }

    #[test]
    fn nutation_fourier_series_nonzero() {
        // At an arbitrary epoch, nutation terms should be nonzero
        let nut = compute_nutation(1000.0); // ~2.7 years after J2000
        assert!(
            nut.psi_nut.abs() > 1e-10,
            "nutation in longitude should be nonzero"
        );
        assert!(
            (nut.obliquity - I_AT_J2000).abs() > 1e-12,
            "obliquity should differ from J2000 value"
        );
    }
}
