//! Post-Newtonian (PPN) relativistic gravity correction.
//!
//! Implements the parametric post-Newtonian acceleration correction from
//! Folkner et al. (DE430/DE431 ephemeris), equation 27, with PPN parameters
//! β = γ = 1 (General Relativity).
//!
//! This correction is additive to the Newtonian gravitational acceleration
//! and is significant for Mercury's orbit (~43 arcsec/century perihelion
//! advance) but negligible for Earth-orbiting satellites.
//!
//! Port of JEOD `GravityControls::calc_relativistic` in
//! `models/environment/gravity/src/gravity_controls.cc`, lines 456-566.

use glam::DVec3;

/// Speed of light squared (m²/s²). Exact value: (299_792_458)².
const C_SQUARED: f64 = 89_875_517_873_681_764.0;

/// A gravity source for relativistic correction computation.
///
/// Each source contributes a potential at the vehicle and at the primary body,
/// and an acceleration of the primary body toward it.
#[derive(Debug, Clone)]
pub struct RelativisticSource {
    /// Gravitational parameter μ = GM (m³/s²).
    pub mu: f64,
    /// Position in inertial frame (m).
    pub position: DVec3,
}

/// Compute the post-Newtonian relativistic acceleration correction.
///
/// Returns the relativistic acceleration of the vehicle due to the primary
/// body, accounting for all other bodies' gravitational influence.
///
/// # Arguments
/// * `primary_mu` — gravitational parameter of the primary body (m³/s²)
/// * `primary_pos` — position of the primary body in inertial frame (m)
/// * `vehicle_pos` — position of the vehicle in inertial frame (m)
/// * `vehicle_vel` — velocity of the vehicle in inertial frame (m/s)
/// * `primary_vel` — velocity of the primary body in inertial frame (m/s)
/// * `other_sources` — all other gravity sources (for potentials and body acceleration)
///
/// # Returns
/// Relativistic acceleration correction (m/s²), to be added to the Newtonian
/// gravitational acceleration.
///
/// # References
/// Folkner et al., DE430/DE431 ephemeris, equation 27.
/// Genova et al., equation 4.
pub fn compute_relativistic_correction(
    primary_mu: f64,
    primary_pos: DVec3,
    vehicle_pos: DVec3,
    vehicle_vel: DVec3,
    primary_vel: DVec3,
    other_sources: &[RelativisticSource],
) -> DVec3 {
    let rel_pos = vehicle_pos - primary_pos; // A - B (vehicle - primary)
    let rel_vel = vehicle_vel - primary_vel;
    let r_sq = rel_pos.length_squared();
    let r_mag = r_sq.sqrt();

    if r_mag < 1.0 {
        return DVec3::ZERO; // Avoid singularity
    }

    // ── Compute potentials and primary body acceleration from other sources ──
    let mut body_potential = 0.0; // Potential at primary due to others
    let mut point_potential = 0.0; // Potential at vehicle due to others
    let mut body_accel = DVec3::ZERO; // Acceleration of primary toward others

    for src in other_sources {
        // Distance from vehicle to this source
        let r_vehicle_src = (vehicle_pos - src.position).length();
        if r_vehicle_src > 1.0 {
            point_potential += src.mu / r_vehicle_src;
        }

        // Distance from primary to this source
        let r_primary_src = (primary_pos - src.position).length();
        if r_primary_src > 1.0 {
            body_potential += src.mu / r_primary_src;

            // Newtonian acceleration of primary toward this source
            let dir = src.position - primary_pos;
            body_accel += src.mu / (r_primary_src * r_primary_src * r_primary_src) * dir;
        }
    }

    // ── Term 1: along relative position ──
    // s_1 scalar (Folkner eq 27 terms, with β=γ=1):
    //   -4 × Φ_A (potential at vehicle from others)
    //   -1 × Φ_B (potential at primary from others) [2β-1 = 1]
    //   +2 × |v_rel|² - v_A²  [(1+γ)|v_A-v_B|² - v_A² → 2|v_rel|² - v_A²]
    //   -1.5 × (r̂_AB · v_B)²
    //   -0.5 × (r_B - r_A) · a_B
    let v_a_sq = vehicle_vel.length_squared();
    let v_rel_sq = rel_vel.length_squared();
    let r_dot_vb = rel_pos.dot(primary_vel);
    let r_dot_vb_over_r = r_dot_vb / r_mag;

    let s1 = -4.0 * point_potential - body_potential + 2.0 * v_rel_sq
        - v_a_sq
        - 1.5 * (r_dot_vb_over_r * r_dot_vb_over_r)
        - 0.5 * (-rel_pos).dot(body_accel);

    let accel1 = rel_pos * (-s1 / r_sq);

    // ── Term 2: along relative velocity ──
    // (r_AB · ((2+2γ)v_A - (1+2γ)v_B)) × (v_A - v_B) / r²
    // With γ=1: (r_AB · (4v_A - 3v_B)) × v_rel / r²
    let term2_vel = 4.0 * vehicle_vel - 3.0 * primary_vel;
    let scale2 = rel_pos.dot(term2_vel) / r_sq;
    let accel2 = rel_vel * scale2;

    // ── Term 3: along primary body's acceleration ──
    // (3+4γ)/2 × a_B = 3.5 × a_B  (with γ=1)
    let accel3 = body_accel * 3.5;

    // ── Combine and scale by common factor ──
    let common_factor = primary_mu / r_mag / C_SQUARED;
    (accel1 + accel2 + accel3) * common_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 1.327_124_400_18e20;
    const MU_EARTH: f64 = 3.986_004_415e14;

    #[test]
    fn relativistic_correction_is_small() {
        // Mercury at perihelion: ~46 million km from Sun, ~59 km/s
        let sun_pos = DVec3::ZERO;
        let mercury_pos = DVec3::new(4.6e10, 0.0, 0.0);
        let mercury_vel = DVec3::new(0.0, 5.9e4, 0.0);
        let sun_vel = DVec3::ZERO;

        let correction = compute_relativistic_correction(
            MU_SUN,
            sun_pos,
            mercury_pos,
            mercury_vel,
            sun_vel,
            &[], // no other sources for simplicity
        );

        // Newtonian acceleration ≈ μ/r² ≈ 1.327e20 / (4.6e10)² ≈ 0.063 m/s²
        let newtonian = MU_SUN / (4.6e10 * 4.6e10);
        let ratio = correction.length() / newtonian;

        // Relativistic correction should be ~1e-7 to 1e-8 of Newtonian
        assert!(
            ratio > 1e-9 && ratio < 1e-5,
            "relativistic/newtonian ratio {ratio:.2e} should be ~1e-7"
        );
    }

    #[test]
    fn correction_increases_near_sun() {
        // Closer to Sun → stronger relativistic effect
        let sun_pos = DVec3::ZERO;
        let sun_vel = DVec3::ZERO;
        let vel = DVec3::new(0.0, 5.0e4, 0.0);

        let c1 = compute_relativistic_correction(
            MU_SUN,
            sun_pos,
            DVec3::new(4.6e10, 0.0, 0.0),
            vel,
            sun_vel,
            &[],
        );
        let c2 = compute_relativistic_correction(
            MU_SUN,
            sun_pos,
            DVec3::new(2.3e10, 0.0, 0.0),
            vel,
            sun_vel,
            &[],
        );

        assert!(
            c2.length() > c1.length(),
            "closer to Sun should give larger correction"
        );
    }

    #[test]
    fn zero_for_zero_mu() {
        let correction = compute_relativistic_correction(
            0.0,
            DVec3::ZERO,
            DVec3::new(1e10, 0.0, 0.0),
            DVec3::new(0.0, 3e4, 0.0),
            DVec3::ZERO,
            &[],
        );
        assert_eq!(correction, DVec3::ZERO);
    }

    #[test]
    fn with_perturbing_body() {
        // Mercury with Earth as perturbing body
        let sun_pos = DVec3::ZERO;
        let mercury_pos = DVec3::new(4.6e10, 0.0, 0.0);
        let mercury_vel = DVec3::new(0.0, 5.9e4, 0.0);
        let sun_vel = DVec3::ZERO;

        let without = compute_relativistic_correction(
            MU_SUN,
            sun_pos,
            mercury_pos,
            mercury_vel,
            sun_vel,
            &[],
        );

        let earth = RelativisticSource {
            mu: MU_EARTH,
            position: DVec3::new(1.496e11, 0.0, 0.0),
        };
        let with = compute_relativistic_correction(
            MU_SUN,
            sun_pos,
            mercury_pos,
            mercury_vel,
            sun_vel,
            &[earth],
        );

        // Earth's contribution should modify the result slightly
        let diff = (with - without).length();
        assert!(
            diff > 0.0,
            "Earth should perturb the relativistic correction"
        );
        // But the difference should be much smaller than the correction itself
        assert!(
            diff < without.length() * 0.1,
            "Earth's perturbation should be small compared to total correction"
        );
    }
}
