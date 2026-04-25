//! Solid body tidal delta coefficients.
//!
//! Port of JEOD `spherical_harmonics_solid_body_tides.cc`. Computes the
//! first-order tidal perturbation ΔC20 from the positions of tidal bodies
//! (Moon, Sun) in the planet-fixed frame.
//!
//! The formula is:
//! ```text
//! F = Σ [μ_body / μ_primary × (R_primary / r)³ × √5 × (1.5 sin²φ - 0.5)]
//! ΔC20 = k2 / 5 × F
//! ```
//! where φ is the latitude of each tidal body in the planet-fixed frame.

use glam::{DMat3, DVec3};

/// Default Love number k2 for Earth solid body tides.
///
/// Value from JEOD `earth_solid_tides.cc:44`: IERS elastic Earth Love number.
pub const EARTH_K2: f64 = 0.29525;

/// Configuration for solid body tidal effects on a gravity source.
#[derive(Debug, Clone)]
pub struct TidalConfig {
    /// Love number k2 for degree-2 first-order tidal effect.
    pub k2: f64,
    /// Gravitational parameter of the primary body (Earth), m³/s².
    pub mu_primary: f64,
    /// Equatorial radius of the primary body (Earth), m.
    pub radius_primary: f64,
    /// Tidal body parameters: `(mu, inertial_position)` for each body (Moon, Sun).
    /// Positions must be updated each step before calling `compute_delta_c20`.
    pub tidal_bodies: Vec<TidalBody>,
}

/// A body that raises tides on the primary (e.g., Moon, Sun).
#[derive(Debug, Clone)]
pub struct TidalBody {
    /// Gravitational parameter (m³/s²).
    pub mu: f64,
    /// Position in the inertial frame (m). Updated each step from ephemeris.
    pub position_inertial: DVec3,
}

/// Typed sibling of [`TidalConfig`].
///
/// `mu_primary` and per-body `mu` carry the [`GravParam`] dimensional
/// type. `radius_primary` carries [`Length`]. `k2` (a Love number,
/// dimensionless) is wrapped in `Ratio` for type-system parity with
/// other dimensionless physical quantities.
#[derive(Debug, Clone)]
pub struct TidalConfigTyped {
    pub k2: uom::si::f64::Ratio,
    pub mu_primary: jeod_quantities::dims::GravParam,
    pub radius_primary: uom::si::f64::Length,
    pub tidal_bodies: Vec<TidalBodyTyped>,
}

/// Typed sibling of [`TidalBody`].
///
/// `position_inertial` carries the [`Position<Inertial>`] phantom tag.
#[derive(Debug, Clone)]
pub struct TidalBodyTyped {
    pub mu: jeod_quantities::dims::GravParam,
    pub position_inertial: jeod_quantities::aliases::Position<jeod_quantities::frame::Inertial>,
}

impl TidalConfigTyped {
    /// Drop the dimensional annotations and emit the untyped storage form.
    /// Numeric values (kg-derived units) are preserved exactly.
    pub fn to_untyped(&self) -> TidalConfig {
        TidalConfig {
            k2: self.k2.value,
            mu_primary: self.mu_primary.value,
            radius_primary: self.radius_primary.value,
            tidal_bodies: self
                .tidal_bodies
                .iter()
                .map(|b| TidalBody {
                    mu: b.mu.value,
                    position_inertial: b.position_inertial.raw_si(),
                })
                .collect(),
        }
    }
}

/// Typed sibling of [`compute_delta_c20`].
///
/// Same numeric kernel — wraps the result in `Ratio` (the physical
/// dimension of the C20 coefficient is dimensionless).
pub fn compute_delta_c20_typed(
    config: &TidalConfigTyped,
    t_inertial_pfix: &DMat3,
) -> uom::si::f64::Ratio {
    let raw = compute_delta_c20(&config.to_untyped(), t_inertial_pfix);
    uom::si::f64::Ratio::new::<uom::si::ratio::ratio>(raw)
}

/// Compute the first-order tidal delta coefficient ΔC20.
///
/// Port of JEOD `spherical_harmonics_solid_body_tides.cc:69-91`.
///
/// # Arguments
/// * `config` — tidal configuration (Love number, primary body params, tidal bodies)
/// * `t_inertial_pfix` — inertial-to-planet-fixed rotation matrix
///
/// # Returns
/// The ΔC20 value to add to the base C20 coefficient before spherical
/// harmonics evaluation.
pub fn compute_delta_c20(config: &TidalConfig, t_inertial_pfix: &DMat3) -> f64 {
    let sqrt5 = 5.0_f64.sqrt();
    let mut f = 0.0;

    for body in &config.tidal_bodies {
        // Transform tidal body position to planet-fixed frame
        let pfix_position = *t_inertial_pfix * body.position_inertial;

        let r = pfix_position.length();
        if r == 0.0 {
            continue; // Skip coincident bodies
        }

        let sin_phi = pfix_position.z / r; // sin(latitude)
        let r_over_r = config.radius_primary / r;
        let r_over_r_3 = r_over_r * r_over_r * r_over_r; // (R/r)³

        // JEOD formula: mu_body/mu_primary × (R/r)³ × √5 × (1.5 sin²φ - 0.5)
        f += body.mu / config.mu_primary * r_over_r_3 * sqrt5 * (1.5 * sin_phi * sin_phi - 0.5);
    }

    // ΔC20 = k2/5 × F
    config.k2 / 5.0 * f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_c20_zero_for_no_tidal_bodies() {
        let config = TidalConfig {
            k2: EARTH_K2,
            mu_primary: 3.986e14,
            radius_primary: 6.378e6,
            tidal_bodies: vec![],
        };
        let delta = compute_delta_c20(&config, &DMat3::IDENTITY);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn delta_c20_nonzero_for_moon() {
        // Moon at ~400,000 km along z-axis (planet-fixed = inertial for identity)
        let config = TidalConfig {
            k2: EARTH_K2,
            mu_primary: 3.986004415e14,
            radius_primary: 6_378_137.0,
            tidal_bodies: vec![TidalBody {
                mu: 4902.79980693169e9,
                position_inertial: DVec3::new(0.0, 0.0, 384_400_000.0),
            }],
        };
        let delta = compute_delta_c20(&config, &DMat3::IDENTITY);

        // Should be a small but non-zero value (order ~1e-8 to 1e-9)
        assert!(delta.abs() > 1e-12, "ΔC20 too small: {delta}");
        assert!(delta.abs() < 1e-6, "ΔC20 too large: {delta}");

        println!("ΔC20 (Moon at pole): {delta:.6e}");
    }

    #[test]
    fn delta_c20_at_equator_is_negative() {
        // Moon along x-axis (equator): sin²φ = 0, so (1.5*0 - 0.5) = -0.5
        // F should be negative, and with k2 > 0, ΔC20 should be negative
        let config = TidalConfig {
            k2: EARTH_K2,
            mu_primary: 3.986004415e14,
            radius_primary: 6_378_137.0,
            tidal_bodies: vec![TidalBody {
                mu: 4902.79980693169e9,
                position_inertial: DVec3::new(384_400_000.0, 0.0, 0.0),
            }],
        };
        let delta = compute_delta_c20(&config, &DMat3::IDENTITY);

        // At equator: 1.5*sin²(0) - 0.5 = -0.5, so F < 0 and ΔC20 < 0
        assert!(delta < 0.0, "ΔC20 should be negative at equator: {delta}");
    }

    #[test]
    fn delta_c20_at_pole_is_positive() {
        // Moon along z-axis (pole): sin²φ = 1, so (1.5*1 - 0.5) = 1.0
        let config = TidalConfig {
            k2: EARTH_K2,
            mu_primary: 3.986004415e14,
            radius_primary: 6_378_137.0,
            tidal_bodies: vec![TidalBody {
                mu: 4902.79980693169e9,
                position_inertial: DVec3::new(0.0, 0.0, 384_400_000.0),
            }],
        };
        let delta = compute_delta_c20(&config, &DMat3::IDENTITY);

        // At pole: 1.5*sin²(π/2) - 0.5 = 1.0, so F > 0 and ΔC20 > 0
        assert!(delta > 0.0, "ΔC20 should be positive at pole: {delta}");
    }
}
