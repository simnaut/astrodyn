// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
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

use astrodyn_quantities::aliases::Position;
use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::frame::{RootInertial, SelfPlanet};
use glam::{DMat3, DVec3};
use uom::si::f64::{Length, Ratio};

/// Default Love number k2 for Earth solid body tides.
///
/// Value from JEOD `earth_solid_tides.cc:44`: IERS elastic Earth Love number.
pub const EARTH_K2: f64 = 0.29525;

/// Configuration for solid body tidal effects on a gravity source.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct TidalBody {
    /// Gravitational parameter (m³/s²).
    pub mu: f64,
    /// Position in the inertial frame (m). Updated each step from ephemeris.
    pub position_inertial: DVec3,
}

/// Typed sibling of [`TidalConfig`].
///
/// `mu_primary` and per-body `mu` carry the [`GravParam<SelfPlanet>`]
/// dimensional type — planet-erased because the tidal config is
/// configured at runtime against whichever primary an entity carries
/// (Earth in current sims, but the struct is otherwise body-agnostic).
/// `radius_primary` carries [`Length`]. `k2` (a Love number,
/// dimensionless) is wrapped in [`Ratio`] for type-system parity with
/// other dimensionless physical quantities.
#[derive(Debug, Clone)]
pub struct TidalConfigTyped {
    /// Love number `k2` of the deformable body (dimensionless).
    pub k2: Ratio,
    /// Gravitational parameter μ of the primary body.
    pub mu_primary: GravParam<SelfPlanet>,
    /// Reference radius of the primary body.
    pub radius_primary: Length,
    /// Tidal-perturber bodies (Sun, Moon, …).
    pub tidal_bodies: Vec<TidalBodyTyped>,
}

/// Typed sibling of [`TidalBody`].
///
/// `position_inertial` carries the [`Position<RootInertial>`] phantom tag.
/// `mu` carries the planet-erased [`GravParam<SelfPlanet>`] tag — see
/// [`TidalConfigTyped`].
#[derive(Debug, Clone)]
pub struct TidalBodyTyped {
    /// Gravitational parameter μ of the perturber.
    pub mu: GravParam<SelfPlanet>,
    /// Position of the perturber in the simulation's root inertial frame.
    pub position_inertial: Position<RootInertial>,
}

impl TidalBodyTyped {
    /// Drop the dimensional annotations and emit the untyped storage form.
    #[inline]
    pub fn to_untyped(&self) -> TidalBody {
        TidalBody {
            mu: self.mu.value,
            position_inertial: self.position_inertial.raw_si(),
        }
    }

    /// Lift an untyped [`TidalBody`] into the typed surface. **The
    /// caller asserts** that `mu` is in m³/s² and `position_inertial`
    /// is in `RootInertial` coordinates.
    #[inline]
    pub fn from_untyped_unchecked(b: &TidalBody) -> Self {
        Self {
            mu: GravParam::<SelfPlanet>::from_si(b.mu),
            position_inertial: Position::<RootInertial>::from_raw_si(b.position_inertial),
        }
    }
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

    /// Lift an untyped [`TidalConfig`] into the typed surface.
    ///
    /// The numeric values are reinterpreted as their SI-base-unit counterparts
    /// (k2 dimensionless, mu in m³/s², radius in m, body positions in m). This
    /// is the inverse of [`Self::to_untyped`] up to the absence of phantom
    /// frame information on input.
    pub fn from_untyped(config: &TidalConfig) -> Self {
        use astrodyn_quantities::ext::{F64Ext, Vec3Ext};
        Self {
            k2: Ratio::new::<uom::si::ratio::ratio>(config.k2),
            mu_primary: F64Ext::m3_per_s2(config.mu_primary),
            radius_primary: Length::new::<uom::si::length::meter>(config.radius_primary),
            tidal_bodies: config
                .tidal_bodies
                .iter()
                .map(|b| TidalBodyTyped {
                    mu: F64Ext::m3_per_s2(b.mu),
                    position_inertial: b.position_inertial.m_at::<RootInertial>(),
                })
                .collect(),
        }
    }
}

/// Typed sibling of [`compute_delta_c20`].
///
/// Same numeric kernel — wraps the result in [`Ratio`] (the physical
/// dimension of the C20 coefficient is dimensionless).
pub fn compute_delta_c20_typed(config: &TidalConfigTyped, t_inertial_pfix: &DMat3) -> Ratio {
    let raw = compute_delta_c20(&config.to_untyped(), t_inertial_pfix);
    Ratio::new::<uom::si::ratio::ratio>(raw)
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
#[allow(
    clippy::float_cmp,
    reason = "tide-tensor tests assert bit-exact zero / typed-vs-raw parity at the type boundary"
)]
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

    /// Typed wrapper round-trips bit-identically to the untyped kernel.
    #[test]
    fn compute_delta_c20_typed_matches_untyped() {
        use astrodyn_quantities::ext::F64Ext;
        use uom::si::length::meter;
        use uom::si::ratio::ratio;
        let untyped = TidalConfig {
            k2: EARTH_K2,
            mu_primary: 3.986004415e14,
            radius_primary: 6_378_137.0,
            tidal_bodies: vec![TidalBody {
                mu: 4902.79980693169e9,
                position_inertial: DVec3::new(0.0, 0.0, 384_400_000.0),
            }],
        };

        let typed = TidalConfigTyped {
            k2: Ratio::new::<ratio>(untyped.k2),
            mu_primary: untyped.mu_primary.m3_per_s2(),
            radius_primary: Length::new::<meter>(untyped.radius_primary),
            tidal_bodies: untyped
                .tidal_bodies
                .iter()
                .map(|b| TidalBodyTyped {
                    mu: b.mu.m3_per_s2(),
                    position_inertial: Position::<RootInertial>::from_raw_si(b.position_inertial),
                })
                .collect(),
        };

        let untyped_delta = compute_delta_c20(&untyped, &DMat3::IDENTITY);
        let typed_delta = compute_delta_c20_typed(&typed, &DMat3::IDENTITY);
        assert_eq!(typed_delta.value, untyped_delta);
    }

    // ---- proptest round-trips (#398) ----------------------------------

    use proptest::prelude::*;

    fn arb_finite_bounded() -> impl Strategy<Value = f64> {
        prop_oneof![
            (1.0e-9_f64..1.0e9_f64),
            (1.0e-9_f64..1.0e9_f64).prop_map(|x| -x),
        ]
    }

    fn arb_dvec3() -> impl Strategy<Value = DVec3> {
        (
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
        )
            .prop_map(|(x, y, z)| DVec3::new(x, y, z))
    }

    fn arb_tidal_body() -> impl Strategy<Value = TidalBody> {
        (arb_finite_bounded(), arb_dvec3()).prop_map(|(mu, position_inertial)| TidalBody {
            mu,
            position_inertial,
        })
    }

    fn arb_tidal_config() -> impl Strategy<Value = TidalConfig> {
        (
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            proptest::collection::vec(arb_tidal_body(), 0..=4),
        )
            .prop_map(
                |(k2, mu_primary, radius_primary, tidal_bodies)| TidalConfig {
                    k2,
                    mu_primary,
                    radius_primary,
                    tidal_bodies,
                },
            )
    }

    proptest! {
        #[test]
        fn round_trip_tidal_body_untyped_typed_untyped(orig in arb_tidal_body()) {
            let typed = TidalBodyTyped::from_untyped_unchecked(&orig);
            prop_assert_eq!(typed.to_untyped(), orig);
        }

        #[test]
        fn round_trip_tidal_body_typed_untyped_typed(orig in arb_tidal_body()) {
            let typed = TidalBodyTyped::from_untyped_unchecked(&orig);
            let lifted = TidalBodyTyped::from_untyped_unchecked(&typed.to_untyped());
            prop_assert_eq!(lifted.to_untyped(), typed.to_untyped());
        }

        #[test]
        fn round_trip_tidal_config_untyped_typed_untyped(orig in arb_tidal_config()) {
            let typed = TidalConfigTyped::from_untyped(&orig);
            prop_assert_eq!(typed.to_untyped(), orig);
        }

        #[test]
        fn round_trip_tidal_config_typed_untyped_typed(orig in arb_tidal_config()) {
            let typed = TidalConfigTyped::from_untyped(&orig);
            let lifted = TidalConfigTyped::from_untyped(&typed.to_untyped());
            prop_assert_eq!(lifted.to_untyped(), typed.to_untyped());
        }
    }
}
