//! Planet configuration presets — single source of truth for simulation setup.
//!
//! Each [`PlanetConfig`] bundles geometric parameters ([`PlanetShape`]) with
//! operational parameters (rotation model, angular velocity, shadow radius)
//! needed by both the standalone runner and Bevy adapter.

use crate::RotationModel;
use astrodyn_planet::PlanetShape;
use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::ext::F64Ext;
use astrodyn_quantities::frame::SelfPlanet;
use uom::si::f64::{AngularVelocity, Length};

/// Complete planet configuration for simulation setup.
///
/// Combines geometric parameters ([`PlanetShape`]: mu, r_eq, r_pol) with
/// operational parameters needed by gravity sources, atmosphere models,
/// geodetic computation, and SRP eclipse calculation.
///
/// Use the provided constants ([`EARTH`], [`MOON`], [`SUN`], [`MARS`]) as
/// the single source of truth when configuring simulations.
#[derive(Debug, Clone, Copy)]
pub struct PlanetConfig {
    /// Geometric parameters (mu, r_eq, r_pol, flattening).
    pub shape: PlanetShape,
    /// How to update the inertial-to-planet-fixed rotation each step.
    pub rotation_model: RotationModel,
    /// Sidereal angular velocity (rad/s) for atmosphere co-rotation wind.
    /// Set to 0.0 for bodies without atmosphere or without wind.
    pub omega: f64,
    /// Body radius (m) for SRP eclipse (shadow) computation.
    /// Typically the mean equatorial radius.
    pub shadow_radius: f64,
}

impl PlanetConfig {
    /// Typed accessor: gravitational parameter as
    /// [`GravParam<SelfPlanet>`] (m³/s²). Delegates to
    /// [`PlanetShape::mu_typed`] so the unit-wrapping lives in one
    /// place. The phantom is [`SelfPlanet`] because `PlanetConfig`
    /// instances are assigned to entities at runtime — the planet
    /// identity is dynamic at this surface. Pin a planet-tagged μ
    /// at the consumer side via the `mu_*()` constants in
    /// `astrodyn::recipes::constants` or via
    /// `mu_typed().relabel::<P>()`.
    // JEOD_INV: TS.01 — `PlanetConfig` is a runtime-resolved boundary:
    // its `mu_typed()` returns `<SelfPlanet>` because the entity that
    // holds this config decides the planet identity dynamically. The
    // typed `recipes::constants::mu_*()` accessors carry `<P>` for
    // mission code.
    #[inline]
    pub fn mu_typed(&self) -> GravParam<SelfPlanet> {
        self.shape.mu_typed()
    }

    /// Typed accessor: sidereal angular velocity as
    /// [`uom::si::f64::AngularVelocity`] (rad/s).
    #[inline]
    pub fn omega_typed(&self) -> AngularVelocity {
        self.omega.rad_per_s()
    }

    /// Typed accessor: SRP shadow body radius as [`Length`] (meters).
    #[inline]
    pub fn shadow_radius_typed(&self) -> Length {
        use uom::si::length::meter;
        Length::new::<meter>(self.shadow_radius)
    }
}

/// Earth — WGS84 ellipsoid, IAU 2000A precession-nutation rotation.
///
/// Constants from JEOD source:
/// - Geometry: `planet/data/src/earth.cc` (WGS84)
/// - Gravity: `earth_GGM05C.cc` (mu = 3.986004415e14 m³/s²)
/// - Rotation: IAU 2000A precession-nutation + GAST + optional polar motion
/// - Angular velocity: `RNPJ2000_data.cc` (7.292115146706388e-5 rad/s)
pub const EARTH: PlanetConfig = PlanetConfig {
    shape: astrodyn_planet::presets::EARTH,
    rotation_model: RotationModel::EarthRNP,
    omega: 7.292_115_146_706_388e-5,
    shadow_radius: 6_378_137.0,
};

/// Moon — IAU 2009 pole + prime meridian model.
///
/// Constants from JEOD source:
/// - Geometry: `planet/data/src/moon.cc`
/// - Gravity: `moon_GRAIL150.cc` (mu = 4902.79980693169e9 m³/s²)
/// - Rotation: IAU 2009 (use [`MoonDE421`](RotationModel::MoonDE421) for
///   high-fidelity libration from BPC data)
pub const MOON: PlanetConfig = PlanetConfig {
    shape: astrodyn_planet::presets::MOON,
    rotation_model: RotationModel::MoonIAU,
    omega: 0.0,
    shadow_radius: 1_738_140.0,
};

/// Sun — no rotation model (point source for SRP and third-body gravity).
///
/// Constants from JEOD source:
/// - Geometry: `planet/data/src/sun.cc`
/// - Gravity: `sun_spherical.cc` (mu = 1.32712440e20 m³/s²)
pub const SUN: PlanetConfig = PlanetConfig {
    shape: astrodyn_planet::presets::SUN,
    rotation_model: RotationModel::None,
    omega: 0.0,
    shadow_radius: 696_000_000.0,
};

/// Mars — IAU pole orientation + spin + nutation Fourier series.
///
/// Constants from JEOD source:
/// - Geometry: `planet/data/src/mars.cc`
/// - Gravity: `mars_MRO110B2.cc` (mu = 4.2828374527e13 m³/s²)
/// - Rotation: IAU 2009 (matching JEOD's `RNPMars`)
/// - Angular velocity: 7.088218e-5 rad/s
pub const MARS: PlanetConfig = PlanetConfig {
    shape: astrodyn_planet::presets::MARS,
    rotation_model: RotationModel::MarsIAU,
    omega: 7.088_218e-5,
    shadow_radius: 3_396_000.0,
};

/// Jupiter — shape preset for rendering / sizing an outer planet.
///
/// Constants from JEOD source:
/// - Geometry: `planet/data/src/jupiter.cc` (1-bar oblate ellipsoid)
/// - Gravity: `gravity/data/src/jupiter_spherical.cc` (mu = 1.26731229e17 m³/s²)
///
/// No spin model is wired ([`RotationModel::None`], `omega = 0`): astrodyn does
/// not propagate or spin Jupiter, so this preset carries geometry + μ only,
/// for sizing and placement. A body-fixed orientation, when one is needed,
/// comes from the ephemeris IAU rotation — never from this config's `omega`.
pub const JUPITER: PlanetConfig = PlanetConfig {
    shape: astrodyn_planet::presets::JUPITER,
    rotation_model: RotationModel::None,
    omega: 0.0,
    shadow_radius: 71_492_000.0,
};

/// Saturn — shape preset for rendering / sizing an outer planet.
///
/// Constants from JEOD source:
/// - Geometry: `ephemerides/verif/SIM_prop_planet/data/src/saturn_planet.cc`
/// - Gravity: `saturn_spherical_gravity.cc` (mu = 3.7931187e16 m³/s²)
///
/// As with [`JUPITER`], no spin model is wired — geometry + μ only.
pub const SATURN: PlanetConfig = PlanetConfig {
    shape: astrodyn_planet::presets::SATURN,
    rotation_model: RotationModel::None,
    omega: 0.0,
    shadow_radius: 60_268_000.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_shadow_radius_matches_r_eq() {
        assert_eq!(EARTH.shadow_radius, EARTH.shape.r_eq());
    }

    #[test]
    fn sun_no_rotation() {
        assert_eq!(SUN.rotation_model, RotationModel::None);
    }

    #[test]
    fn all_configs_positive_radii() {
        for config in [EARTH, MOON, SUN, MARS, JUPITER, SATURN] {
            assert!(
                config.shadow_radius > 0.0,
                "{}: shadow_radius must be positive",
                config.shape.name
            );
            assert!(
                config.omega >= 0.0,
                "{}: omega must be non-negative",
                config.shape.name
            );
            // PlanetShape::new already enforces r_pol ≤ r_eq (oblate/spherical);
            // re-assert at the config level so a fat-fingered preset trips here.
            assert!(
                config.shape.r_pol() <= config.shape.r_eq(),
                "{}: r_pol must not exceed r_eq (oblate only)",
                config.shape.name
            );
        }
    }

    // Spot-check the JEOD-sourced outer-planet geometry so a transcription
    // error in the body constants is caught here rather than downstream.
    // Bit-exact comparison (`to_bits`) keeps the denied `clippy::float_cmp`
    // lint quiet while still pinning the exact JEOD values.
    #[test]
    fn jupiter_saturn_geometry_matches_jeod() {
        // Full geometry (r_eq, flattening, derived r_pol) plus μ, pinned to the
        // exact JEOD source values. Jupiter: jupiter.cc / jupiter_spherical.cc.
        assert_eq!(JUPITER.shape.r_eq().to_bits(), 71_492_000.0_f64.to_bits());
        assert_eq!(JUPITER.shape.flat_coeff.to_bits(), 0.06487_f64.to_bits());
        assert_eq!(
            JUPITER.shape.r_pol().to_bits(),
            (71_492_000.0_f64 * (1.0 - 0.06487)).to_bits()
        );
        assert_eq!(JUPITER.shape.mu.to_bits(), 1.267_312_29e17_f64.to_bits());
        // Saturn: saturn_planet.cc / saturn_spherical_gravity.cc.
        assert_eq!(SATURN.shape.r_eq().to_bits(), 60_268_000.0_f64.to_bits());
        assert_eq!(SATURN.shape.flat_coeff.to_bits(), 0.09796_f64.to_bits());
        assert_eq!(
            SATURN.shape.r_pol().to_bits(),
            (60_268_000.0_f64 * (1.0 - 0.09796)).to_bits()
        );
        assert_eq!(SATURN.shape.mu.to_bits(), 3.793_118_7e16_f64.to_bits());
    }
}
