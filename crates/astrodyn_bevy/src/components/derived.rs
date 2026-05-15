//! Bevy `Component` newtypes for derived states — orbital elements,
//! Euler angles, LVLH frame, geodetic state, solar beta angle, and
//! Earth lighting (eclipse / albedo) — plus the per-state config
//! components that gate computation.

use astrodyn::{Angle, Earth, Planet};
use bevy::prelude::*;

// ── Derived State Configuration ──

/// Configuration for orbital elements computation.
///
/// The `gravity_source` entity is queried for `GravitySourceC` to obtain `mu`.
/// Presence of this component + `OrbitalElementsC` on an entity enables
/// per-step orbital elements computation in `AstrodynSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
#[require(OrbitalElementsC::<Earth>)]
pub struct OrbitalElementsConfigC {
    /// Gravity source entity supplying `mu` for the conversion.
    pub gravity_source: Entity,
}

/// Configuration for Euler angle decomposition.
///
/// Presence of this component + `EulerAnglesC` on an entity enables
/// per-step Euler angle computation in `AstrodynSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
#[require(EulerAnglesC)]
pub struct EulerAnglesConfigC {
    /// Euler-angle decomposition convention (e.g., 3-2-1 yaw/pitch/roll).
    pub sequence: astrodyn::EulerSequence,
}

/// Configuration for geodetic state computation.
///
/// The `planet` entity is queried for `PlanetFixedRotationC<P>` to obtain
/// the inertial→planet-fixed rotation each step; the ellipsoid radii
/// `r_eq` / `r_pol` are carried directly on this config (mirroring the
/// runner's `body.geodetic_planet: (idx, r_eq, r_pol)` shape) so a
/// scenario whose source entity does not carry the planet shape — e.g.
/// the `SimulationBuilder::populate_app` path, which inserts only the
/// gravity/rotation components on the source — can still drive
/// geodetic computation. Presence of this component + `GeodeticStateC`
/// on an entity enables per-step geodetic computation in
/// `AstrodynSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
#[require(GeodeticStateC)]
pub struct GeodeticConfigC {
    /// Planet entity supplying `T_inertial→pfix` (`PlanetFixedRotationC<P>`).
    pub planet: Entity,
    /// Equatorial radius of the reference ellipsoid (m).
    pub r_eq: f64,
    /// Polar radius of the reference ellipsoid (m).
    pub r_pol: f64,
}

// ── Derived State Outputs ──

/// Orbital elements computed each step.
///
/// Written by `orbital_elements_system` for entities that also have
/// `OrbitalElementsConfigC`. Generic over the planet `P` whose
/// gravitational parameter `mu` was used in the conversion. Every call
/// site must pin `P` explicitly — there is no fallback.
#[derive(Component, Debug, Clone)]
pub struct OrbitalElementsC<P: Planet>(pub astrodyn::OrbitalElements<P>);

impl<P: Planet> Default for OrbitalElementsC<P> {
    #[inline]
    fn default() -> Self {
        Self(astrodyn::OrbitalElements::default())
    }
}

/// Euler angles `[phi, theta, psi]` computed each step.
///
/// Written by `euler_angles_system` for entities that also have
/// `EulerAnglesConfigC`. Each component is a [`Angle`] (uom radian-backed
/// scalar) so consumers don't have to remember the radian convention.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EulerAnglesC(pub [Angle; 3]);

/// LVLH (Local Vertical Local Horizontal) frame computed each step.
///
/// Presence of this component alone enables computation — no separate
/// config component needed (only requires translational state).
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct LvlhFrameC(pub astrodyn::LvlhFrame);

/// Geodetic state (latitude, longitude, altitude) computed each step.
///
/// Written by `geodetic_system` for entities that also have `GeodeticConfigC`.
///
/// # Numerical stability at the poles
///
/// The inner [`astrodyn::GeodeticState`] inherits the longitude
/// instability of the underlying chart: at 89.8° latitude, longitude has
/// roughly `3.7e-6 rad/m` sensitivity to planet-fixed position drift, and
/// at the pole itself the kernel assigns `longitude = 0.0` by convention
/// because all meridians converge. Mission code reading this component
/// near the poles should treat longitude as low-confidence (the Tier 3 NED
/// suite uses `~3.3e-5 rad` polar tolerance vs `~6.5e-8 rad` inclined-orbit
/// tolerance). This is fundamental geometry, not a bug — see
/// [`astrodyn::GeodeticState`] for the full rationale.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct GeodeticStateC(pub astrodyn::GeodeticState);

/// Solar beta angle (radians) computed each step.
///
/// Presence of this component alone enables computation — requires a
/// `SunMarker` entity to exist in the world.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct SolarBetaC(pub f64);

/// Configuration for Earth lighting (eclipse/albedo) computation.
///
/// Requires `SunMarker` and `MoonMarker` entities to exist in the world.
/// Presence of this component + `EarthLightingStateC` on an entity enables
/// per-step earth lighting computation in `AstrodynSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
#[require(EarthLightingStateC)]
pub struct EarthLightingConfigC {
    /// Earth equatorial radius (m).
    pub earth_radius: f64,
    /// Moon mean radius (m).
    pub moon_radius: f64,
    /// Sun mean radius (m).
    pub sun_radius: f64,
}

/// Earth lighting state computed each step.
///
/// Written by `earth_lighting_system` for entities that also have
/// `EarthLightingConfigC`.
#[derive(Component, Debug, Clone, Default)]
pub struct EarthLightingStateC(pub astrodyn::EarthLightingState);
