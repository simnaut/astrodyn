//! Convenience bundles for spawning common entity types.
//!
//! These bundles reduce boilerplate when spawning planet, Sun, and Moon
//! entities. They are entirely optional — you can always spawn individual
//! components directly.

use bevy::prelude::*;
use jeod_sim::{GravityModel, GravitySource, PlanetConfig};

use crate::components::*;

/// Bundle for spawning a gravity source planet entity with rotation and shape.
///
/// Includes all components needed for a planet that participates in gravity,
/// rotation, and geodetic computation.
///
/// # Example
/// ```ignore
/// use bevy_jeod::PlanetBundle;
/// use jeod_sim::EARTH;
///
/// let earth = commands.spawn(PlanetBundle::point_mass("Earth", &EARTH)).id();
/// ```
#[derive(Bundle)]
pub struct PlanetBundle {
    pub name: Name,
    pub source: GravitySourceC,
    pub position: SourceInertialPositionC,
    pub trans: TranslationalStateC,
    pub rotation: PlanetFixedRotationC,
    pub rotation_model: RotationModelC,
    pub shape: PlanetC,
}

impl PlanetBundle {
    /// Create a planet bundle from a [`PlanetConfig`] with a custom gravity source.
    ///
    /// Use this when you have spherical harmonics data or a custom mu.
    pub fn from_config(name: &str, config: &PlanetConfig, source: GravitySource) -> Self {
        Self {
            name: Name::new(name.to_string()),
            source: GravitySourceC(source),
            position: SourceInertialPositionC::default(),
            trans: TranslationalStateC::default(),
            rotation: PlanetFixedRotationC(glam::DMat3::IDENTITY),
            rotation_model: RotationModelC(config.rotation_model),
            shape: PlanetC(config.shape),
        }
    }

    /// Create a planet bundle with point-mass gravity from a [`PlanetConfig`].
    ///
    /// Uses `mu` from the planet config's shape.
    pub fn point_mass(name: &str, config: &PlanetConfig) -> Self {
        Self::from_config(
            name,
            config,
            GravitySource {
                mu: config.shape.mu,
                model: GravityModel::PointMass,
            },
        )
    }
}

/// Bundle for spawning the Sun entity.
///
/// Includes [`SunMarker`] and [`TranslationalStateC`] — the minimum needed
/// by SRP and solar-beta systems.
///
/// # Example
/// ```ignore
/// use bevy_jeod::SunBundle;
///
/// commands.spawn(SunBundle::new(sun_state));
/// ```
#[derive(Bundle)]
pub struct SunBundle {
    pub name: Name,
    pub marker: SunMarker,
    pub trans: TranslationalStateC,
}

impl SunBundle {
    pub fn new(state: jeod_sim::TranslationalState) -> Self {
        Self {
            name: Name::new("Sun"),
            marker: SunMarker,
            trans: TranslationalStateC(state),
        }
    }
}

/// Bundle for spawning the Moon entity.
///
/// Includes [`MoonMarker`] and [`TranslationalStateC`] — the minimum needed
/// by the earth lighting system.
///
/// # Example
/// ```ignore
/// use bevy_jeod::MoonBundle;
///
/// commands.spawn(MoonBundle::new(moon_state));
/// ```
#[derive(Bundle)]
pub struct MoonBundle {
    pub name: Name,
    pub marker: MoonMarker,
    pub trans: TranslationalStateC,
}

impl MoonBundle {
    pub fn new(state: jeod_sim::TranslationalState) -> Self {
        Self {
            name: Name::new("Moon"),
            marker: MoonMarker,
            trans: TranslationalStateC(state),
        }
    }
}
