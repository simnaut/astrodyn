//! Convenience bundles for spawning common entity types.
//!
//! These bundles reduce boilerplate when spawning planet, Sun, and Moon
//! entities. They are entirely optional — you can always spawn individual
//! components directly.

use bevy::prelude::*;
use jeod_sim::{FrameTransform, GravityModel, GravitySource, PlanetConfig};

use crate::components::*;

/// Bundle for spawning a gravity source planet entity with rotation and shape.
///
/// Includes all components needed for a planet that participates in gravity,
/// rotation, and geodetic computation.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::PlanetBundle;
/// use jeod_sim::EARTH;
///
/// let mut world = World::new();
/// let earth = world.spawn(PlanetBundle::point_mass("Earth", &EARTH)).id();
/// assert!(world.get_entity(earth).is_ok());
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
            rotation: PlanetFixedRotationC(FrameTransform::from_matrix(glam::DMat3::IDENTITY)),
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
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::SunBundle;
/// use jeod_sim::TranslationalState;
///
/// let mut world = World::new();
/// let sun = world.spawn(SunBundle::new(TranslationalState::default())).id();
/// assert!(world.get_entity(sun).is_ok());
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
            trans: TranslationalStateC::from(state),
        }
    }
}

/// Bundle for spawning the Moon entity.
///
/// Includes [`MoonMarker`] and [`TranslationalStateC`] — the minimum needed
/// by the earth lighting system.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::MoonBundle;
/// use jeod_sim::TranslationalState;
///
/// let mut world = World::new();
/// let moon = world.spawn(MoonBundle::new(TranslationalState::default())).id();
/// assert!(world.get_entity(moon).is_ok());
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
            trans: TranslationalStateC::from(state),
        }
    }
}
