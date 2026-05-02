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
    /// Bevy `Name` used for debug output.
    pub name: Name,
    /// Gravity source (point-mass or spherical-harmonics).
    pub source: GravitySourceC,
    /// RootInertial-frame position of the source (m).
    pub position: SourceInertialPositionC,
    /// Translational state used by per-step systems.
    pub trans: TranslationalStateC,
    /// `T_inertial→pfix` rotation, updated each step by
    /// `planet_fixed_rotation_system` per the chosen [`RotationModelC`].
    pub rotation: PlanetFixedRotationC,
    /// Selector that drives [`Self::rotation`] each step.
    pub rotation_model: RotationModelC,
    /// Planet shape (radii, mu, flattening).
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
            // allowed: IDENTITY placeholder; planet_fixed_rotation_system overwrites on tick 1
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
    /// Bevy `Name`, defaults to `"Sun"`.
    pub name: Name,
    /// Discriminator queried by SRP / solar-beta / lighting systems.
    pub marker: SunMarker,
    /// RootInertial position used by the same systems.
    pub trans: TranslationalStateC,
}

impl SunBundle {
    /// Build a Sun bundle from an inertial translational state.
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
    /// Bevy `Name`, defaults to `"Moon"`.
    pub name: Name,
    /// Discriminator queried by the earth-lighting system.
    pub marker: MoonMarker,
    /// RootInertial position used by the earth-lighting system.
    pub trans: TranslationalStateC,
}

impl MoonBundle {
    /// Build a Moon bundle from an inertial translational state.
    pub fn new(state: jeod_sim::TranslationalState) -> Self {
        Self {
            name: Name::new("Moon"),
            marker: MoonMarker,
            trans: TranslationalStateC::from(state),
        }
    }
}
