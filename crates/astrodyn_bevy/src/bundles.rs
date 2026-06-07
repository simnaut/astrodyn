//! Convenience bundles for spawning common entity types.
//!
//! These bundles reduce boilerplate when spawning planet, Sun, and Moon
//! entities. They are entirely optional — you can always spawn individual
//! components directly.

use astrodyn::{
    FrameTransform, FrameUid, GravityModel, GravitySource, Planet, PlanetConfig, PlanetInertial,
};
use bevy::prelude::*;

use crate::components::*;

/// Bundle for spawning a gravity source planet entity with rotation and shape.
///
/// Includes all components needed for a planet that participates in gravity,
/// rotation, and geodetic computation.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use astrodyn_bevy::PlanetBundle;
/// use astrodyn::EARTH;
///
/// let mut world = World::new();
/// let earth = world.spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH)).id();
/// assert!(world.get_entity(earth).is_ok());
/// ```
#[derive(Bundle)]
pub struct PlanetBundle<P: Planet> {
    /// Bevy `Name` used for debug output.
    pub name: Name,
    /// Gravity source (point-mass or spherical-harmonics).
    pub source: GravitySourceC,
    /// RootInertial-frame position of the source (m).
    pub position: SourceInertialPositionC,
    /// Translational state used by per-step systems.
    pub trans: TranslationalStateC<P>,
    /// `T_inertial→pfix` rotation, updated each step by
    /// `planet_fixed_rotation_system` per the chosen [`RotationModelC`].
    pub rotation: PlanetFixedRotationC<P>,
    /// Sidereal rotation rate sourced from [`PlanetConfig::omega`].
    /// Drives [`PlanetAngularVelocityC`] each step (issue #71 item 1).
    pub omega: PlanetOmegaC,
    /// Angular velocity of the pfix frame relative to inertial, in pfix
    /// coordinates. Written each step by `planet_fixed_rotation_system`.
    pub ang_vel: PlanetAngularVelocityC<P>,
    /// Selector that drives [`Self::rotation`] each step.
    pub rotation_model: RotationModelC,
    /// Planet shape (radii, mu, flattening).
    pub shape: PlanetC,
    /// The source's inertial-frame identity (issue #664) —
    /// `FrameUid::of::<PlanetInertial<P>>()`, copied onto the source's
    /// frame entity at registration; the pfix sibling is derived via
    /// [`astrodyn::pfix_sibling_uid`].
    pub uid: FrameUidC,
}

impl<P: Planet> PlanetBundle<P> {
    /// Create a planet bundle from a [`PlanetConfig`] with a custom gravity source.
    ///
    /// Use this when you have spherical harmonics data or a custom mu.
    pub fn from_config(name: &str, config: &PlanetConfig, source: GravitySource) -> Self {
        Self {
            name: Name::new(name.to_string()),
            source: GravitySourceC(source),
            position: SourceInertialPositionC::default(),
            trans: TranslationalStateC::<P>::default(),
            // allowed: IDENTITY placeholder; planet_fixed_rotation_system overwrites on tick 1
            rotation: PlanetFixedRotationC::<P>(FrameTransform::from_matrix(glam::DMat3::IDENTITY)),
            omega: PlanetOmegaC(config.omega),
            ang_vel: PlanetAngularVelocityC::<P>::default(),
            rotation_model: RotationModelC(config.rotation_model),
            shape: PlanetC(config.shape),
            uid: FrameUidC(FrameUid::of::<PlanetInertial<P>>()),
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

    /// Bundle with only the components needed for point-mass gravity:
    /// [`Name`], [`GravitySourceC`], [`SourceInertialPositionC`], and
    /// [`TranslationalStateC<P>`].
    ///
    /// Omits the rotation/shape/RNP components that the full
    /// [`Self::point_mass`] / [`Self::from_config`] bundles insert
    /// ([`PlanetFixedRotationC`], [`PlanetOmegaC`],
    /// [`PlanetAngularVelocityC`], [`RotationModelC`], [`PlanetC`]). Those
    /// extra components are what activate the planet-fixed-rotation,
    /// atmosphere, and geodetic systems each step; tests and missions that
    /// model point-mass gravity only must not insert them or the schedule
    /// will run physics they aren't configured for.
    ///
    /// Use [`Self::point_mass`] for a planet that participates in rotation
    /// or geodetic computation.
    ///
    /// # Example
    /// ```
    /// use bevy::prelude::*;
    /// use astrodyn_bevy::{
    ///     GravitySourceC, PlanetBundle, SourceInertialPositionC, TranslationalStateC,
    /// };
    /// use astrodyn::EARTH;
    ///
    /// let mut world = World::new();
    /// let earth = world
    ///     .spawn(PlanetBundle::<astrodyn::Earth>::point_mass_only(
    ///         "Earth",
    ///         astrodyn::GravitySource {
    ///             mu: EARTH.shape.mu,
    ///             model: astrodyn::GravityModel::PointMass,
    ///         },
    ///     ))
    ///     .id();
    /// let e = world.entity(earth);
    /// assert!(e.contains::<Name>());
    /// assert!(e.contains::<GravitySourceC>());
    /// assert!(e.contains::<SourceInertialPositionC>());
    /// assert!(e.contains::<TranslationalStateC<astrodyn::Earth>>());
    /// ```
    pub fn point_mass_only(
        name: impl Into<Name>,
        source: GravitySource,
    ) -> (
        Name,
        GravitySourceC,
        SourceInertialPositionC,
        TranslationalStateC<P>,
        FrameUidC,
    ) {
        (
            name.into(),
            GravitySourceC(source),
            SourceInertialPositionC::default(),
            TranslationalStateC::<P>::default(),
            FrameUidC(FrameUid::of::<PlanetInertial<P>>()),
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
/// use astrodyn_bevy::SunBundle;
/// use astrodyn::TranslationalState;
///
/// let mut world = World::new();
/// let sun = world.spawn(SunBundle::<astrodyn::Earth>::new(TranslationalState::default())).id();
/// assert!(world.get_entity(sun).is_ok());
/// ```
#[derive(Bundle)]
pub struct SunBundle<P: Planet> {
    /// Bevy `Name`, defaults to `"Sun"`.
    pub name: Name,
    /// Discriminator queried by SRP / solar-beta / lighting systems.
    pub marker: SunMarker,
    /// Position used by SRP / solar-beta / lighting systems. Tagged
    /// with the planet `<P>` whose body-side instantiation reads this
    /// component (the systems mix body and Sun state at the type
    /// level, so they must share a planet phantom — by convention the
    /// simulation's central planet, which is `<P>`).
    pub trans: TranslationalStateC<P>,
}

impl<P: Planet> SunBundle<P> {
    /// Build a Sun bundle from an inertial translational state.
    pub fn new(state: astrodyn::TranslationalState) -> Self {
        Self {
            name: Name::new("Sun"),
            marker: SunMarker,
            // allowed: typed↔raw kernel boundary at the public bundle
            // ctor — named-method opt-in (see #397).
            trans: TranslationalStateC::<P>::from_untyped(state),
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
/// use astrodyn_bevy::MoonBundle;
/// use astrodyn::TranslationalState;
///
/// let mut world = World::new();
/// let moon = world.spawn(MoonBundle::<astrodyn::Earth>::new(TranslationalState::default())).id();
/// assert!(world.get_entity(moon).is_ok());
/// ```
#[derive(Bundle)]
pub struct MoonBundle<P: Planet> {
    /// Bevy `Name`, defaults to `"Moon"`.
    pub name: Name,
    /// Discriminator queried by the earth-lighting system.
    pub marker: MoonMarker,
    /// Position used by the earth-lighting system. Tagged with the
    /// planet `<P>` whose body-side instantiation reads this component
    /// (see `SunBundle` for the same convention rationale).
    pub trans: TranslationalStateC<P>,
}

impl<P: Planet> MoonBundle<P> {
    /// Build a Moon bundle from an inertial translational state.
    pub fn new(state: astrodyn::TranslationalState) -> Self {
        Self {
            name: Name::new("Moon"),
            marker: MoonMarker,
            // allowed: typed↔raw kernel boundary at the public bundle
            // ctor — named-method opt-in (see #397).
            trans: TranslationalStateC::<P>::from_untyped(state),
        }
    }
}
