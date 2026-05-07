//! Gravity source registry entries.
//!
//! [`GravitySourceEntry`] is the user-facing description of a single gravity
//! source (Earth, Moon, Sun, …) consumed by [`SimulationBuilder::add_source`](crate::SimulationBuilder::add_source)
//! and ECS adapters. Phase 6 of #101 relocated this type out of `astrodyn_runner`
//! so the runner and the future Bevy adapter share one description.

use glam::DMat3;

use crate::planet_config::PlanetConfig;
use crate::rotation_model::RotationModel;
use astrodyn_gravity::{GravityModel, GravitySource, SphericalHarmonicsData};
use astrodyn_quantities::aliases::{Position, Velocity};
use astrodyn_quantities::frame::RootInertial;

/// Entry in the gravity source table.
///
/// Gravity sources are referenced by index (`usize`) from body gravity controls.
///
/// # RF.10 structural guard
///
/// `position` and `velocity` are typed `Position<RootInertial>` /
/// `Velocity<RootInertial>` — gravity arithmetic in `accumulate_gravity`
/// does `body_root_pos - source.position` and the structural guard
/// requires the body to be in the same frame. Mixing
/// `Position<IntegrationFrame>` (the body's untyped storage) with
/// `Position<RootInertial>` (the source) is a compile error, forcing
/// callers through `body.trans.to_inertial(&integ_origin)`.
///
/// **`Position<IntegrationFrame> - source.position` does not compile:**
///
/// ```compile_fail
/// use astrodyn::{GravitySourceEntry, GravitySource, GravityModel};
/// use astrodyn_quantities::prelude::*;
/// let entry = GravitySourceEntry {
///     source: GravitySource { mu: 1.0, model: GravityModel::PointMass },
///     position: Position::<RootInertial>::zero(),
///     velocity: Velocity::<RootInertial>::zero(),
///     t_inertial_pfix: None,
///     rotation_model: astrodyn::RotationModel::None,
///     delta_c20: 0.0,
///     tidal_config: None,
///     planet_omega: 0.0,
///     central: true,
/// };
/// let body_pos: Position<IntegrationFrame> = Position::zero();
/// let _bug = body_pos - entry.position;   // frames mismatch
/// ```
pub struct GravitySourceEntry {
    /// Physical gravity source (mu, model).
    pub source: GravitySource,
    /// Position in the simulation's root inertial frame (m). For
    /// Earth-centered sims, Earth is at origin.
    pub position: Position<RootInertial>,
    /// Velocity in the simulation's root inertial frame (m/s). Required
    /// for relativistic corrections. Zero for stationary sources (e.g.,
    /// central body at origin).
    pub velocity: Velocity<RootInertial>,
    /// RootInertial-to-planet-fixed rotation matrix. Updated each step when
    /// `rotation_model` is not `None`. If `None`, no rotation is applied
    /// (point-mass only).
    pub t_inertial_pfix: Option<DMat3>,
    /// Rotation model for updating `t_inertial_pfix` each step.
    pub rotation_model: RotationModel,
    /// Tidal ΔC20 to add to the base C20 coefficient before spherical harmonics
    /// evaluation. Updated each step by the environment stage if tidal effects
    /// are configured. Zero when no tides.
    pub delta_c20: f64,
    /// Tidal configuration. When `Some`, the simulation computes ΔC20 each step.
    pub tidal_config: Option<astrodyn_gravity::tides::TidalConfig>,
    /// Sidereal angular velocity (rad/s) for the planet-fixed frame's
    /// `ang_vel_this`. Sourced from [`PlanetConfig::omega`]. Set to 0.0
    /// for sources without a rotation model.
    pub planet_omega: f64,
    /// Whether this is the central body (uses the root frame in the tree).
    /// Set automatically by [`central_body`](Self::central_body) and
    /// [`central_body_sh`](Self::central_body_sh).
    pub central: bool,
}

impl GravitySourceEntry {
    /// Create a new gravity source entry without tidal effects.
    ///
    /// `rotation_model` defaults to `None`. Set it explicitly after construction
    /// (or use struct literal syntax) to enable per-step rotation updates.
    pub fn new(
        source: GravitySource,
        position: Position<RootInertial>,
        t_inertial_pfix: Option<DMat3>,
    ) -> Self {
        Self {
            source,
            position,
            velocity: Velocity::<RootInertial>::zero(),
            t_inertial_pfix,
            rotation_model: RotationModel::None,
            planet_omega: 0.0,
            delta_c20: 0.0,
            tidal_config: None,
            central: false,
        }
    }

    /// Central body at the origin with point-mass gravity and rotation from a
    /// [`PlanetConfig`] preset.
    ///
    /// Sets rotation model and initial identity rotation matrix (if the planet
    /// has a rotation model). Position and velocity are zero (central body).
    pub fn central_body(planet: &PlanetConfig) -> Self {
        Self {
            source: GravitySource {
                mu: planet.shape.mu,
                model: GravityModel::PointMass,
            },
            position: Position::<RootInertial>::zero(),
            velocity: Velocity::<RootInertial>::zero(),
            t_inertial_pfix: if planet.rotation_model != RotationModel::None {
                Some(DMat3::IDENTITY)
            } else {
                None
            },
            rotation_model: planet.rotation_model,
            planet_omega: planet.omega,
            delta_c20: 0.0,
            tidal_config: None,
            central: true,
        }
    }

    /// Central body at the origin with spherical harmonics gravity and rotation
    /// from a [`PlanetConfig`] preset.
    ///
    /// Uses `mu` from the spherical harmonics data (which may differ slightly
    /// from the planet preset's geodetic mu).
    pub fn central_body_sh(planet: &PlanetConfig, sh_data: SphericalHarmonicsData) -> Self {
        Self {
            source: GravitySource {
                mu: sh_data.mu,
                model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
            },
            position: Position::<RootInertial>::zero(),
            velocity: Velocity::<RootInertial>::zero(),
            t_inertial_pfix: if planet.rotation_model != RotationModel::None {
                Some(DMat3::IDENTITY)
            } else {
                None
            },
            rotation_model: planet.rotation_model,
            planet_omega: planet.omega,
            delta_c20: 0.0,
            tidal_config: None,
            central: true,
        }
    }

    /// Third body (perturbation source) at a given position.
    ///
    /// Point-mass only, no rotation. Typical use: Sun or Moon as a third-body
    /// perturbation in Earth-centered integration.
    pub fn third_body(planet: &PlanetConfig, position: Position<RootInertial>) -> Self {
        Self {
            source: GravitySource {
                mu: planet.shape.mu,
                model: GravityModel::PointMass,
            },
            position,
            velocity: Velocity::<RootInertial>::zero(),
            t_inertial_pfix: None,
            rotation_model: RotationModel::None,
            planet_omega: 0.0,
            delta_c20: 0.0,
            tidal_config: None,
            central: false,
        }
    }

    /// Add tidal configuration (builder-style, consumes and returns self).
    pub fn with_tidal(mut self, config: astrodyn_gravity::tides::TidalConfig) -> Self {
        self.tidal_config = Some(config);
        self
    }
}
