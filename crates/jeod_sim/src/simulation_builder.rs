//! Declarative scenario builder shared by the standalone runner and the
//! Bevy adapter.
//!
//! [`SimulationBuilder`] is a declarative bundle: the user accumulates time
//! settings, gravity sources, vehicles, atmosphere, ephemeris, and mass-tree
//! attachments. The terminal step — actually constructing a runtime
//! simulation — lives in the consumer crate:
//!
//! - `jeod_runner` provides `Simulation::from_builder(b)` (and an extension
//!   trait `SimulationBuilderExt::build`) that materializes a
//!   `Simulation`.
//! - `bevy_jeod` (Phase 9) will provide `Commands::spawn_scenario(b)` that
//!   materializes ECS entities.
//!
//! Phase 6 of #101 relocated this type out of `jeod_runner` so the same
//! scenario library serves both adapters without duplication.
//!
//! Field visibility note: all fields are `pub` so consumer crates can
//! materialize the builder without extra accessors. Pre-release status
//! (#58) makes that exposure acceptable; Phase 10 may re-encapsulate.

use glam::{DMat3, DVec3};

use crate::atmosphere::AtmosphereConfig;
use crate::sources::GravitySourceEntry;
use crate::vehicle_config::VehicleConfig;
use crate::SimulationTime;
use jeod_ephemeris::{Ephemeris, EphemerisBody};

/// A pending mass-tree attachment, resolved when the consumer materializes
/// the builder into a runtime simulation.
#[derive(Debug, Clone)]
pub struct MassTreeAttachment {
    /// Body index of the child.
    pub child_idx: usize,
    /// Body index of the parent.
    pub parent_idx: usize,
    /// Child structural origin in parent's structural frame (m).
    pub offset: DVec3,
    /// Rotation from parent structural frame to child structural frame.
    pub t_parent_child: DMat3,
}

/// Declarative scenario builder.
///
/// See module docs for the consumer-side terminal methods.
pub struct SimulationBuilder {
    pub time: SimulationTime,
    pub dt: f64,
    pub atmosphere: Option<AtmosphereConfig>,
    pub atmosphere_planet_source: Option<usize>,
    pub ephemeris: Option<Ephemeris>,
    pub polar_motion: Option<(f64, f64)>,
    pub sun_source: Option<usize>,
    pub moon_source: Option<usize>,
    pub sources: Vec<(String, GravitySourceEntry)>,
    pub source_ephem_bodies: Vec<Option<(EphemerisBody, EphemerisBody)>>,
    pub bodies: Vec<VehicleConfig>,
    /// Body names for mass tree registration (index matches `bodies`).
    pub mass_tree_names: Vec<Option<String>>,
    /// Pending attachments, resolved during the consumer's terminal `build` /
    /// `spawn` step.
    pub mass_tree_attachments: Vec<MassTreeAttachment>,
}

impl SimulationBuilder {
    /// Start building a simulation with the given time and timestep.
    pub fn new(time: SimulationTime, dt: f64) -> Self {
        Self {
            time,
            dt,
            atmosphere: None,
            atmosphere_planet_source: None,
            ephemeris: None,
            polar_motion: None,
            sun_source: None,
            moon_source: None,
            sources: Vec::new(),
            source_ephem_bodies: Vec::new(),
            bodies: Vec::new(),
            mass_tree_names: Vec::new(),
            mass_tree_attachments: Vec::new(),
        }
    }

    // ── Global config (fluent, consumes self) ──

    /// Set atmosphere configuration with explicit planet source index.
    pub fn atmosphere(mut self, config: AtmosphereConfig, planet_source: usize) -> Self {
        self.atmosphere = Some(config);
        self.atmosphere_planet_source = Some(planet_source);
        self
    }

    /// Set atmosphere configuration from a [`PlanetConfig`](crate::PlanetConfig)
    /// preset.
    pub fn atmosphere_from_planet(
        mut self,
        model: crate::AtmosphereModel,
        planet: &crate::PlanetConfig,
        planet_source: usize,
    ) -> Self {
        self.atmosphere = Some(AtmosphereConfig::from_planet(model, planet));
        self.atmosphere_planet_source = Some(planet_source);
        self
    }

    /// Set ephemeris data (DE421/DE430) for per-step source position updates.
    pub fn ephemeris(mut self, eph: Ephemeris) -> Self {
        self.ephemeris = Some(eph);
        self
    }

    /// Set polar motion parameters (xp, yp) in radians.
    pub fn polar_motion(mut self, xp: f64, yp: f64) -> Self {
        self.polar_motion = Some((xp, yp));
        self
    }

    /// Mark a source as the Sun (for SRP, solar beta, earth lighting).
    pub fn sun(mut self, idx: usize) -> Self {
        self.sun_source = Some(idx);
        self
    }

    /// Mark a source as the Moon (for earth lighting).
    pub fn moon(mut self, idx: usize) -> Self {
        self.moon_source = Some(idx);
        self
    }

    // ── Sources and bodies (&mut self for index returns) ──

    /// Add a gravity source with a name for the frame tree. Returns its index.
    pub fn add_source(&mut self, name: impl Into<String>, entry: GravitySourceEntry) -> usize {
        let idx = self.sources.len();
        self.sources.push((name.into(), entry));
        self.source_ephem_bodies.push(None);
        idx
    }

    /// Configure ephemeris-based position updates for a source.
    pub fn set_source_ephemeris(
        &mut self,
        idx: usize,
        target: EphemerisBody,
        observer: EphemerisBody,
    ) -> &mut Self {
        self.source_ephem_bodies[idx] = Some((target, observer));
        self
    }

    /// Add a vehicle. Returns its index.
    pub fn add_body(&mut self, config: VehicleConfig) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(config);
        self.mass_tree_names.push(None);
        idx
    }

    /// Register a body in the mass tree with the given name.
    ///
    /// Must be called after [`add_body`](Self::add_body). Bodies registered in
    /// the tree can be connected via [`attach_bodies`](Self::attach_bodies).
    ///
    /// # Panics
    /// Panics if the body does not define mass properties.
    pub fn register_in_mass_tree(&mut self, body_idx: usize, name: impl Into<String>) -> &mut Self {
        assert!(
            self.bodies[body_idx].mass.is_some(),
            "register_in_mass_tree: body {body_idx} has no mass properties"
        );
        self.mass_tree_names[body_idx] = Some(name.into());
        self
    }

    /// Declare a mass-tree attachment between two bodies.
    ///
    /// Both bodies must be registered via
    /// [`register_in_mass_tree`](Self::register_in_mass_tree).
    /// The attachment is resolved when the consumer materializes the builder.
    ///
    /// # Panics
    /// Panics if either body has not been registered in the mass tree.
    pub fn attach_bodies(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) -> &mut Self {
        assert!(
            self.mass_tree_names[child_idx].is_some(),
            "attach_bodies: child body {child_idx} not registered in mass tree"
        );
        assert!(
            self.mass_tree_names[parent_idx].is_some(),
            "attach_bodies: parent body {parent_idx} not registered in mass tree"
        );
        self.mass_tree_attachments.push(MassTreeAttachment {
            child_idx,
            parent_idx,
            offset,
            t_parent_child,
        });
        self
    }
}
