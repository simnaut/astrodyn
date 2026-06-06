//! Declarative scenario builder shared by the standalone runner and the
//! Bevy adapter.
//!
//! [`SimulationBuilder`] is a declarative bundle: the user accumulates time
//! settings, gravity sources, vehicles, atmosphere, ephemeris, and mass-tree
//! attachments. The terminal step — actually constructing a runtime
//! simulation — lives in the consumer crate:
//!
//! - `astrodyn_runner` provides `Simulation::from_builder(b)` (and an extension
//!   trait `SimulationBuilderExt::build`) that materializes a
//!   `Simulation`.
//! - `astrodyn_bevy` (Phase 9) will provide `Commands::spawn_scenario(b)` that
//!   materializes ECS entities.
//!
//! Phase 6 of #101 relocated this type out of `astrodyn_runner` so the same
//! scenario library serves both adapters without duplication.
//!
//! Field visibility note: all fields are `pub` so consumer crates can
//! materialize the builder without extra accessors. Pre-release status
//! (#58) makes that exposure acceptable; Phase 10 may re-encapsulate.

use astrodyn_quantities::frame_descriptor::FrameUid;
use glam::{DMat3, DVec3};

use crate::atmosphere::AtmosphereConfig;
use crate::sources::GravitySourceEntry;
use crate::vehicle_config::VehicleConfig;
use crate::SimulationTime;
use astrodyn_ephemeris::{Ephemeris, EphemerisBody};

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
    /// Initial simulation time.
    pub time: SimulationTime,
    /// Fixed integrator timestep in seconds.
    pub dt: f64,
    /// Optional atmosphere configuration (model + radii + wind).
    pub atmosphere: Option<AtmosphereConfig>,
    /// Source index of the planet whose rotation drives geodetic conversion
    /// for atmospheric evaluation. `None` when [`Self::atmosphere`] is `None`.
    pub atmosphere_planet_source: Option<usize>,
    /// Optional DE4xx ephemeris used to update source positions per step.
    pub ephemeris: Option<Ephemeris>,
    /// Optional polar-motion `(xp, yp)` in radians applied to Earth rotation.
    pub polar_motion: Option<(f64, f64)>,
    /// Source index of the Sun, when needed for SRP / solar beta / lighting.
    pub sun_source: Option<usize>,
    /// Source index of the Moon, when needed for Earth lighting.
    pub moon_source: Option<usize>,
    /// Gravity sources keyed by name in declaration order.
    pub sources: Vec<(String, GravitySourceEntry)>,
    /// Pre-minted frame identities `(inertial, pfix)` per source; index
    /// matches [`Self::sources`]. `Some` for sources added via
    /// [`Self::add_source_typed`] (identity travels as configuration
    /// data); `None` falls back to the runner's sealed-planet name
    /// dispatch at materialization.
    pub source_uids: Vec<Option<(FrameUid, FrameUid)>>,
    /// Per-source `(body, parent)` ephemeris bodies; index matches
    /// [`Self::sources`]. `None` for sources that do not move via DE4xx.
    pub source_ephem_bodies: Vec<Option<(EphemerisBody, EphemerisBody)>>,
    /// Vehicles in declaration order. Body index = position in this vec.
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
            source_uids: Vec::new(),
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
        self.source_uids.push(None);
        self.sources.push((name.into(), entry));
        self.source_ephem_bodies.push(None);
        idx
    }

    /// Typed sibling of [`Self::add_source`]: mints the source's frame
    /// identities from the planet marker `P` and carries them as
    /// configuration data to the consumer (the runner's
    /// `add_source_with_uids`, or the Bevy adapter in issue #664). Use for
    /// planets minted downstream via `define_planet!`; the six built-in
    /// planets may use the plain `add_source` name dispatch.
    pub fn add_source_typed<P: astrodyn_quantities::frame::Planet>(
        &mut self,
        name: impl Into<String>,
        entry: GravitySourceEntry,
    ) -> usize {
        let idx = self.sources.len();
        self.source_uids.push(Some((
            FrameUid::of::<astrodyn_quantities::frame::PlanetInertial<P>>(),
            FrameUid::of::<astrodyn_quantities::frame::PlanetFixed<P>>(),
        )));
        self.sources.push((name.into(), entry));
        self.source_ephem_bodies.push(None);
        idx
    }

    /// Configure ephemeris-based position updates for a source.
    ///
    /// # Panics
    /// Panics if `idx` is not a valid source index (i.e., greater than or
    /// equal to the number of sources added so far). Mirrors
    /// `Simulation::set_source_ephemeris` so callers see the same
    /// diagnostic regardless of which surface they hit first.
    pub fn set_source_ephemeris(
        &mut self,
        idx: usize,
        target: EphemerisBody,
        observer: EphemerisBody,
    ) -> &mut Self {
        assert!(
            idx < self.source_ephem_bodies.len(),
            "set_source_ephemeris: source index {idx} out of range \
             ({} sources added)",
            self.source_ephem_bodies.len()
        );
        self.source_ephem_bodies[idx] = Some((target, observer));
        self
    }

    /// Add a third-body perturbation source whose inertial position is
    /// updated each step from the simulation's ephemeris.
    ///
    /// Convenience over the two-step
    /// [`add_source`](Self::add_source) +
    /// [`set_source_ephemeris`](Self::set_source_ephemeris) flow:
    /// constructs a point-mass third-body
    /// [`GravitySourceEntry`] at the
    /// origin and immediately wires its `(target, observer)` ephemeris
    /// pair.
    ///
    /// `target` is the perturber (Earth, Sun, …); `observer` is the
    /// integration-frame origin body (typically the central planet —
    /// e.g., Moon for a lunar-orbit scenario, Earth for a LEO scenario).
    ///
    /// The seed position is zero — the per-step ephemeris stage
    /// rewrites it before any gravity evaluation, so the literal value
    /// is immaterial. Mission code that wants a non-ephemeris-driven
    /// third body should call [`add_source`](Self::add_source) directly.
    ///
    /// Returns the source's index in [`Self::sources`].
    ///
    /// ```ignore
    /// use astrodyn::recipes::moon;
    /// use astrodyn::{EphemerisBody, EARTH, SimulationBuilder, SimulationTime};
    /// use astrodyn::default_leap_second_table;
    ///
    /// let time = SimulationTime::at_j2000(default_leap_second_table());
    /// let mut sb = SimulationBuilder::new(time, 1.0);
    /// let _moon_idx = sb.add_source("Moon", moon::grail150_with_libration());
    /// let _earth_idx = sb.add_third_body_with_ephemeris(
    ///     "Earth", &EARTH, EphemerisBody::Earth, EphemerisBody::Moon);
    /// ```
    pub fn add_third_body_with_ephemeris(
        &mut self,
        name: impl Into<String>,
        planet: &crate::PlanetConfig,
        target: EphemerisBody,
        observer: EphemerisBody,
    ) -> usize {
        let entry = crate::sources::GravitySourceEntry::third_body(
            planet,
            astrodyn_quantities::aliases::Position::<astrodyn_quantities::frame::RootInertial>::zero(),
        );
        let idx = self.add_source(name, entry);
        self.set_source_ephemeris(idx, target, observer);
        idx
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
    /// - `body_idx` is out of range for the bodies added so far.
    /// - The body does not define mass properties.
    pub fn register_in_mass_tree(&mut self, body_idx: usize, name: impl Into<String>) -> &mut Self {
        assert!(
            body_idx < self.bodies.len(),
            "register_in_mass_tree: body index {body_idx} out of range \
             ({} bodies added)",
            self.bodies.len()
        );
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
    /// - `child_idx` or `parent_idx` is out of range for the bodies
    ///   added so far.
    /// - Either body has not been registered in the mass tree.
    pub fn attach_bodies(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) -> &mut Self {
        assert!(
            child_idx < self.bodies.len(),
            "attach_bodies: child body index {child_idx} out of range \
             ({} bodies added)",
            self.bodies.len()
        );
        assert!(
            parent_idx < self.bodies.len(),
            "attach_bodies: parent body index {parent_idx} out of range \
             ({} bodies added)",
            self.bodies.len()
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GravityModel, GravitySource};

    mod tags {
        crate::define_planet!(BuilderTestPlanet);
    }

    fn entry() -> GravitySourceEntry {
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn_quantities::aliases::Position::zero(),
            velocity: astrodyn_quantities::aliases::Velocity::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: crate::RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        }
    }

    /// `sources`, `source_uids`, and `source_ephem_bodies` are parallel
    /// vectors consumed positionally by both the runner's `from_builder`
    /// and the Bevy adapter's `populate_app` — every registration path
    /// must push to all three. A missed push desyncs the indices and
    /// fails the consumer's length assert (or worse, mis-wires ephemeris
    /// updates to the wrong source).
    #[test]
    fn source_registration_keeps_parallel_vecs_in_sync() {
        let time = crate::SimulationTime::at_j2000(crate::default_leap_second_table());
        let mut sb = SimulationBuilder::new(time, 1.0);
        sb.add_source("Earth", entry());
        sb.add_source_typed::<tags::BuilderTestPlanet>("BuilderTestPlanet", entry());
        assert_eq!(sb.sources.len(), 2);
        assert_eq!(sb.source_uids.len(), 2);
        assert_eq!(sb.source_ephem_bodies.len(), 2);
        assert!(
            sb.source_uids[0].is_none(),
            "name-dispatch path carries no uid"
        );
        assert!(
            sb.source_uids[1].is_some(),
            "typed path carries minted uids"
        );
    }
}
