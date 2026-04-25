//! Builder patterns for ergonomic simulation configuration.
//!
//! [`VehicleBuilder`] provides a fluent API for constructing
//! [`VehicleConfig`], grouping related concerns (state, gravity,
//! interactions, derived states) and auto-deriving `DynamicsConfig`
//! from what's configured.
//!
//! Phase 6 of #101 relocated `SimulationBuilder` and the per-vehicle
//! configuration types ([`VehicleConfig`], [`SrpModel`], …) into
//! `jeod_sim` so they can be shared between the standalone runner and
//! the future Bevy adapter. This module retains the runtime fluent
//! [`VehicleBuilder`] and adds the runner-specific terminal methods
//! (`SimulationBuilder::build` / `build_unchecked`) via the
//! [`SimulationBuilderExt`] trait.

use glam::{DMat3, DVec3};

use jeod_sim::simulation_builder::SimulationBuilder;
use jeod_sim::{
    DerivedStateConfig, DragConfig, EarthLightingConfig, EulerSequence, FrameSwitchConfig,
    GeodeticConfig, GravityControl, GravityControls, MassProperties, PlanetConfig, RotationalState,
    ShadowBody, SimulationTime, SrpModel, TranslationalState, VehicleConfig,
};

use crate::Simulation;

// ══════════════════════════════════════════════════════════════════════════════
// VehicleBuilder
// ══════════════════════════════════════════════════════════════════════════════

/// Fluent builder for [`VehicleConfig`].
///
/// # Example
/// ```ignore
/// use jeod_runner::VehicleConfig;
/// use jeod_sim::{EARTH, GravityControl};
///
/// let vehicle = VehicleBuilder::new(initial_trans)
///     .sixdof(rotation, mass)
///     .gravity(GravityControl::new_spherical(earth_idx, false))
///     .drag(DragConfig { cd: 2.2, area: 1000.0, constant_density: None })
///     .orbital_elements(earth_idx)
///     .lvlh()
///     .build();
/// ```
pub struct VehicleBuilder {
    trans: TranslationalState,
    rot: Option<RotationalState>,
    mass: Option<MassProperties>,
    integrator: jeod_dynamics::IntegratorType,
    t_struct_body: DMat3,
    gravity_controls: GravityControls<usize>,
    compute_gravity_gradient: bool,
    drag: Option<DragConfig>,
    srp: Option<SrpModel>,
    shadow_body: Option<ShadowBody>,
    derived: DerivedStateConfig,
    external_force: DVec3,
    external_torque: DVec3,
    integ_source: Option<usize>,
    frame_switches: Vec<FrameSwitchConfig>,
}

impl VehicleBuilder {
    /// Start building a vehicle at the given initial translational state.
    ///
    /// (Phase 6 of #101: replaces the previous
    /// `VehicleConfig::builder(trans)` constructor — the orphan rule
    /// prevents an inherent `impl VehicleConfig` here now that
    /// `VehicleConfig` lives in `jeod_sim`.)
    pub fn new(trans: TranslationalState) -> VehicleBuilder {
        VehicleBuilder {
            trans,
            rot: None,
            mass: None,
            integrator: jeod_dynamics::IntegratorType::default(),
            t_struct_body: DMat3::IDENTITY,
            gravity_controls: GravityControls::default(),
            compute_gravity_gradient: false,
            drag: None,
            srp: None,
            shadow_body: None,
            derived: DerivedStateConfig::default(),
            external_force: DVec3::ZERO,
            external_torque: DVec3::ZERO,
            integ_source: None,
            frame_switches: Vec::new(),
        }
    }

    // ── Core state ──

    /// Set rotational state (enables 6-DOF dynamics).
    pub fn rotation(mut self, rot: RotationalState) -> Self {
        self.rot = Some(rot);
        self
    }

    /// Set mass properties.
    pub fn mass(mut self, mass: MassProperties) -> Self {
        self.mass = Some(mass);
        self
    }

    /// Set rotation and mass together (6-DOF shorthand).
    pub fn sixdof(mut self, rot: RotationalState, mass: MassProperties) -> Self {
        self.rot = Some(rot);
        self.mass = Some(mass);
        self
    }

    /// Set the integration method (default: RK4).
    pub fn integrator(mut self, integrator: jeod_dynamics::IntegratorType) -> Self {
        self.integrator = integrator;
        self
    }

    /// Set the structural-to-body frame rotation (default: identity).
    pub fn structural_transform(mut self, t: DMat3) -> Self {
        self.t_struct_body = t;
        self
    }

    // ── Gravity ──

    /// Add a gravity control. Call multiple times for multi-source gravity.
    pub fn gravity(mut self, control: GravityControl<usize>) -> Self {
        self.gravity_controls.controls.push(control);
        self
    }

    /// Enable gravity gradient computation (needed for gravity torque).
    pub fn gravity_gradient(mut self) -> Self {
        self.compute_gravity_gradient = true;
        self
    }

    // ── Interactions ──

    /// Enable aerodynamic drag. Atmosphere state is auto-enabled.
    pub fn drag(mut self, config: DragConfig) -> Self {
        self.drag = Some(config);
        self
    }

    /// Enable flat-plate solar radiation pressure.
    pub fn flat_plate_srp(mut self, state: jeod_sim::FlatPlateState) -> Self {
        self.srp = Some(SrpModel::FlatPlate(state));
        self
    }

    /// Enable cannonball solar radiation pressure.
    pub fn cannonball_srp(mut self, cx_area: f64, albedo: f64, diffuse: f64) -> Self {
        self.srp = Some(SrpModel::Cannonball {
            cx_area,
            albedo,
            diffuse,
        });
        self
    }

    /// Set shadow-casting body for SRP eclipse computation.
    /// Uses [`PlanetConfig::shadow_radius`] for consistent radius.
    pub fn shadow(mut self, source_idx: usize, planet: &PlanetConfig) -> Self {
        self.shadow_body = Some(ShadowBody {
            source_idx,
            radius: planet.shadow_radius,
        });
        self
    }

    /// Set shadow-casting body with explicit radius.
    pub fn shadow_with_radius(mut self, source_idx: usize, radius: f64) -> Self {
        self.shadow_body = Some(ShadowBody { source_idx, radius });
        self
    }

    // ── Derived states ──

    /// Compute orbital elements relative to the given gravity source.
    pub fn orbital_elements(mut self, source_idx: usize) -> Self {
        self.derived.orbital_elements_source = Some(source_idx);
        self
    }

    /// Compute Euler angles with the given decomposition sequence.
    pub fn euler_angles(mut self, sequence: EulerSequence) -> Self {
        self.derived.euler_sequence = Some(sequence);
        self
    }

    /// Compute LVLH frame each step.
    pub fn lvlh(mut self) -> Self {
        self.derived.lvlh = true;
        self
    }

    /// Compute geodetic state. Uses [`PlanetConfig`] for consistent radii.
    pub fn geodetic(mut self, source_idx: usize, planet: &PlanetConfig) -> Self {
        self.derived.geodetic = Some(GeodeticConfig {
            source_idx,
            r_eq: planet.shape.r_eq,
            r_pol: planet.shape.r_pol,
        });
        self
    }

    /// Compute solar beta angle. Requires `sun_source` on the Simulation.
    pub fn solar_beta(mut self) -> Self {
        self.derived.solar_beta = true;
        self
    }

    /// Compute earth lighting. Uses [`PlanetConfig`] presets for consistent radii.
    /// Requires `sun_source` and `moon_source` on the Simulation.
    pub fn earth_lighting(
        mut self,
        earth: &PlanetConfig,
        moon: &PlanetConfig,
        sun: &PlanetConfig,
    ) -> Self {
        self.derived.earth_lighting = Some(EarthLightingConfig {
            earth_radius: earth.shape.r_eq,
            moon_radius: moon.shape.r_eq,
            sun_radius: sun.shape.r_eq,
        });
        self
    }

    // ── External loads ──

    /// Set initial external force (inertial frame, N).
    pub fn external_force(mut self, f: DVec3) -> Self {
        self.external_force = f;
        self
    }

    /// Set initial external torque (body frame, N·m).
    pub fn external_torque(mut self, t: DVec3) -> Self {
        self.external_torque = t;
        self
    }

    // ── Frame switching ──

    /// Set the initial integration source (default: simulation root / central body).
    /// `source_idx` is the index returned by `SimulationBuilder::add_source()`.
    pub fn integ_source(mut self, source_idx: usize) -> Self {
        self.integ_source = Some(source_idx);
        self
    }

    /// Set distance-based frame switch triggers.
    pub fn frame_switches(mut self, switches: Vec<FrameSwitchConfig>) -> Self {
        self.frame_switches = switches;
        self
    }

    // ── Build ──

    /// Build the [`VehicleConfig`].
    ///
    /// `DynamicsConfig` is auto-derived: `rotational_dynamics` is enabled if
    /// rotation was set via `.rotation()` or `.sixdof()`.
    pub fn build(self) -> VehicleConfig {
        VehicleConfig {
            trans: self.trans,
            rot: self.rot,
            mass: self.mass,
            integrator: self.integrator,
            t_struct_body: self.t_struct_body,
            gravity_controls: self.gravity_controls,
            compute_gravity_gradient: self.compute_gravity_gradient,
            drag: self.drag,
            srp: self.srp,
            shadow_body: self.shadow_body,
            derived: self.derived,
            external_force: self.external_force,
            external_torque: self.external_torque,
            integ_source: self.integ_source,
            frame_switches: self.frame_switches,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Simulation construction from a relocated `jeod_sim::SimulationBuilder`
// ══════════════════════════════════════════════════════════════════════════════
//
// `SimulationBuilder` itself lives in `jeod_sim` (Phase 6 of #101 relocated it
// so the same scenario library serves the standalone runner and the future
// Bevy adapter without duplication). Materializing a builder into a runtime
// `Simulation` is runner-specific and lives here, both as an inherent
// constructor (`Simulation::from_builder`) and as an extension trait
// (`SimulationBuilderExt::build`) so call sites read identically to the
// pre-relocation API.

impl Simulation {
    /// Start building a simulation with the given time and timestep.
    ///
    /// Convenience wrapper around `SimulationBuilder::new` that mirrors the
    /// pre-relocation `Simulation::builder(time, dt)` API.
    pub fn builder(time: SimulationTime, dt: f64) -> SimulationBuilder {
        SimulationBuilder::new(time, dt)
    }

    /// Materialize a [`SimulationBuilder`] into a runtime [`Simulation`]
    /// without running validation. Use for tests that intentionally
    /// exercise invalid configurations.
    pub fn from_builder_unchecked(builder: SimulationBuilder) -> Self {
        let SimulationBuilder {
            time,
            dt,
            atmosphere,
            atmosphere_planet_source,
            ephemeris,
            polar_motion,
            sun_source,
            moon_source,
            sources,
            source_ephem_bodies,
            bodies,
            mass_tree_names,
            mass_tree_attachments,
        } = builder;

        let mut sim = Simulation::new(time, dt);
        sim.atmosphere = atmosphere;
        sim.atmosphere_planet_source = atmosphere_planet_source;
        sim.ephemeris = ephemeris;
        sim.polar_motion = polar_motion;
        sim.sun_source = sun_source;
        sim.moon_source = moon_source;

        for (i, (name, source)) in sources.into_iter().enumerate() {
            sim.add_source(name, source);
            if let Some(Some((target, observer))) = source_ephem_bodies.get(i) {
                sim.set_source_ephemeris(i, *target, *observer);
            }
        }

        for body in bodies {
            sim.add_body(body);
        }

        // Wire up mass tree if any bodies were registered.
        let has_tree = mass_tree_names.iter().any(|n| n.is_some());
        if has_tree {
            for (idx, name) in mass_tree_names.into_iter().enumerate() {
                if let Some(name) = name {
                    sim.add_body_to_tree(idx, name);
                }
            }
            for att in mass_tree_attachments {
                sim.attach(
                    att.child_idx,
                    att.parent_idx,
                    att.offset,
                    att.t_parent_child,
                );
            }
        }

        sim
    }

    /// Materialize a [`SimulationBuilder`] into a runtime [`Simulation`]
    /// and run validation. Returns `Err` on validation failure.
    pub fn from_builder(
        builder: SimulationBuilder,
    ) -> Result<Self, Vec<jeod_sim::ValidationError>> {
        let mut sim = Self::from_builder_unchecked(builder);
        sim.validate()?;
        Ok(sim)
    }
}

/// Extension trait providing terminal `.build()` / `.build_unchecked()`
/// methods on the relocated [`jeod_sim::SimulationBuilder`].
///
/// Mission code typically imports this via `jeod_runner::prelude::*` so
/// `scenarios::iss_leo().build()?` reads naturally. Callers that prefer
/// the explicit form can use [`Simulation::from_builder`] directly.
pub trait SimulationBuilderExt: Sized {
    /// Build and validate the simulation. Returns `Err` on validation
    /// failure.
    fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>>;

    /// Build without validation. Use only for tests that intentionally
    /// exercise invalid configurations.
    fn build_unchecked(self) -> Simulation;
}

impl SimulationBuilderExt for SimulationBuilder {
    fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>> {
        Simulation::from_builder(self)
    }

    fn build_unchecked(self) -> Simulation {
        Simulation::from_builder_unchecked(self)
    }
}
