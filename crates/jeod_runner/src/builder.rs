//! Builder patterns for ergonomic simulation configuration.
//!
//! [`VehicleBuilder`] provides a fluent API for constructing [`VehicleConfig`],
//! grouping related concerns (state, gravity, interactions, derived states) and
//! auto-deriving `DynamicsConfig` from what's configured.
//!
//! [`SimulationBuilder`] wraps [`Simulation`] construction with auto-validation
//! on `build()`.

use glam::{DMat3, DVec3};

use jeod_sim::{
    DragConfig, EulerSequence, GravityControl, GravityControls, MassProperties, PlanetConfig,
    RotationalState, SimulationTime, TranslationalState,
};

use crate::{
    DerivedStateConfig, EarthLightingConfig, GeodeticConfig, GravitySourceEntry, ShadowBody,
    Simulation, SrpModel, VehicleConfig,
};

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
/// let vehicle = VehicleConfig::builder(initial_trans)
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
}

impl VehicleConfig {
    /// Start building a vehicle at the given initial translational state.
    pub fn builder(trans: TranslationalState) -> VehicleBuilder {
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
        }
    }
}

impl VehicleBuilder {
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
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SimulationBuilder
// ══════════════════════════════════════════════════════════════════════════════

/// Fluent builder for [`Simulation`].
///
/// # Example
/// ```ignore
/// use jeod_runner::Simulation;
/// use jeod_sim::{SimulationTime, EARTH, AtmosphereModel};
///
/// let mut builder = Simulation::builder(time, 10.0);
/// let earth = builder.add_source(GravitySourceEntry::central_body(&EARTH));
/// builder.add_body(VehicleConfig::builder(trans).gravity(ctrl).build());
/// let mut sim = builder.build()?;
/// ```
pub struct SimulationBuilder {
    time: SimulationTime,
    dt: f64,
    atmosphere: Option<jeod_sim::AtmosphereConfig>,
    atmosphere_planet_source: Option<usize>,
    ephemeris: Option<jeod_sim::Ephemeris>,
    polar_motion: Option<(f64, f64)>,
    sun_source: Option<usize>,
    moon_source: Option<usize>,
    sources: Vec<GravitySourceEntry>,
    source_ephem_bodies: Vec<Option<(jeod_sim::EphemerisBody, jeod_sim::EphemerisBody)>>,
    bodies: Vec<VehicleConfig>,
}

impl Simulation {
    /// Start building a simulation with the given time and timestep.
    pub fn builder(time: SimulationTime, dt: f64) -> SimulationBuilder {
        SimulationBuilder {
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
        }
    }
}

impl SimulationBuilder {
    // ── Global config (fluent, consumes self) ──

    /// Set atmosphere configuration with explicit planet source index.
    pub fn atmosphere(mut self, config: jeod_sim::AtmosphereConfig, planet_source: usize) -> Self {
        self.atmosphere = Some(config);
        self.atmosphere_planet_source = Some(planet_source);
        self
    }

    /// Set atmosphere configuration from a [`PlanetConfig`] preset.
    pub fn atmosphere_from_planet(
        mut self,
        model: jeod_sim::AtmosphereModel,
        planet: &PlanetConfig,
        planet_source: usize,
    ) -> Self {
        self.atmosphere = Some(jeod_sim::AtmosphereConfig::from_planet(model, planet));
        self.atmosphere_planet_source = Some(planet_source);
        self
    }

    /// Set ephemeris data (DE421/DE430) for per-step source position updates.
    pub fn ephemeris(mut self, eph: jeod_sim::Ephemeris) -> Self {
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

    /// Add a gravity source. Returns its index.
    pub fn add_source(&mut self, entry: GravitySourceEntry) -> usize {
        let idx = self.sources.len();
        self.sources.push(entry);
        self.source_ephem_bodies.push(None);
        idx
    }

    /// Configure ephemeris-based position updates for a source.
    pub fn set_source_ephemeris(
        &mut self,
        idx: usize,
        target: jeod_sim::EphemerisBody,
        observer: jeod_sim::EphemerisBody,
    ) -> &mut Self {
        self.source_ephem_bodies[idx] = Some((target, observer));
        self
    }

    /// Add a vehicle. Returns its index.
    pub fn add_body(&mut self, config: VehicleConfig) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(config);
        idx
    }

    // ── Build ──

    /// Build and validate the simulation.
    ///
    /// Returns `Err` if validation fails (e.g., missing prerequisites,
    /// invalid source indices, force producers without mass).
    pub fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>> {
        let mut sim = self.build_unchecked();
        sim.validate()?;
        Ok(sim)
    }

    /// Build without validation. Use for tests that intentionally exercise
    /// invalid configurations.
    pub fn build_unchecked(self) -> Simulation {
        let mut sim = Simulation::new(self.time, self.dt);
        sim.atmosphere = self.atmosphere;
        sim.atmosphere_planet_source = self.atmosphere_planet_source;
        sim.ephemeris = self.ephemeris;
        sim.polar_motion = self.polar_motion;
        sim.sun_source = self.sun_source;
        sim.moon_source = self.moon_source;

        for (i, source) in self.sources.into_iter().enumerate() {
            sim.add_source(source);
            if let Some(Some((target, observer))) = self.source_ephem_bodies.get(i) {
                sim.set_source_ephemeris(i, *target, *observer);
            }
        }

        for body in self.bodies {
            sim.add_body(body);
        }

        sim
    }
}
