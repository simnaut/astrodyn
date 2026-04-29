#![forbid(unsafe_code)]

pub mod bundles;
pub mod components;
pub mod prelude;
pub mod recipes;
pub mod sets;
pub mod systems;
pub mod validation;

pub use bundles::*;
pub use components::*;
pub use sets::*;
pub use systems::*;

use bevy::prelude::*;

// Re-export jeod_sim types that form the public atmosphere API.
pub use jeod_sim::atmosphere::{AtmosphereConfig, AtmosphereModel};

/// Bevy resource wrapping `SimulationTime`.
// JEOD_INV: TM.07 — JEOD uses -1.0 sentinel; we call recompute_derived() at construction instead
#[derive(Resource, Debug, Deref, DerefMut)]
pub struct SimulationTimeR(pub jeod_sim::SimulationTime);

impl Default for SimulationTimeR {
    fn default() -> Self {
        Self(jeod_sim::SimulationTime::at_j2000(
            jeod_sim::default_leap_second_table(),
        ))
    }
}

/// Optional Bevy resource for polar motion (xp, yp) in radians.
///
/// When inserted, the `planet_fixed_rotation_system` includes polar motion
/// in the RNP composition: W(xp,yp) × R(GAST) × N × P.
/// When absent, polar motion is omitted (equivalent to `enable_polar=false`).
///
/// For time-varying polar motion, update this resource each step from
/// IERS EOP data.
#[derive(Resource, Debug, Clone, Copy)]
pub struct PolarMotionR {
    /// Polar motion x_p in radians.
    pub xp: f64,
    /// Polar motion y_p in radians.
    pub yp: f64,
}

/// Bevy resource wrapping [`AtmosphereConfig`] with an entity reference for
/// the planet whose rotation matrix is used for geodetic conversion.
#[derive(Resource, Debug, Clone)]
pub struct AtmosphereModelR {
    /// ECS-agnostic atmosphere configuration (model, radii, wind).
    pub config: AtmosphereConfig,
    /// Entity of the planet whose `PlanetFixedRotationC` is used.
    /// `None` means no rotation (position assumed planet-fixed).
    pub planet_entity: Option<Entity>,
}

/// Bevy resource wrapping [`jeod_sim::Ephemeris`] for DE4xx ephemeris access.
///
/// When inserted, `planet_fixed_rotation_system` can use `MoonDE421` rotation
/// and `ephemeris_update_system` can update source positions from DE421/DE440.
#[derive(Resource, Deref, DerefMut)]
pub struct EphemerisR(pub jeod_sim::Ephemeris);

/// Bevy resource wrapping `MassTree` for multi-body vehicles.
///
/// Shared by all entities that have [`components::MassBodyIdC`].
/// The `staging_system` processes [`components::AttachEvent`] and
/// [`components::DetachEvent`] to modify the tree and sync
/// composite mass properties back to affected entities.
///
/// This resource is not inserted automatically by [`JeodPlugin`]. Applications
/// that use staging must insert `MassTreeR` before sending
/// [`components::AttachEvent`] or
/// [`components::DetachEvent`]. If the resource is absent, staging
/// events are silently drained.
#[derive(Resource, Deref, DerefMut)]
pub struct MassTreeR(pub jeod_sim::MassTree);

/// Unified JEOD plugin — registers all pipeline systems and schedule sets.
pub struct JeodPlugin;

impl Plugin for JeodPlugin {
    fn build(&self, app: &mut App) {
        // ── Schedule set ordering ──
        // JEOD_INV: DM.04 — init order: time -> ephemeris -> environment -> interaction -> forces -> integration -> derived
        // JEOD_INV: DM.13 — ephemeris updated before gravity (EphemerisUpdate before Environment)
        app.configure_sets(
            FixedUpdate,
            (
                JeodSet::TimeUpdate,
                JeodSet::EphemerisUpdate.after(JeodSet::TimeUpdate),
                JeodSet::Environment.after(JeodSet::EphemerisUpdate),
                JeodSet::Interaction.after(JeodSet::Environment),
                JeodSet::ForceCollection.after(JeodSet::Interaction),
                JeodSet::Integration.after(JeodSet::ForceCollection),
                JeodSet::DerivedState.after(JeodSet::Integration),
            ),
        );

        // ── Resources ──
        app.init_resource::<SimulationTimeR>();

        // ── Typed-Component reflection (#154) ──
        // Centralized in `register_jeod_component_types` so the smoke
        // test and any other consumer registers exactly the same set.
        register_jeod_component_types(app);

        // ── Events ──
        app.add_message::<AttachEvent>();
        app.add_message::<DetachEvent>();

        // ── Systems ──
        // Split into two add_systems calls to stay within Bevy's tuple size limit.
        app.add_systems(
            FixedUpdate,
            (
                // Validation runs first — matches JEOD's initialize_simulation()
                validation::validate_jeod_invariants.before(JeodSet::TimeUpdate),
                // Time advance
                systems::time_advance_system.in_set(JeodSet::TimeUpdate),
                // Planet-fixed rotation (RNP)
                systems::planet_fixed_rotation_system.in_set(JeodSet::EphemerisUpdate),
                // Ephemeris position updates (DE4xx)
                systems::ephemeris_update_system.in_set(JeodSet::EphemerisUpdate),
                // Tidal ΔC20 (must run after planet-fixed rotation)
                systems::tidal_update_system
                    .in_set(JeodSet::EphemerisUpdate)
                    .after(systems::planet_fixed_rotation_system),
                // Mass update: recompute inverse_mass/inverse_inertia each step.
                systems::mass_update_system
                    .after(JeodSet::TimeUpdate)
                    .before(JeodSet::EphemerisUpdate),
                // Gravity pre-computation
                systems::gravity_computation_system.in_set(JeodSet::Environment),
                // Atmosphere evaluation
                systems::atmosphere_update_system.in_set(JeodSet::Environment),
                // Interactions
                // Mass tree staging (attach/detach) — runs before interactions
                // so mass changes affect the current step's forces and integration.
                systems::staging_system
                    .after(JeodSet::Environment)
                    .before(JeodSet::Interaction),
                systems::aero_drag_system.in_set(JeodSet::Interaction),
                systems::gravity_torque_system.in_set(JeodSet::Interaction),
                systems::flat_plate_srp_system.in_set(JeodSet::Interaction),
                systems::cannonball_srp_system.in_set(JeodSet::Interaction),
            ),
        );
        app.add_systems(
            FixedUpdate,
            (
                // Force collection and integration
                systems::force_collection_system.in_set(JeodSet::ForceCollection),
                systems::integration_system.in_set(JeodSet::Integration),
                // Derived states
                systems::orbital_elements_system.in_set(JeodSet::DerivedState),
                systems::euler_angles_system.in_set(JeodSet::DerivedState),
                systems::lvlh_system.in_set(JeodSet::DerivedState),
                systems::geodetic_system.in_set(JeodSet::DerivedState),
                systems::solar_beta_system.in_set(JeodSet::DerivedState),
                systems::earth_lighting_system.in_set(JeodSet::DerivedState),
            ),
        );
    }
}

/// Register every `Reflect`-derived Component from
/// [`crate::components`] in the `App`'s `TypeRegistry`.
///
/// `JeodPlugin::build` calls this; downstream consumers that don't use
/// `JeodPlugin` (e.g. test harnesses, custom adapters that compose only
/// a subset of systems) can call it directly to populate the same
/// registry. Tests use this through the same entry point so the list
/// can't drift between production and verification.
///
/// Inner `jeod_*` types are `#[reflect(opaque)]` so the Component
/// appears as a leaf with its type name. Field-level introspection of
/// `Position<Inertial>`, `RotationalState`, etc. would require
/// propagating `Reflect` into the source crates and is out of scope
/// here.
pub fn register_jeod_component_types(app: &mut App) {
    // Dynamics state
    app.register_type::<components::TranslationalStateC>();
    app.register_type::<components::RotationalStateC>();
    app.register_type::<components::MassPropertiesC>();
    app.register_type::<components::GravityAccelerationC>();
    app.register_type::<components::TotalForceC>();
    app.register_type::<components::FrameDerivativesC>();
    // Dynamics config + integrator state
    app.register_type::<components::DynamicsConfigC>();
    app.register_type::<components::IntegratorTypeC>();
    app.register_type::<components::GaussJacksonStateC>();
    app.register_type::<components::Abm4StateC>();
    // Gravity
    app.register_type::<components::GravityControlsC>();
    app.register_type::<components::GravitySourceC>();
    app.register_type::<components::SourceInertialPositionC>();
    app.register_type::<components::SourceInertialVelocityC>();
    // Interactions
    app.register_type::<components::AerodynamicForceC>();
    app.register_type::<components::RadiationForceC>();
    app.register_type::<components::GravityTorqueC>();
    app.register_type::<components::AtmosphericStateC>();
    // Frame transforms
    app.register_type::<components::StructuralTransformC>();
    app.register_type::<components::PlanetFixedRotationC>();
    // Tidal
    app.register_type::<components::TidalConfigC>();
    app.register_type::<components::TidalDeltaC20C>();
    // Drag / SRP
    app.register_type::<components::DragConfigC>();
    app.register_type::<components::FlatPlateConfigC>();
    app.register_type::<components::CannonballSrpC>();
    app.register_type::<components::ShadowBodyC>();
    // External loads
    app.register_type::<components::ExternalForceC>();
    app.register_type::<components::ExternalTorqueC>();
    // Body / planet identity + ephemeris
    app.register_type::<components::MassBodyIdC>();
    app.register_type::<components::PlanetC>();
    app.register_type::<components::RotationModelC>();
    app.register_type::<components::EphemerisBodyC>();
    app.register_type::<components::SunMarker>();
    app.register_type::<components::MoonMarker>();
    // Derived-state config
    app.register_type::<components::OrbitalElementsConfigC>();
    app.register_type::<components::EulerAnglesConfigC>();
    app.register_type::<components::GeodeticConfigC>();
    app.register_type::<components::EarthLightingConfigC>();
    // Derived-state output
    app.register_type::<components::OrbitalElementsC>();
    app.register_type::<components::EulerAnglesC>();
    app.register_type::<components::LvlhFrameC>();
    app.register_type::<components::GeodeticStateC>();
    app.register_type::<components::SolarBetaC>();
    app.register_type::<components::EarthLightingStateC>();
}

// ── Bevy spawn helpers for the typestate VehicleBuilder ──

/// Bevy-side terminal for [`jeod_sim::VehicleBuilder`].
///
/// `VehicleBuilder<Ready>::build()` returns a [`jeod_sim::VehicleConfig`]
/// that the standalone `jeod_runner::Simulation` consumes via
/// `SimulationBuilder::add_body`. This trait provides the parallel
/// terminal for Bevy: given a runtime mapping from gravity-source indices
/// (the `usize`-indexed [`GravityControl`](jeod_sim::GravityControl)s in
/// the built config) to ECS [`Entity`]s, it spawns the vehicle entity
/// with all the required JEOD components attached.
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::{PlanetBundle, VehicleConfigBevyExt};
/// use jeod_sim::recipes::{constants, orbital_elements, vehicle};
/// use jeod_sim::{GravityControl, VehicleBuilder, EARTH};
///
/// let mut app = App::new();
/// app.add_systems(Startup, |mut commands: Commands| {
///     let earth = commands.spawn(PlanetBundle::point_mass("Earth", &EARTH)).id();
///     let cfg = VehicleBuilder::new()
///         .from_orbital_elements(orbital_elements::iss(), constants::mu_ggm05c())
///         .three_dof_point_mass(vehicle::iss_mass())
///         .rk4()
///         .gravity(GravityControl::new_spherical(0_usize, false))
///         .build();
///     cfg.spawn_bevy(&mut commands, &[earth]);
/// });
/// app.update();
/// ```
pub trait VehicleConfigBevyExt {
    /// Spawn a Bevy entity carrying the core components implied by this
    /// vehicle configuration.
    ///
    /// Currently inserts: translational state, optional rotational state,
    /// optional mass properties, dynamics config, gravity controls,
    /// integrator type, structural transform, optional external force /
    /// torque, and (when `compute_gravity_gradient`) a default gravity
    /// torque component. `source_entities` resolves each `usize` index in
    /// `gravity_controls` to the corresponding ECS [`Entity`].
    ///
    /// Not yet wired (callers must insert these manually): drag, SRP
    /// (flat-plate / cannonball), shadow body, derived-state requests
    /// (orbital elements, Euler, LVLH, geodetic, solar beta, earth
    /// lighting), `integ_source`, and frame switches. These are tracked
    /// for future expansion of `spawn_bevy`.
    ///
    /// Panics if any `GravityControl::source_name` index is out of bounds
    /// in `source_entities`.
    ///
    /// Returns the spawned vehicle entity ID.
    fn spawn_bevy(self, commands: &mut Commands, source_entities: &[Entity]) -> Entity;
}

/// Resolve a `usize` source index against the caller-supplied entity
/// table, panicking with a descriptive error when the index is out of
/// bounds. Centralizes the error message so every site in
/// [`VehicleConfigBevyExt::spawn_bevy`] that translates a source index
/// produces the same actionable diagnostic.
fn resolve_source_entity(source_entities: &[Entity], idx: usize, what: &str) -> Entity {
    *source_entities.get(idx).unwrap_or_else(|| {
        panic!(
            "spawn_bevy: {what} references source index {idx} but only {len} source \
             entities were provided. Spawn all gravity sources before calling spawn_bevy.",
            what = what,
            idx = idx,
            len = source_entities.len()
        )
    })
}

impl VehicleConfigBevyExt for jeod_sim::VehicleConfig {
    fn spawn_bevy(self, commands: &mut Commands, source_entities: &[Entity]) -> Entity {
        // Translate `GravityControls<usize>` to `GravityControls<Entity>` by
        // retagging the source identifier on each control via the
        // `GravityControl::retag_source` helper. The field list lives in
        // exactly one place (`jeod_gravity::gravity_controls`), so adding a
        // new field there does not require touching this site.
        let entity_controls = jeod_sim::GravityControls::<Entity> {
            controls: self
                .gravity_controls
                .controls
                .into_iter()
                .map(|c| {
                    c.retag_source(|idx| {
                        resolve_source_entity(source_entities, idx, "GravityControl")
                    })
                })
                .collect(),
        };

        let dynamics_config = jeod_sim::DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: self.rot.is_some(),
            three_dof: self.rot.is_none(),
        };

        let mut entity = commands.spawn((
            components::TranslationalStateC::from(self.trans),
            components::DynamicsConfigC(dynamics_config),
            components::GravityControlsC(entity_controls),
            components::IntegratorTypeC(self.integrator),
            components::StructuralTransformC(jeod_sim::FrameTransform::from_matrix(
                self.t_struct_body,
            )),
        ));
        if let Some(rot) = self.rot {
            entity.insert(components::RotationalStateC::from(rot));
        }
        if let Some(mass) = self.mass {
            entity.insert(components::MassPropertiesC::from(mass));
        }
        if self.external_force != glam::DVec3::ZERO {
            // `VehicleConfig.external_force` is still an untyped
            // `DVec3` field on the `jeod_sim` runtime fluent builder
            // API. The Bevy `ExternalForceC` is typed (`Force<Inertial>`),
            // so this is a one-time insertion-time lift — not a per-step
            // bypass. Migrating `VehicleConfig` itself to typed external
            // fields is a deeper refactor inside `jeod_sim`; out of
            // scope for the Bevy-adapter boundary that #172 H1 targets.
            let f = jeod_sim::Force::<jeod_sim::Inertial>::from_raw_si(self.external_force); // allowed: #172 H1 insertion-time boundary (VehicleConfig still untyped)
            entity.insert(components::ExternalForceC(f));
        }
        if self.external_torque != glam::DVec3::ZERO {
            let t = jeod_sim::Torque::<jeod_sim::BodyFrame<jeod_sim::SelfRef>>::from_raw_si(
                self.external_torque,
            ); // allowed: #172 H1 insertion-time boundary (VehicleConfig still untyped)
            entity.insert(components::ExternalTorqueC(t));
        }
        if self.compute_gravity_gradient {
            entity.insert(components::GravityTorqueC::default());
        }
        entity.id()
    }
}
