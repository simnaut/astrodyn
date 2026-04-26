pub mod bundles;
pub mod components;
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
/// Shared by all entities that have [`MassBodyIdC`](components::MassBodyIdC).
/// The `staging_system` processes [`AttachEvent`](components::AttachEvent) and
/// [`DetachEvent`](components::DetachEvent) to modify the tree and sync
/// composite mass properties back to affected entities.
///
/// This resource is not inserted automatically by [`JeodPlugin`]. Applications
/// that use staging must insert `MassTreeR` before sending
/// [`AttachEvent`](components::AttachEvent) or
/// [`DetachEvent`](components::DetachEvent). If the resource is absent, staging
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
/// ```ignore
/// use bevy_jeod::{JeodPlugin, VehicleBuilderBevyExt};
/// use jeod_sim::recipes::{earth, orbital_elements};
/// use jeod_sim::{GravityControl, VehicleBuilder};
/// use jeod_quantities::ext::F64Ext;
///
/// fn setup(mut commands: Commands) {
///     let earth = commands.spawn(/* gravity-source bundle */).id();
///     let cfg = VehicleBuilder::new()
///         .from_orbital_elements(orbital_elements::iss(), earth::point_mass().mu_typed())
///         .three_dof_point_mass(420_000.0.kg())
///         .rk4()
///         .gravity(GravityControl::new_spherical(0_usize, false))
///         .build();
///     // Resolve source-index 0 to the earth entity.
///     cfg.spawn_bevy(&mut commands, &[earth]);
/// }
/// ```
pub trait VehicleConfigBevyExt {
    /// Spawn a Bevy entity carrying every component implied by this
    /// vehicle configuration.
    ///
    /// `source_entities` resolves each `usize` index in the config's
    /// `gravity_controls` (and shadow / orbital-elements / geodetic /
    /// integ-source references) to the corresponding ECS [`Entity`].
    ///
    /// Returns the spawned vehicle entity ID.
    fn spawn_bevy(self, commands: &mut Commands, source_entities: &[Entity]) -> Entity;
}

impl VehicleConfigBevyExt for jeod_sim::VehicleConfig {
    fn spawn_bevy(self, commands: &mut Commands, source_entities: &[Entity]) -> Entity {
        // Translate `GravityControls<usize>` to `GravityControls<Entity>`
        // by retagging the source identifier on each control. The
        // `From<&GravityControl<A>> for GravityControl<B>` analog
        // doesn't exist, so we build by field-by-field copy.
        let entity_controls = jeod_sim::GravityControls::<Entity> {
            controls: self
                .gravity_controls
                .controls
                .into_iter()
                .map(|c| {
                    let mut nc = jeod_sim::GravityControl::new_spherical(
                        source_entities[c.source_name],
                        c.gradient,
                    );
                    nc.spherical = c.spherical;
                    nc.degree = c.degree;
                    nc.order = c.order;
                    nc.perturbing_only = c.perturbing_only;
                    nc.gradient_degree = c.gradient_degree;
                    nc.gradient_order = c.gradient_order;
                    nc.relativistic = c.relativistic;
                    nc
                })
                .collect(),
        };

        let dynamics_config = jeod_sim::DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: self.rot.is_some(),
            three_dof: self.rot.is_none(),
        };

        let mut entity = commands.spawn((
            components::TranslationalStateC(self.trans),
            components::DynamicsConfigC(dynamics_config),
            components::GravityControlsC(entity_controls),
            components::IntegratorTypeC(self.integrator),
            components::StructuralTransformC(self.t_struct_body),
        ));
        if let Some(rot) = self.rot {
            entity.insert(components::RotationalStateC(rot));
        }
        if let Some(mass) = self.mass {
            entity.insert(components::MassPropertiesC(mass));
        }
        if self.external_force != glam::DVec3::ZERO {
            entity.insert(components::ExternalForceC(jeod_sim::Force::<
                jeod_sim::Inertial,
            >::from_raw_si(
                self.external_force
            )));
        }
        if self.external_torque != glam::DVec3::ZERO {
            entity.insert(components::ExternalTorqueC(jeod_sim::Torque::<
                jeod_sim::BodyFrame<jeod_sim::SelfRef>,
            >::from_raw_si(
                self.external_torque
            )));
        }
        if self.compute_gravity_gradient {
            entity.insert(components::GravityTorqueC::default());
        }
        entity.id()
    }
}
