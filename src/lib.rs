#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod bundles;
pub mod components;
pub mod prelude;
pub mod recipes;
pub mod sets;
pub mod source_mutator;
pub mod systems;
pub mod validation;

pub use bundles::*;
pub use components::*;
pub use sets::*;
pub use source_mutator::SourceMutator;
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

/// Bevy resource wrapping the simulation's [`jeod_sim::FrameTree`].
///
/// Mirrors `jeod_runner::Simulation::frame_tree`. Inserted at startup by
/// [`JeodPlugin`] with a single root inertial frame node; mission code or
/// recipes can register additional frame nodes (source inertials, pfix
/// frames, body frames) during entity spawning.
///
/// Issue #71: this resource is the data structure that the lifted
/// `jeod_sim::{frame_orchestration, source_state}` helpers operate on,
/// so the Bevy adapter can consume the same orchestration code as
/// `jeod_runner` instead of re-implementing it.
#[derive(Resource, Deref, DerefMut)]
pub struct FrameTreeR(pub jeod_sim::FrameTree);

impl FrameTreeR {
    /// Create a new frame tree pre-populated with a permanent
    /// `root.inertial` root frame. Unlike `jeod_runner::Simulation::new`
    /// (which renames the root to `<central>.inertial` when a central
    /// body is registered), the Bevy adapter keeps a generic root and
    /// registers every gravity source as its child — see
    /// `register_source_frames_system` for the divergence rationale.
    /// Returns the resource and the root inertial [`jeod_sim::FrameId`].
    pub fn new() -> (Self, jeod_sim::FrameId) {
        let mut tree = jeod_sim::FrameTree::new();
        let root = tree.add_root("root.inertial".into(), jeod_sim::RefFrameKind::Inertial);
        (Self(tree), root)
    }
}

impl Default for FrameTreeR {
    fn default() -> Self {
        Self::new().0
    }
}

/// Bevy resource carrying the [`jeod_sim::FrameId`] of the root inertial
/// frame inside [`FrameTreeR`]. Used by source-mutation helpers and
/// (forthcoming) frame-switch and non-root integration systems to
/// distinguish the root from non-root sources.
#[derive(Resource, Debug, Clone, Copy, Deref, DerefMut)]
pub struct RootFrameIdR(pub jeod_sim::FrameId);

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
        // Frame tree + root: only seed when the caller hasn't pre-installed
        // them. Mission code that wants to pre-seed extra root-level
        // frames (or a custom root name) inserts both `FrameTreeR` and
        // `RootFrameIdR` *before* adding `JeodPlugin`; the plugin then
        // preserves them. Inserting either alone is rejected — they
        // describe the same tree and must stay consistent.
        match (
            app.world().contains_resource::<FrameTreeR>(),
            app.world().contains_resource::<RootFrameIdR>(),
        ) {
            (false, false) => {
                let (frame_tree, root_id) = FrameTreeR::new();
                app.insert_resource(frame_tree);
                app.insert_resource(RootFrameIdR(root_id));
            }
            (true, true) => {
                // Caller pre-installed both; leave untouched.
            }
            (true, false) => panic!(
                "JeodPlugin: FrameTreeR was pre-installed but RootFrameIdR was not. \
                 Insert both together (e.g. via FrameTreeR::new()) before adding JeodPlugin, \
                 or insert neither and let the plugin create them.",
            ),
            (false, true) => panic!(
                "JeodPlugin: RootFrameIdR was pre-installed but FrameTreeR was not. \
                 Insert both together (e.g. via FrameTreeR::new()) before adding JeodPlugin, \
                 or insert neither and let the plugin create them.",
            ),
        }

        // ── Typed-Component reflection (#154) ──
        // Centralized in `register_jeod_component_types` so the smoke
        // test and any other consumer registers exactly the same set.
        register_jeod_component_types(app);

        // ── Events ──
        app.add_message::<AttachEvent>();
        app.add_message::<DetachEvent>();

        // ── Systems ──
        // Source-frame registration runs at Startup to populate FrameTreeR
        // with every spawned source, and again before each FixedUpdate's
        // EphemerisUpdate to catch late-spawned sources. The latter
        // filters by `Without<SourceFrameIdC>` so already-registered
        // sources are skipped — registering is one-time per source.
        // Body-frame registration follows so bodies can resolve
        // `IntegSourceC(Some(source_entity))` against an already-registered
        // source. Issue #71 items 2, 4, 5.
        //
        // Registration is wired into three schedules so it catches every
        // spawn surface (PR #260 round-3 R3 fixup):
        //   - Startup: initial spawns before any tick.
        //   - PreUpdate: catches entities spawned during the previous
        //     frame's `Update` / `PostUpdate`. They are registered before
        //     the *next* frame's `Update` runs. Same-frame spawn-and-
        //     mutate inside one `Update` (spawn + `SourceMutator` call in
        //     consecutive systems of the same frame) is *not* supported
        //     by this scheduling, since `Update` runs after `PreUpdate`;
        //     callers needing that pattern must add a manual
        //     registration call in `Update` with explicit ordering.
        //   - Before `JeodSet::EphemerisUpdate` (FixedUpdate): catches
        //     entities spawned between fixed ticks before they hit the
        //     ephemeris / rotation / integration pipeline.
        // Each pass is a no-op for already-registered entities (the
        // `Without<SourceFrameIdC>` / `Without<BodyFrameIdC>` filters
        // make repeated runs cost a single query iteration).
        // `register_pfix_frames_system` covers a rare but real case
        // (round-9 fixup): a source spawned without `PlanetFixedRotationC`
        // that gains it after the initial registration. The main
        // `register_source_frames_system` filters by
        // `Without<SourceFrameIdC>` so it can't observe that mutation;
        // the dedicated pfix pass uses `Without<SourcePfixFrameIdC>` +
        // `With<PlanetFixedRotationC>` instead.
        app.add_systems(
            Startup,
            (
                systems::register_source_frames_system,
                systems::register_pfix_frames_system.after(systems::register_source_frames_system),
                systems::register_body_frames_system.after(systems::register_pfix_frames_system),
            ),
        );
        app.add_systems(
            PreUpdate,
            (
                systems::register_source_frames_system,
                systems::register_pfix_frames_system.after(systems::register_source_frames_system),
                systems::register_body_frames_system.after(systems::register_pfix_frames_system),
            ),
        );
        // Split into two add_systems calls to stay within Bevy's tuple size limit.
        app.add_systems(
            FixedUpdate,
            (
                // Validation runs first — matches JEOD's initialize_simulation()
                validation::validate_jeod_invariants.before(JeodSet::TimeUpdate),
                // Time advance
                systems::time_advance_system.in_set(JeodSet::TimeUpdate),
                // Catch dynamically-spawned sources before they hit
                // `planet_fixed_rotation_system` / `ephemeris_update_system`.
                systems::register_source_frames_system.before(JeodSet::EphemerisUpdate),
                // Late-attached `PlanetFixedRotationC` → pfix child node
                // (round-9 fixup; see `register_pfix_frames_system` doc).
                systems::register_pfix_frames_system
                    .after(systems::register_source_frames_system)
                    .before(JeodSet::EphemerisUpdate),
                // Catch dynamically-spawned bodies (after source registration so
                // any IntegSourceC reference resolves to a registered source).
                systems::register_body_frames_system
                    .after(systems::register_pfix_frames_system)
                    .before(JeodSet::EphemerisUpdate),
                // After ephemeris_update_system writes new source position /
                // velocity, mirror the values into FrameTreeR so frame-tree
                // consumers (compute_relative_state, frame_origin) see the
                // latest state. PR #260 review fixup.
                systems::sync_source_to_frame_system
                    .in_set(JeodSet::EphemerisUpdate)
                    .after(systems::ephemeris_update_system)
                    .after(systems::planet_fixed_rotation_system),
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
                // After integration, sync the body's typed state into its
                // FrameTreeR node so frame-switch evaluation sees current
                // distances. Issue #71 item 2.
                systems::sync_body_to_frame_system
                    .in_set(JeodSet::Integration)
                    .after(systems::integration_system),
                // Evaluate distance-based frame switches and reparent the
                // body in the frame tree on trigger. Issue #71 item 3.
                systems::frame_switch_system
                    .in_set(JeodSet::Integration)
                    .after(systems::sync_body_to_frame_system),
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
/// `Position<RootInertial>`, `RotationalState`, etc. would require
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
    app.register_type::<components::PlanetOmegaC>();
    app.register_type::<components::PlanetAngularVelocityC>();
    app.register_type::<components::SourceFrameIdC>();
    app.register_type::<components::SourcePfixFrameIdC>();
    app.register_type::<components::IntegSourceC>();
    app.register_type::<components::FrameSwitchesC>();
    app.register_type::<components::BodyFrameIdC>();
    app.register_type::<components::IntegFrameIdC>();
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
    /// Wired in PR #260: `integ_source` (translated to
    /// [`components::IntegSourceC`] when `Some`) and `frame_switches`
    /// (translated to [`components::FrameSwitchesC`] when non-empty),
    /// retagging each `usize` source index to the matching ECS
    /// [`Entity`] from `source_entities`.
    ///
    /// Not yet wired (callers must insert these manually): drag, SRP
    /// (flat-plate / cannonball), shadow body, derived-state requests
    /// (orbital elements, Euler, LVLH, geodetic, solar beta, earth
    /// lighting). These are tracked for future expansion of
    /// `spawn_bevy`.
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
            // API. The Bevy `ExternalForceC` is typed (`Force<RootInertial>`),
            // so this is a one-time insertion-time lift — not a per-step
            // bypass. Migrating `VehicleConfig` itself to typed external
            // fields is a deeper refactor inside `jeod_sim`; out of
            // scope for the Bevy-adapter boundary that #172 H1 targets.
            let f = jeod_sim::Force::<jeod_sim::RootInertial>::from_raw_si(self.external_force); // allowed: #172 H1 insertion-time boundary (VehicleConfig still untyped)
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
        // Non-root integration: translate the `usize` source index to
        // the matching ECS Entity so `register_body_frames_system` can
        // resolve the body's integration frame against `FrameTreeR`.
        // `IntegSourceC(None)` is the implicit default (root), so we
        // only insert when the builder set a non-default integ source.
        if let Some(idx) = self.integ_source {
            let src = resolve_source_entity(source_entities, idx, "integ_source");
            entity.insert(components::IntegSourceC(Some(src)));
        }
        // Frame switches: translate each `FrameSwitchConfig<usize>` to
        // `FrameSwitchConfig<Entity>` by retagging `target_source`. The
        // bevy adapter's `frame_switch_system` reads
        // `FrameSwitchConfig<Entity>` directly. Skip the insertion when
        // the builder didn't configure any switches.
        if !self.frame_switches.is_empty() {
            let entity_switches: Vec<jeod_sim::FrameSwitchConfig<Entity>> = self
                .frame_switches
                .into_iter()
                .map(|sw| jeod_sim::FrameSwitchConfig::<Entity> {
                    target_source: resolve_source_entity(
                        source_entities,
                        sw.target_source,
                        "FrameSwitchConfig::target_source",
                    ),
                    switch_sense: sw.switch_sense,
                    switch_distance: sw.switch_distance,
                    active: sw.active,
                })
                .collect();
            entity.insert(components::FrameSwitchesC(entity_switches));
        }
        entity.id()
    }
}
