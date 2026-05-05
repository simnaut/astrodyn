//! Smoke test for the Bevy `Reflect` derives on the typed Components (#154).
//!
//! Verifies three things:
//!
//! 1. **The registration list is centralized.** Both `JeodPlugin::build`
//!    and downstream callers populate the `AppTypeRegistry` via
//!    [`register_jeod_component_types`], so the production registration
//!    set and what the tests cover cannot drift apart.
//! 2. **Every `Reflect`-derived Component in `src/components.rs` shows
//!    up in the registry** — addresses the previous miss where
//!    `JeodPlugin` registered only a 26-entry subset of the 44 derived
//!    types.
//! 3. **Reflective component insertion actually works.** A live `World`
//!    looks up `TranslationalStateC` by type path, retrieves its
//!    `ReflectComponent` registration, and uses it to insert / remove
//!    the Component on an entity through the `Reflect` trait object —
//!    proving the editor / scene-tooling path is wired, not just the
//!    static type-registry side.
//!
//! Inner `jeod_*` types are `#[reflect(opaque)]`, so field-level
//! introspection (e.g. `position[0]: f64` inside a
//! `TranslationalStateC`) is *not* in scope here. That requires
//! propagating `Reflect` into the source crates and is the natural
//! follow-up.

use bevy::prelude::*;
use bevy_jeod::*;

/// The full inventory of `Reflect`-derived Components in
/// `src/components.rs`. Kept in sync with
/// `register_jeod_component_types`. If a Component is added there but
/// not added here, `every_reflect_derived_component_is_registered`
/// will fail and the test forces the inventory to update.
const EXPECTED_REGISTERED_TYPE_PATHS: &[&str] = &[
    // Dynamics state
    "bevy_jeod::components::TranslationalStateC",
    "bevy_jeod::components::RotationalStateC",
    "bevy_jeod::components::MassPropertiesC",
    "bevy_jeod::components::GravityAccelerationC",
    "bevy_jeod::components::TotalForceC",
    "bevy_jeod::components::FrameDerivativesC",
    // Dynamics config + integrator state
    "bevy_jeod::components::DynamicsConfigC",
    "bevy_jeod::components::IntegratorTypeC",
    "bevy_jeod::components::GaussJacksonStateC",
    "bevy_jeod::components::Abm4StateC",
    // Gravity
    "bevy_jeod::components::GravityControlsC",
    "bevy_jeod::components::GravitySourceC",
    "bevy_jeod::components::SourceInertialPositionC",
    "bevy_jeod::components::SourceInertialVelocityC",
    // Interactions
    "bevy_jeod::components::AerodynamicForceC",
    "bevy_jeod::components::RadiationForceC",
    "bevy_jeod::components::GravityTorqueC",
    "bevy_jeod::components::AtmosphericStateC",
    // Frame transforms
    "bevy_jeod::components::StructuralTransformC",
    "bevy_jeod::components::PlanetFixedRotationC",
    // Frame-tree (issue #71)
    "bevy_jeod::components::PlanetOmegaC",
    "bevy_jeod::components::PlanetAngularVelocityC",
    "bevy_jeod::components::IntegSourceC",
    "bevy_jeod::components::FrameSwitchesC",
    // Frame-tree as ECS entities (issue #277)
    "bevy_jeod::components::FrameTransC",
    "bevy_jeod::components::FrameRotC",
    "bevy_jeod::components::FrameAngVelC",
    "bevy_jeod::components::InertialFrameMarker",
    "bevy_jeod::components::PlanetFixedFrameMarker",
    "bevy_jeod::components::BodyFrameMarker",
    "bevy_jeod::components::IntegrationFrameMarker",
    "bevy_jeod::components::FrameEntityC",
    "bevy_jeod::components::PfixFrameEntityC",
    "bevy_jeod::components::RetiredPfixFrameEntityC",
    "bevy_jeod::components::FrameAttachedC",
    "bevy_jeod::components::JointKinematicsC",
    "bevy_jeod::components::SinusoidalJointKinematicsC",
    "bevy_jeod::components::ClosureJointKinematicsC",
    "bevy_jeod::components::MultiDofJointKinematicsC",
    // Tidal
    "bevy_jeod::components::TidalConfigC",
    "bevy_jeod::components::TidalDeltaC20C",
    // Drag / SRP
    "bevy_jeod::components::DragConfigC",
    "bevy_jeod::components::FlatPlateConfigC",
    "bevy_jeod::components::CannonballSrpC",
    "bevy_jeod::components::ShadowBodyC",
    // External loads
    "bevy_jeod::components::ExternalForceC",
    "bevy_jeod::components::ExternalTorqueC",
    // Body / planet identity + ephemeris
    "bevy_jeod::components::MassBodyIdC",
    // Mass-tree relations (issue #271)
    "bevy_jeod::components::MassChildOf",
    "bevy_jeod::components::MassPointRef",
    // Detached subtree state (carried by detached chain roots).
    "bevy_jeod::components::DetachedSubtreeStateC",
    // Composite-rigid-body kinematic-children gate.
    "bevy_jeod::components::KinematicChildC",
    "bevy_jeod::components::PlanetC",
    "bevy_jeod::components::RotationModelC",
    "bevy_jeod::components::EphemerisBodyC",
    "bevy_jeod::components::SunMarker",
    "bevy_jeod::components::MoonMarker",
    "bevy_jeod::components::CentralSourceMarker",
    // Derived-state config
    "bevy_jeod::components::OrbitalElementsConfigC",
    "bevy_jeod::components::EulerAnglesConfigC",
    "bevy_jeod::components::GeodeticConfigC",
    "bevy_jeod::components::EarthLightingConfigC",
    // Derived-state output
    "bevy_jeod::components::OrbitalElementsC",
    "bevy_jeod::components::EulerAnglesC",
    "bevy_jeod::components::LvlhFrameC",
    "bevy_jeod::components::GeodeticStateC",
    "bevy_jeod::components::SolarBetaC",
    "bevy_jeod::components::EarthLightingStateC",
];

/// Every `Reflect`-derived Component in `src/components.rs` is
/// reachable from the `AppTypeRegistry` after
/// `register_jeod_component_types` runs against a bare `App`.
#[test]
fn every_reflect_derived_component_is_registered() {
    let mut app = App::new();
    register_jeod_component_types(&mut app);
    let registry = app.world().resource::<AppTypeRegistry>().read();
    let missing: Vec<&str> = EXPECTED_REGISTERED_TYPE_PATHS
        .iter()
        .copied()
        .filter(|p| registry.get_with_type_path(p).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "register_jeod_component_types missed Components:\n  {}",
        missing.join("\n  "),
    );
}

/// Bidirectional check (PR #283 review thread PRRT_kwDORtae6c5_KHnW
/// — Copilot round 3): every `bevy_jeod::components::*` type that
/// appears in the live registry is enumerated in
/// [`EXPECTED_REGISTERED_TYPE_PATHS`]. A new
/// `register_type::<components::Foo>` call without a matching entry
/// in this inventory is now a test failure, instead of silently
/// drifting. This prevents the original `MassChildOf` / `MassPointRef`
/// gap (registered in `JeodPlugin::build` but unlisted) from
/// recurring.
#[test]
fn no_unlisted_bevy_jeod_component_is_registered() {
    let mut app = App::new();
    register_jeod_component_types(&mut app);
    let registry = app.world().resource::<AppTypeRegistry>().read();
    let expected: std::collections::HashSet<&str> =
        EXPECTED_REGISTERED_TYPE_PATHS.iter().copied().collect();
    let extras: Vec<String> = registry
        .iter()
        .map(|reg| reg.type_info().type_path().to_string())
        .filter(|p| p.starts_with("bevy_jeod::components::"))
        .filter(|p| !expected.contains(p.as_str()))
        .collect();
    assert!(
        extras.is_empty(),
        "register_jeod_component_types registered Components not listed in \
         EXPECTED_REGISTERED_TYPE_PATHS — add them to the inventory:\n  {}",
        extras.join("\n  "),
    );
}

/// `JeodPlugin::build` populates the same registry as the standalone
/// helper. Confirms the production path matches the test path.
#[test]
fn jeod_plugin_registers_full_component_inventory() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);

    let registry = app.world().resource::<AppTypeRegistry>().read();
    let missing: Vec<&str> = EXPECTED_REGISTERED_TYPE_PATHS
        .iter()
        .copied()
        .filter(|p| registry.get_with_type_path(p).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "JeodPlugin missed Components:\n  {}",
        missing.join("\n  "),
    );
}

/// Reflective component insertion: looks up `TranslationalStateC` in
/// the registry, retrieves the `ReflectComponent` registration, and
/// inserts the Component onto an entity through a `&dyn Reflect`. This
/// is the path Bevy editor / scene-deserialization tooling uses.
#[test]
fn reflective_component_insertion_round_trips() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);

    let entity = app.world_mut().spawn_empty().id();

    // Build a default `TranslationalStateC`, lift it to `&dyn Reflect`,
    // and insert via the registry's `ReflectComponent`. This exercises
    // the same code path Bevy editor tooling uses to materialize a
    // Component from a scene file.
    let initial = components::TranslationalStateC::default();
    let registry = app.world().resource::<AppTypeRegistry>().clone();
    {
        let registry_read = registry.read();
        let registration = registry_read
            .get_with_type_path("bevy_jeod::components::TranslationalStateC")
            .expect("TranslationalStateC must be registered");
        let reflect_component = registration
            .data::<bevy::ecs::reflect::ReflectComponent>()
            .expect("TranslationalStateC must carry ReflectComponent data");
        reflect_component.insert(
            &mut app.world_mut().entity_mut(entity),
            &initial,
            &registry_read,
        );
    }

    // Verify the insert took effect — the Component is on the entity
    // and matches what we inserted.
    let stored = app
        .world()
        .get::<components::TranslationalStateC>(entity)
        .expect("TranslationalStateC should be on the entity after reflective insert");
    assert_eq!(stored.0.position, initial.0.position);
    assert_eq!(stored.0.velocity, initial.0.velocity);

    // And reflective remove works too.
    {
        let registry_read = registry.read();
        let registration = registry_read
            .get_with_type_path("bevy_jeod::components::TranslationalStateC")
            .expect("TranslationalStateC must be registered");
        let reflect_component = registration
            .data::<bevy::ecs::reflect::ReflectComponent>()
            .expect("TranslationalStateC must carry ReflectComponent data");
        reflect_component.remove(&mut app.world_mut().entity_mut(entity));
    }
    assert!(
        app.world()
            .get::<components::TranslationalStateC>(entity)
            .is_none(),
        "TranslationalStateC should have been removed by reflective remove",
    );
}
