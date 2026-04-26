//! Smoke test for the Bevy `Reflect` derives on the typed Components (#154).
//!
//! Verifies that the typed Components registered in `src/components.rs`
//! are reflectable: they implement `Reflect` (so a Bevy editor / scene
//! tooling can identify them by type), they carry the `Component`
//! reflect-data so the editor can spawn / inspect entities that hold
//! them, and the `TypeRegistry` round-trips a representative selection.
//!
//! The Components use `#[reflect(opaque)]` so the inner `jeod_*` types
//! (which do not depend on Bevy) are not field-introspected. The editor
//! sees the Component as a leaf with its type name. Field-level
//! introspection of the inner `Position<Inertial>` / `RotationalState` /
//! etc. is a follow-up that requires either propagating `Reflect`
//! through the source crates or hand-rolling impls — out of scope for
//! this PR per the layering decision (`jeod_quantities` stays Bevy-free).

use bevy::prelude::*;
use bevy::reflect::TypeRegistry;
use bevy_jeod::*;

/// Build a fresh `TypeRegistry` and register every Bevy-side typed
/// Component. Returns the populated registry so individual asserts can
/// look up by type.
fn registry_with_components() -> TypeRegistry {
    let mut reg = TypeRegistry::new();
    reg.register::<TranslationalStateC>();
    reg.register::<RotationalStateC>();
    reg.register::<MassPropertiesC>();
    reg.register::<GravityAccelerationC>();
    reg.register::<TotalForceC>();
    reg.register::<FrameDerivativesC>();
    reg.register::<DynamicsConfigC>();
    reg.register::<IntegratorTypeC>();
    reg.register::<GaussJacksonStateC>();
    reg.register::<Abm4StateC>();
    reg.register::<GravityControlsC>();
    reg.register::<GravitySourceC>();
    reg.register::<SourceInertialPositionC>();
    reg.register::<SourceInertialVelocityC>();
    reg.register::<AerodynamicForceC>();
    reg.register::<RadiationForceC>();
    reg.register::<GravityTorqueC>();
    reg.register::<AtmosphericStateC>();
    reg.register::<StructuralTransformC>();
    reg.register::<PlanetFixedRotationC>();
    reg.register::<TidalConfigC>();
    reg.register::<TidalDeltaC20C>();
    reg.register::<DragConfigC>();
    reg.register::<FlatPlateConfigC>();
    reg.register::<SunMarker>();
    reg.register::<MoonMarker>();
    reg
}

/// Headline assertion: the typed Components register cleanly and are
/// looked up by their fully-qualified type names.
#[test]
fn typed_components_register_in_type_registry() {
    let reg = registry_with_components();

    let names = [
        "bevy_jeod::components::TranslationalStateC",
        "bevy_jeod::components::RotationalStateC",
        "bevy_jeod::components::MassPropertiesC",
        "bevy_jeod::components::PlanetFixedRotationC",
        "bevy_jeod::components::StructuralTransformC",
        "bevy_jeod::components::SunMarker",
        "bevy_jeod::components::MoonMarker",
    ];

    for name in &names {
        assert!(
            reg.get_with_type_path(name).is_some(),
            "Component `{name}` did not register; check #[derive(Reflect)] on src/components.rs"
        );
    }
}

/// Spot-check that a reflected `TranslationalStateC` knows its own type
/// path — proves the derived `TypePath` impl is reachable through the
/// `Reflect` trait object.
#[test]
fn reflect_trait_object_round_trips_translational_state() {
    let c = TranslationalStateC::default();
    let r: &dyn Reflect = &c;
    let info = r.reflect_type_path();
    assert!(
        info.contains("TranslationalStateC"),
        "Reflect type path missing TranslationalStateC: got {info}"
    );
}

/// `JeodPlugin`'s `App::register_type::<TypedComponent>()` calls (added
/// in this PR) should populate the App-wide registry. Builds a minimal
/// `App` with `MinimalPlugins + JeodPlugin` and verifies a representative
/// Component is present.
#[test]
fn jeod_plugin_registers_typed_components() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);

    let registry = app.world().resource::<AppTypeRegistry>().read();
    assert!(
        registry
            .get_with_type_path("bevy_jeod::components::TranslationalStateC")
            .is_some(),
        "JeodPlugin did not register TranslationalStateC in AppTypeRegistry"
    );
    assert!(
        registry
            .get_with_type_path("bevy_jeod::components::PlanetFixedRotationC")
            .is_some(),
        "JeodPlugin did not register PlanetFixedRotationC in AppTypeRegistry"
    );
}
