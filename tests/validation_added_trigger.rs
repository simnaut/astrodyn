//! Regression test for #172 M3: validation runs on bodies added mid-simulation.
//!
//! Before #172 M3 the validation system was gated by `Local<bool> has_run`,
//! so any body added after the first `FixedUpdate` tick skipped validation
//! entirely. The fix replaces the local with an `Added<GravityControlsC>`
//! filter on the body query.
//!
//! This test exercises the late-addition path by:
//!   1. Building an `App` with `JeodPlugin` and one Earth source plus one
//!      vehicle attached at startup.
//!   2. Running several `FixedUpdate` ticks so the startup body has been
//!      validated and `Added` is no longer set on it.
//!   3. Spawning a *second* vehicle after those ticks whose
//!      `GravityControlsC` still references source index `0` (which
//!      resolves to Earth), but is intentionally invalid for that source.
//!
//! The deliberate trip wire is `GravityControl::new_nonspherical(0, 4, 4,
//! false)` against a `PointMass` Earth source. When the late-added body is
//! picked up by the `Added<GravityControlsC>` validation path on the next
//! tick, `check_validity()` rejects that configuration and panics. That
//! panic is what proves validation is being rerun for newly-added bodies,
//! rather than the body slipping past validation after startup. We capture
//! the panic via `std::panic::catch_unwind` to keep the test deterministic.

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::prelude::*;
use bevy_jeod::recipes::{earth, orbital_elements, vehicle};

fn build_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_plugins(JeodPlugin);

    // Spawn Earth + ISS body via Startup so the regular pipeline runs them.
    let earth_recipe = earth::point_mass();
    let earth_mu = earth_recipe.source.mu;
    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_recipe.source),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), earth_mu.m3_per_s2())
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let mut commands_state = bevy::ecs::system::SystemState::<Commands>::new(app.world_mut());
    let mut commands = commands_state.get_mut(app.world_mut());
    let _ = cfg.spawn_bevy(&mut commands, &[earth]);
    commands_state.apply(app.world_mut());

    // Drive startup + two FixedUpdate ticks so the body has been validated
    // and the `Added<GravityControlsC>` filter no longer matches it.
    app.update();
    for _ in 0..2 {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(10.0));
        app.world_mut().run_schedule(FixedUpdate);
    }

    (app, earth)
}

#[test]
fn validation_fires_for_body_added_after_startup() {
    let (mut app, earth) = build_app();

    // Add a *second* vehicle mid-simulation. Give it a non-spherical
    // GravityControl targeting the existing Earth source, even though
    // Earth is configured as a point-mass model. When this body is
    // inserted, `Added<GravityControlsC>` should cause validation to run
    // on the next `FixedUpdate`. That validation panics immediately
    // because non-spherical gravity is only supported for
    // `SphericalHarmonics` sources; if the old `Local<bool>` gate were
    // still suppressing validation after startup, the schedule would
    // complete without error and the assertion below would fail.

    let earth_mu = earth::point_mass().source.mu;
    let bogus_cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), earth_mu.m3_per_s2())
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        // Request degree=4 against a point-mass source. `check_validity`
        // panics with "Non-spherical gravity (spherical=false) is only
        // supported for SphericalHarmonics gravity models."
        .gravity(GravityControl::new_nonspherical(0_usize, 4, 4, false))
        .build();

    let mut commands_state = bevy::ecs::system::SystemState::<Commands>::new(app.world_mut());
    let mut commands = commands_state.get_mut(app.world_mut());
    let _ = bogus_cfg.spawn_bevy(&mut commands, &[earth]);
    commands_state.apply(app.world_mut());

    // Step once more — validation must run for the new body and panic.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(10.0));
        app.world_mut().run_schedule(FixedUpdate);
    }));

    // If the validation system had been gated by `Local<bool>`, no panic
    // would fire because the gate would have closed after the first tick.
    let panic = result.expect_err(
        "expected validation to panic on the late-added body's bad GravityControl, \
         but the FixedUpdate schedule completed without error — \
         the Added<GravityControlsC> trigger is not firing for late additions",
    );
    let msg = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&'static str>().copied())
        .unwrap_or("<non-string panic payload>");
    assert!(
        msg.contains("Non-spherical gravity")
            || msg.contains("PointMass")
            || msg.contains("SphericalHarmonics"),
        "panic message did not mention gravity validation: {msg}"
    );
}
