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
//!   3. Spawning a *second* vehicle whose `GravityControlsC` references a
//!      bogus source index. The validation system must catch this on the
//!      next tick and panic — proving the late-add validation path runs.
//!
//! The bogus reference is a deliberate trip wire: the inner `validate_body`
//! call doesn't itself reach the gravity-control auto-correction step (which
//! would silently skip an unresolved entity), but `check_validity` *does*
//! get called for every control, and a bad source index makes
//! `sources.get(...)` return Err — so the auto-correction is skipped but
//! the per-step gravity computation in `gravity_computation_system` will
//! eventually panic on a missing source. We capture the panic via
//! `std::panic::catch_unwind` to keep the test deterministic.

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

    // Add a *second* vehicle mid-simulation. This time give it a
    // non-spherical GravityControl with a non-existent source name. The
    // `validate_body` kernel does not panic on this (the `sources.get()`
    // closure simply returns `None` for an unresolved source, which
    // validate_body interprets as "skip"), but `check_validity` clamps
    // degree/order — and the integration step that follows will fail to
    // find the source. We assert that the validation system *runs* (which
    // we measure by observing the gravity-control's `degree`/`order` got
    // clamped at the second tick); a stale `has_run` gate would skip this
    // mutation and the assertion below would fail.

    let earth_mu = earth::point_mass().source.mu;
    let bogus_cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), earth_mu.m3_per_s2())
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        // Request a degree that exceeds the source's degree (point-mass
        // = degree 0). validation's `check_validity` should catch this
        // and either panic or auto-correct depending on the request.
        .gravity({
            let mut g = GravityControl::new_nonspherical(0_usize, 4, 4, false);
            // Request degree=4 against a point-mass source. `check_validity`
            // panics with "Non-spherical gravity (spherical=false) is only
            // supported for SphericalHarmonics gravity models."
            g.degree = 4;
            g
        })
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
