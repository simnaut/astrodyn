//! Mission-crate ergonomics regression net.
//!
//! Phase 11 of #101. This integration test mocks the lifecycle of a
//! downstream mission crate that depends only on `astrodyn_bevy` — it imports
//! from `astrodyn_bevy::prelude` and `astrodyn_bevy::recipes`, spawns an ISS-like
//! LEO scenario via the typestate `VehicleBuilder`, and propagates for
//! ~1 hour via Bevy's `FixedUpdate`. The assertion verifies the final
//! position magnitude is plausible for a 400 km orbit.
//!
//! If a future refactor breaks the prelude / recipes facade — e.g., a
//! removed re-export, a renamed Component, a typestate gate that no longer
//! type-checks the canonical mission flow — this test breaks, surfacing
//! the regression at PR time rather than after a downstream consumer's
//! upgrade.
//!
//! The test is gated on neither `JEOD_HOME` nor any `.bsp` file: it uses
//! point-mass gravity from `recipes::earth::point_mass()` and the ISS
//! orbital elements from `recipes::orbital_elements::iss()`, both of which
//! are pure constants compiled into the recipe.

use std::time::Duration;

use astrodyn_bevy::prelude::*;
use astrodyn_bevy::recipes::{earth, orbital_elements, vehicle};
use bevy::prelude::*;

#[derive(Resource)]
struct VehicleEntity(Entity);

fn setup_iss(mut commands: Commands) {
    let earth_recipe = earth::point_mass();
    let earth_mu = earth_recipe.source.mu;
    let earth = commands
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_recipe.source),
            SourceInertialPositionC::default(),
            TranslationalStateC::<Earth>::default(),
        ))
        .id();

    let cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), earth_mu.m3_per_s2())
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle_entity = cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth]);
    commands.insert_resource(VehicleEntity(vehicle_entity));
}

#[test]
fn mission_crate_sanity_iss_one_hour() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_plugins(JeodPlugin)
        .add_systems(Startup, setup_iss);

    // Run startup once.
    app.update();

    // Step 360 × 10 s = 3600 s ≈ 1 hour. Each FixedUpdate tick advances
    // the JEOD pipeline by one `dt`.
    let total_seconds = 3600.0;
    let dt = 10.0;
    let n_steps = (total_seconds / dt) as u32;
    for _ in 0..n_steps {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }

    // Read the typed Component back via a query — the same path a
    // downstream consumer uses.
    let vehicle = app.world().resource::<VehicleEntity>().0;
    let state = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
        .expect("vehicle has TranslationalStateC after propagation");
    // `state.position` / `state.velocity` are typed; `.length()` returns
    // a `Quantity` (Length / Velocity). Drop to f64 SI base for the
    // numeric range checks below.
    let r_mag: f64 = state.position.length().value;
    let v_mag: f64 = state.velocity.length().value;

    // ISS reference orbit: ~408 km altitude → r ≈ 6.778 Mm.
    // Energy conservation under point-mass gravity holds r within a
    // narrow band over 1 hour (0.058 of a 5550 s period). Allow ±50 km
    // generously; a regression in the facade breaks orders of magnitude,
    // not km.
    assert!(
        (6.728e6..=6.828e6).contains(&r_mag),
        "ISS position magnitude after 1h propagation outside plausible band: \
         r = {r_mag:.1} m (expected ~6.778 Mm ± 50 km)"
    );

    // Circular-orbit speed at this altitude is ~7.66 km/s. Same generous
    // ±100 m/s band — facade-regression magnitudes, not numerical noise.
    assert!(
        (7560.0..=7760.0).contains(&v_mag),
        "ISS velocity magnitude after 1h propagation outside plausible band: \
         v = {v_mag:.1} m/s (expected ~7660 m/s ± 100 m/s)"
    );
}
