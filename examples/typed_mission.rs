//! Typed mission example — demonstrates the typed `jeod_sim::VehicleBuilder`
//! terminating into a Bevy spawn.
//!
//! This is the user-facing demonstration described in #101's end-state for
//! Phase 9: a mission author composes the vehicle with the typestate
//! `VehicleBuilder` (which gates "no state set" / "no integrator chosen"
//! at compile time) and ends with `spawn_bevy(...)` to put the resulting
//! components on a Bevy entity.
//!
//! Phase 9 of #101: this is the parallel of the standalone-runner
//! `Simulation` build path. Both consume the same `VehicleConfig` produced
//! by the same typestate builder, so mission code can swap between Bevy
//! and the standalone runner without rebuilding the configuration.

use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy_jeod::{
    GravityAccelerationC, GravitySourceC, JeodPlugin, JeodSet, SourceInertialPositionC,
    TotalForceC, TranslationalStateC, VehicleConfigBevyExt,
};
use jeod_sim::recipes::{constants, earth, orbital_elements, vehicle};
use jeod_sim::{F64Ext, GravityControl, VehicleBuilder};
use uom::si::length::meter;

#[derive(Resource)]
struct StepCounter(usize);

fn main() {
    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(0))))
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_plugins(JeodPlugin)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, log_orbit.after(JeodSet::Integration))
        .insert_resource(StepCounter(0))
        .run();
}

fn setup(mut commands: Commands, mut time: ResMut<Time<Virtual>>) {
    // Speed wall-clock so the example finishes in seconds.
    time.set_relative_speed_f64(1e6);

    // Spawn the Earth gravity source. The Bevy-side bundle wires only the
    // gravity-source half of the recipe; rotation and atmosphere are added
    // separately when needed.
    let earth_recipe = earth::point_mass();
    let earth_mu_raw = earth_recipe.source.mu;
    let earth = commands
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_recipe.source),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    // Build the vehicle via the typestate `VehicleBuilder`. The compiler
    // refuses to call `.three_dof_point_mass()` until a state is set; it
    // refuses `.rk4()` until the mass is set; it refuses `.build()` until
    // an integrator is chosen. All three are typestate gates.
    //
    // `from_orbital_elements` is the typed entry point — it consumes a
    // typed `GravParam` (here from `earth::point_mass().mu_typed()`) and
    // emits a typed translational state internally.
    // The recipe `earth::point_mass()` returns a `GravitySourceEntry`
    // (the runner's source-table row, with the raw `f64` mu). For the
    // typed builder, lift `mu` into a typed `GravParam`.
    let mu_typed = earth_mu_raw.m3_per_s2();
    let cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), mu_typed)
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        // Source index 0 in the per-config map → the Earth entity below.
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle_entity = cfg.spawn_bevy(&mut commands, &[earth]);
    commands
        .entity(vehicle_entity)
        .insert(Name::new("Satellite"));

    println!("Bevy JEOD typed-mission example");
    println!("===============================");
    println!("Spawned satellite via VehicleBuilder + spawn_bevy.");
    println!(
        "Earth radius: {:.0} m",
        constants::r_eq_earth().get::<meter>()
    );
    println!(
        "Vehicle mass: {:.0} kg",
        vehicle::iss_mass().get::<uom::si::mass::kilogram>()
    );
}

fn log_orbit(
    query: Query<(
        &Name,
        &TranslationalStateC,
        Option<&GravityAccelerationC>,
        Option<&TotalForceC>,
    )>,
    mut counter: ResMut<StepCounter>,
    mut exit: MessageWriter<AppExit>,
) {
    if counter.0 >= 560 {
        return;
    }
    counter.0 += 1;
    for (name, state, _grav, _total) in &query {
        if name.as_str() != "Satellite" {
            continue;
        }
        if counter.0 == 1 || counter.0.is_multiple_of(100) {
            let r_km = state.position.length() / 1000.0;
            let v = state.velocity.length();
            println!(
                "step={:5}  t={:8.0}s  |r|={:8.1}km  |v|={:.1}m/s",
                counter.0,
                counter.0 as f64 * 10.0,
                r_km,
                v
            );
        }
        if counter.0 >= 560 {
            println!("Completed ~1 orbit. Exiting.");
            exit.write(AppExit::Success);
            return;
        }
    }
}

// `F64Ext` is brought in scope above so downstream calls like
// `1.5.ms()`, `42.0.kg()` work for mission code that wants typed
// constructions. The example itself uses recipe presets, but the
// import demonstrates the available surface.
#[allow(dead_code)]
fn _showcase_f64_ext() {
    let _: uom::si::f64::Mass = 420_000.0.kg();
    let _: uom::si::f64::Length = 6_378_137.0.m();
    let _: uom::si::f64::Angle = 51.6_f64.deg();
}
