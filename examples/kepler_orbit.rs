//! Bevy ECS Kepler orbit example.
//!
//! Spawns an Earth gravity source and a satellite in a 400 km circular
//! orbit, then propagates for approximately one orbital period using
//! JEOD's RK4 integrator in `FixedUpdate`. Prints orbital state every
//! 100 steps and exits after ~1 orbit.
//!
//! Phase 6 of #101: building blocks for the Earth source and the
//! initial state come from
//! [`recipes`](jeod_sim::recipes) so this example shares its
//! "Earth+ISS" composition with the standalone-runner examples and
//! Tier 3 cases. Phase 9 will introduce a `commands.spawn_scenario(s)`
//! extension that lets a Bevy app consume a full scenario in one
//! line; until then, the Bevy spawning is still manual.

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, FrameDerivativesC, GravityAccelerationC, GravityControlsC, GravitySourceC,
    JeodPlugin, JeodSet, MassPropertiesC, SourceInertialPositionC, TotalForceC,
    TranslationalStateC,
};
use glam::DVec3;
use jeod_dynamics::body_init::init_from_orbital_elements_typed;
use jeod_sim::recipes::{constants, earth, orbital_elements, vehicle};
use jeod_sim::{GravityControl, GravityControls, MassProperties, TranslationalState};
use std::time::Duration;
use uom::si::angle::radian;
use uom::si::f64::{Angle, Length};
use uom::si::length::meter;
use uom::si::mass::kilogram;

const MU_EARTH: f64 = 3.986_004_415e14;

/// Default step count: ~one ISS orbit at dt=10s.
const DEFAULT_STEPS: usize = 560;

/// Parse `--steps N` from CLI args; default to [`DEFAULT_STEPS`] when absent.
/// Panics with a clear message on a malformed value (per fail-loudly policy).
fn parse_steps_arg() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--steps" {
            let val = args
                .next()
                .expect("--steps requires a value, e.g. --steps 10");
            return val
                .parse::<usize>()
                .unwrap_or_else(|err| panic!("--steps value {val:?} is not a usize: {err}"));
        }
    }
    DEFAULT_STEPS
}

/// Earth equatorial radius from `recipes::constants::r_eq_earth()`.
/// Used to compute the printed altitude — keeps the example aligned
/// with the rest of the recipe-based examples if the constant ever
/// changes.
fn r_eq_earth_m() -> f64 {
    constants::r_eq_earth().get::<meter>()
}

fn eccentricity(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    let h = position.cross(velocity);
    let e_vec = velocity.cross(h) / mu - position.normalize();
    e_vec.length()
}

fn main() {
    let max_steps = parse_steps_arg();
    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(0))))
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_plugins(JeodPlugin)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, print_state.after(JeodSet::Integration))
        .insert_resource(StepCounter(0))
        .insert_resource(MaxSteps(max_steps))
        .run();
}

#[derive(Resource)]
struct StepCounter(usize);

#[derive(Resource)]
struct MaxSteps(usize);

fn setup(mut commands: Commands, mut time: ResMut<Time<Virtual>>) {
    time.set_relative_speed_f64(1e6);

    // Earth gravity source. The Bevy adapter consumes only the
    // `GravitySource` (mu + model) from the recipe entry; the rest of
    // the standalone-runner `GravitySourceEntry` (rotation model,
    // pfix transform, …) is wired by Bevy systems separately. Phase 9
    // will add a `commands.spawn_scenario(s)` extension that hides
    // this conversion.
    let earth_recipe = earth::point_mass();
    let earth = commands
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_recipe.source),
            SourceInertialPositionC::default(),
        ))
        .id();

    // Pull the orbital-elements preset (`OrbitalElements` carries plain
    // `f64` SI fields — JEOD-faithful but not yet `uom`-typed) and
    // initialize a typed `TranslationalState` via the typed orbit-init
    // helper. Phase 9 will introduce a `commands.spawn_scenario(s)`
    // extension that hides this conversion.
    let oe = orbital_elements::iss();
    let trans_typed = init_from_orbital_elements_typed(
        Length::new::<meter>(oe.semi_major_axis),
        oe.e_mag,
        Angle::new::<radian>(oe.inclination),
        Angle::new::<radian>(oe.long_asc_node),
        Angle::new::<radian>(oe.arg_periapsis),
        Angle::new::<radian>(oe.true_anom),
        constants::mu_ggm05c(),
    );
    let trans: TranslationalState = trans_typed.to_untyped();

    let mass_kg = vehicle::iss_mass().get::<kilogram>();
    let controls = GravityControls {
        controls: vec![GravityControl::new_spherical(earth, false)],
    };

    commands.spawn((
        Name::new("Satellite"),
        TranslationalStateC::from(trans),
        MassPropertiesC::from(MassProperties::new(mass_kg)),
        GravityAccelerationC::default(),
        TotalForceC::default(),
        FrameDerivativesC::default(),
        DynamicsConfigC::default(),
        GravityControlsC(controls),
    ));

    println!("Bevy JEOD Kepler Orbit Example");
    println!("==============================");
    println!(
        "Initial altitude: {:.1} km",
        (trans.position.length() - r_eq_earth_m()) / 1000.0
    );
}

fn print_state(
    query: Query<(&Name, &TranslationalStateC)>,
    mut counter: ResMut<StepCounter>,
    max_steps: Res<MaxSteps>,
    mut exit: MessageWriter<AppExit>,
) {
    if counter.0 >= max_steps.0 {
        return;
    }
    counter.0 += 1;
    for (name, state) in &query {
        if name.as_str() != "Satellite" {
            continue;
        }
        if counter.0.is_multiple_of(100) || counter.0 <= 1 {
            // `state.position` / `state.velocity` are typed; `.length().value`
            // reads the SI base. The `eccentricity` helper still takes
            // raw `DVec3`, so drop the phantom there too.
            let v: f64 = state.velocity.length().value;
            let alt_km: f64 = (state.position.length().value - r_eq_earth_m()) / 1000.0;
            let e_mag = eccentricity(MU_EARTH, state.position.raw_si(), state.velocity.raw_si());
            println!(
                "step={:5}  t={:8.0}s  alt={:7.1}km  v={:.1}m/s  e={:.2e}",
                counter.0,
                counter.0 as f64 * 10.0,
                alt_km,
                v,
                e_mag
            );
        }
        if counter.0 >= max_steps.0 {
            println!("Completed {} steps. Exiting.", max_steps.0);
            exit.write(AppExit::Success);
            return;
        }
    }
}
