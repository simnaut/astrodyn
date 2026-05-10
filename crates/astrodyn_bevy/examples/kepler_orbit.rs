//! Bevy ECS Kepler orbit example.
//!
//! Spawns an Earth gravity source and a satellite in a 400 km circular
//! orbit, then propagates for approximately one orbital period using
//! JEOD's RK4 integrator in `FixedUpdate`. Prints orbital state every
//! 100 steps and exits after ~1 orbit.
//!
//! Phase 6 of #101: building blocks for the Earth source and the
//! initial state come from [`recipes`](astrodyn::recipes) so this
//! example shares its "Earth+ISS" composition with the standalone-runner
//! examples and Tier 3 cases. The Bevy spawning here is intentionally
//! manual — it shows how the underlying components fit together. For
//! whole-scenario composition (multiple sources, mass trees,
//! ephemeris, atmosphere, polar motion) see
//! [`SimulationBuilderBevyExt::populate_app`](astrodyn_bevy::SimulationBuilderBevyExt::populate_app)
//! and `examples/multi_body_scenario.rs` — that's the canonical
//! recipe-driven entry point. A future `commands.spawn_scenario(s)`
//! extension on `&mut Commands` is a separate, system-friendly form
//! that's distinct from the existing `&mut App` terminal.

use astrodyn::init_from_orbital_elements_typed;
use astrodyn::recipes::{constants, earth, orbital_elements, vehicle};
use astrodyn::{GravityControl, GravityControls, GravityGradient};
use astrodyn_bevy::{
    AstrodynAppExt, AstrodynSet, DynamicsConfigC, FrameDerivativesC, GravityAccelerationC,
    GravityControlsC, GravitySourceC, MassPropertiesC, SourceInertialPositionC, TotalForceC,
    TranslationalStateC,
};
use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use glam::DVec3;
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
/// Rejects `0` because `MaxSteps(0)` would make `print_state` early-return on
/// every tick without ever writing `AppExit` — the app would hang forever.
fn parse_steps_arg() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--steps" {
            let val = args
                .next()
                .expect("--steps requires a value, e.g. --steps 10");
            let n: usize = val
                .parse()
                .unwrap_or_else(|err| panic!("--steps value {val:?} is not a usize: {err}"));
            assert!(
                n >= 1,
                "--steps must be >= 1; got {n}. The example exits after writing \
                 AppExit on step >= max_steps; with max_steps == 0 the app would \
                 hang forever. Pass at least --steps 1.",
            );
            return n;
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
        .add_astrodyn(10.0)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, print_state.after(AstrodynSet::Integration))
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
    // pfix transform, …) is wired by Bevy systems separately. For
    // whole-scenario composition that hides this manual wiring, reach
    // for `SimulationBuilderBevyExt::populate_app` (see
    // `examples/multi_body_scenario.rs`); the manual setup below is
    // kept here as the documentation of the underlying components.
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
    // helper. The typed output is `<RootInertial>`; relabel to
    // `<PlanetInertial<Earth>>` for the Bevy component (the numerics
    // are bit-identical for the root-integrated body's planet). The
    // recipe-driven path (`SimulationBuilderBevyExt::populate_app`)
    // folds this conversion into the scenario factory; see
    // `examples/multi_body_scenario.rs` for that flow.
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
    let trans_planet = trans_typed.relabel_to::<astrodyn::PlanetInertial<astrodyn::Earth>>();
    let initial_radius_m: f64 = trans_planet.position.length().value;

    let mass_kg = vehicle::iss_mass().get::<kilogram>();
    let controls = GravityControls {
        controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
    };

    commands.spawn((
        Name::new("Satellite"),
        TranslationalStateC::<astrodyn::Earth>::point_mass(
            trans_planet.position,
            trans_planet.velocity,
        ),
        // JEOD_INV: TS.01 — `<SelfRef>` is the storage-side wildcard for the spawned vehicle's MassPropertiesC.
        MassPropertiesC::from(astrodyn::MassPropertiesTyped::<astrodyn::SelfRef>::new(
            astrodyn::Mass::new::<astrodyn::kilogram>(mass_kg),
        )),
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
        (initial_radius_m - r_eq_earth_m()) / 1000.0
    );
}

fn print_state(
    query: Query<(&Name, &TranslationalStateC<astrodyn::Earth>)>,
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
