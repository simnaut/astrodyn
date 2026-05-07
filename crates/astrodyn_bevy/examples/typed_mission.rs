//! Typed mission example — demonstrates the typed `astrodyn::VehicleBuilder`
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

use astrodyn::recipes::{constants, earth, orbital_elements, vehicle};
use astrodyn::{F64Ext, GravityControl, VehicleBuilder};
use astrodyn_bevy::{
    AstrodynPlugin, AstrodynSet, GravityAccelerationC, GravitySourceC, SourceInertialPositionC,
    TotalForceC, TranslationalStateC, VehicleConfigBevyExt,
};
use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;

#[derive(Resource)]
struct StepCounter(usize);

#[derive(Resource)]
struct MaxSteps(usize);

/// Default step count: ~one ISS orbit at dt=10s.
const DEFAULT_STEPS: usize = 560;

/// Parse `--steps N` from CLI args; default to [`DEFAULT_STEPS`] when absent.
/// Panics with a clear message on a malformed value (per fail-loudly policy).
/// Rejects `0` because `MaxSteps(0)` would make `log_orbit` early-return on
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

fn main() {
    let max_steps = parse_steps_arg();
    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(0))))
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_plugins(AstrodynPlugin)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, log_orbit.after(AstrodynSet::Integration))
        .insert_resource(StepCounter(0))
        .insert_resource(MaxSteps(max_steps))
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
            TranslationalStateC::<astrodyn::Earth>::default(),
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

    let vehicle_entity = cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth]);
    commands
        .entity(vehicle_entity)
        .insert(Name::new("Satellite"));

    println!("Bevy JEOD typed-mission example");
    println!("===============================");
    println!("Spawned satellite via VehicleBuilder + spawn_bevy.");
    // `Length::value` and `Mass::value` are SI base units (m, kg) by uom
    // convention — equivalent to `.get::<meter>()` / `.get::<kilogram>()`
    // but without importing uom unit types directly.
    println!("Earth radius: {:.0} m", constants::r_eq_earth().value);
    println!("Vehicle mass: {:.0} kg", vehicle::iss_mass().value);
}

#[allow(clippy::type_complexity)]
fn log_orbit(
    query: Query<(
        &Name,
        &TranslationalStateC<astrodyn::Earth>,
        Option<&GravityAccelerationC>,
        Option<&TotalForceC>,
    )>,
    mut counter: ResMut<StepCounter>,
    max_steps: Res<MaxSteps>,
    mut exit: MessageWriter<AppExit>,
) {
    if counter.0 >= max_steps.0 {
        return;
    }
    counter.0 += 1;
    for (name, state, _grav, _total) in &query {
        if name.as_str() != "Satellite" {
            continue;
        }
        if counter.0 == 1 || counter.0.is_multiple_of(100) {
            // `state.position.length()` returns a typed `Length`; `.value`
            // reads the SI base (meters). Same for velocity.
            let r_km: f64 = state.position.length().value / 1000.0;
            let v: f64 = state.velocity.length().value;
            println!(
                "step={:5}  t={:8.0}s  |r|={:8.1}km  |v|={:.1}m/s",
                counter.0,
                counter.0 as f64 * 10.0,
                r_km,
                v
            );
        }
        if counter.0 >= max_steps.0 {
            println!("Completed {} steps. Exiting.", max_steps.0);
            exit.write(AppExit::Success);
            return;
        }
    }
}

// `F64Ext` is brought in scope above so downstream calls like
// `1.5.ms()`, `42.0.kg()` work for mission code that wants typed
// constructions. The example itself uses recipe presets, but the
// import demonstrates the available surface — mission authors should
// consume units via this facade instead of importing `uom::si::*`
// directly. This function is documentation-only; never called.
#[allow(dead_code)]
fn _showcase_f64_ext() {
    let _mass = 420_000.0.kg();
    let _radius = 6_378_137.0.m();
    let _inclination = 51.6_f64.deg();
}
