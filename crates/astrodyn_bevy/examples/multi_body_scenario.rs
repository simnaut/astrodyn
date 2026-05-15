//! Multi-body scenario example — declarative composition via the
//! [`SimulationBuilderBevyExt::populate_app`] terminal.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "example step counts (hours of propagation) fit exactly in f64 mantissa and usize"
)]
#![allow(
    clippy::float_cmp,
    reason = "example assertions match literal-built state fields bit-exactly"
)]
//!
//! This is the "scenario in one call" pattern: a recipe assembles the
//! whole `SimulationBuilder` (sources, bodies, mass tree, ephemeris,
//! atmosphere, polar motion, integrator state), then `populate_app`
//! materializes the result into a Bevy [`App`] in a single step. Use
//! this path whenever the mission can be expressed as a recipe — it's
//! the canonical entry point for multi-body composition and it removes
//! every chance of forgetting a component on a hand-spawned entity.
//!
//! The complementary fine-grained surface is
//! [`astrodyn_bevy::VehicleConfigBevyExt::spawn_bevy`] (see
//! `examples/typed_mission.rs`): use that when the caller is composing
//! one vehicle at a time and doesn't want to assemble a
//! `SimulationBuilder` first.
//!
//! The scenario chosen here is
//! [`Mission::apollo_translunar`](astrodyn::recipes::Mission::apollo_translunar):
//! Earth as the central point-mass source, Moon and Sun as third
//! bodies, one CSM-class vehicle in a 200 km parking orbit. DE421
//! ephemeris is wired on the Moon and Sun source slots so their
//! positions update from the bundled JPL kernel each tick — this is
//! the multi-source story that
//! [`SimulationBuilderBevyExt::populate_app`] handles end-to-end.
//!
//! ```bash
//! cargo run -p astrodyn_bevy --example multi_body_scenario
//! ```

#![forbid(unsafe_code)]

use std::time::Duration;

use astrodyn::recipes::scenarios::apollo;
use astrodyn::recipes::{ephemeris as ephemeris_recipes, Mission};
use astrodyn::EphemerisBody;
use astrodyn_bevy::{AstrodynSet, IntegrationDtR, SimulationBuilderBevyExt, TranslationalStateC};
use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;

#[derive(Resource)]
struct StepCounter(usize);

#[derive(Resource)]
struct MaxSteps(usize);

#[derive(Resource)]
struct VehicleEntity(Entity);

#[derive(Resource)]
struct MoonEntity(Entity);

/// Default step count: a representative slice of the trans-lunar coast
/// at the recipe's 60 s timestep. Short enough to finish in seconds,
/// long enough to make the printed altitude / Moon-distance evolve
/// visibly.
const DEFAULT_STEPS: usize = 240;

/// Parse `--steps N` from CLI args; default to [`DEFAULT_STEPS`] when
/// absent. Mirrors the `kepler_orbit` / `typed_mission` examples — the
/// smoke test in CI passes `--steps 1` to validate the wiring without
/// burning cycles on a full propagation.
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

    // Compose the scenario through the Mission catalogue. This is one
    // line of declarative recipe composition — Earth point-mass +
    // Moon/Sun third bodies + 200 km parking-orbit CSM, all assembled
    // by the recipe.
    let mut sb = Mission::apollo_translunar().into_builder();

    // Layer DE421 ephemeris on the Moon / Sun source slots so their
    // positions advance from the bundled JPL kernel each tick. The
    // recipe exposes the source indices as named constants
    // (`apollo::EARTH_IDX` / `MOON_IDX` / `SUN_IDX`) so this wiring
    // doesn't depend on internal ordering of the recipe's
    // `add_source` calls.
    let ephemeris = ephemeris_recipes::de421().expect("DE421 ephemeris loads from embedded blob");
    sb.set_source_ephemeris(apollo::MOON_IDX, EphemerisBody::Moon, EphemerisBody::Earth);
    sb.set_source_ephemeris(apollo::SUN_IDX, EphemerisBody::Sun, EphemerisBody::Earth);
    sb = sb.ephemeris(ephemeris);

    // ── The canonical one-call materialization ──
    //
    // `populate_app` consumes the fully-composed builder, installs the
    // `AstrodynPlugin`, writes time/ephemeris resources, spawns one
    // entity per source and one per body, pre-allocates any mass tree,
    // auto-initializes integrator state, and returns
    // `ScenarioHandles` keyed parallel to the builder's `sources` /
    // `bodies` vecs.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(0))));
    let dt = sb.dt;
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.insert_resource(IntegrationDtR(dt));

    let handles = sb
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("populate_app: apollo_translunar materializes under <Earth>");

    // The handles' parallel-index contract is the load-bearing one for
    // mission code: index 0 in the recipe → index 0 in
    // `handles.body_entities`. Stash the vehicle entity so the
    // log_state system can read its translational state each tick.
    let vehicle_entity = handles.body_entities[0];
    let moon_entity = handles.source_entities[apollo::MOON_IDX];

    println!("Bevy multi-body scenario example");
    println!("================================");
    println!(
        "Sources: {} (Earth + Moon + Sun) | Bodies: {} (CSM)",
        handles.source_entities.len(),
        handles.body_entities.len(),
    );
    println!("Integration frame: PlanetInertial<Earth> | dt = {dt} s");
    println!("Steps: {max_steps}");
    println!();

    app.insert_resource(StepCounter(0))
        .insert_resource(MaxSteps(max_steps))
        .insert_resource(VehicleEntity(vehicle_entity))
        .insert_resource(MoonEntity(moon_entity))
        .add_systems(Startup, accelerate_virtual_time)
        .add_systems(FixedUpdate, log_state.after(AstrodynSet::Integration))
        .run();
}

/// Speed wall-clock time so the example finishes in seconds rather
/// than wall-real propagation time. Mirrors the `kepler_orbit` /
/// `typed_mission` examples — `ScheduleRunnerPlugin::run_loop`
/// advances `Time::<Virtual>` from the real clock, which then drives
/// `Time::<Fixed>` and the `FixedUpdate` schedule. Without the
/// relative-speed bump, a `dt = 60 s` propagation would take real
/// minutes per step.
fn accelerate_virtual_time(mut time: ResMut<Time<Virtual>>) {
    time.set_relative_speed_f64(1e6);
}

/// Per-step logger: prints the CSM's altitude above Earth and its
/// distance to the Moon at a coarse cadence. Reads
/// [`TranslationalStateC<Earth>`] off both the vehicle entity and the
/// Moon source entity — the same bridge that
/// [`SimulationBuilderBevyExt::populate_app`] populates also writes
/// `TranslationalStateC` on every gravity source so cross-source
/// queries (third-body distance, shadow geometry, …) read from
/// uniform component storage.
fn log_state(
    query: Query<&TranslationalStateC<astrodyn::Earth>>,
    vehicle: Res<VehicleEntity>,
    moon: Res<MoonEntity>,
    mut counter: ResMut<StepCounter>,
    max_steps: Res<MaxSteps>,
    mut exit: MessageWriter<AppExit>,
) {
    if counter.0 >= max_steps.0 {
        return;
    }
    counter.0 += 1;

    // Bail out before the first useful tick if the smoke-test path
    // (--steps 1) is in effect — the example exits cleanly without
    // having to fetch state.
    let header_step = counter.0 == 1;
    let cadence_step = counter.0.is_multiple_of(60);
    if header_step || cadence_step {
        let v_state = query
            .get(vehicle.0)
            .expect("vehicle carries TranslationalStateC<Earth> after populate_app");
        let m_state = query
            .get(moon.0)
            .expect("moon source carries TranslationalStateC<Earth> after populate_app");

        let earth_radius_km = 6_371.0_f64;
        let alt_km = v_state.position.length().value / 1000.0 - earth_radius_km;
        let moon_dist_km =
            (v_state.position.raw_si() - m_state.position.raw_si()).length() / 1000.0;
        let speed = v_state.velocity.length().value;
        println!(
            "step={:5}  t={:8.0}s  alt={:8.1}km  d_moon={:9.1}km  |v|={:.1}m/s",
            counter.0,
            counter.0 as f64 * 60.0,
            alt_km,
            moon_dist_km,
            speed,
        );
    }
    if counter.0 >= max_steps.0 {
        println!("Completed {} steps. Exiting.", max_steps.0);
        exit.write(AppExit::Success);
    }
}
