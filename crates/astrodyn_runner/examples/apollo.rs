//! Apollo trans-lunar injection: multi-body gravity, staging, impulsive maneuver.
//!
//! Demonstrates a recipe-aware pattern: starts from
//! [`Mission::apollo_translunar`](astrodyn::recipes::Mission::apollo_translunar)
//! for the simulation shape (Earth + Moon + Sun point-mass gravity),
//! then layers mission-specific maneuvers — TLI impulsive Δv, stage
//! separation via mass tree, parking-orbit override — directly on the
//! `Simulation`. This is the "use a scenario as the starting point,
//! then customize" archetype recipes are designed to support.
//!
//! Uses DE421 ephemeris from `test_data/de421.bsp` for Moon and Sun
//! positions.
//!
//! ```bash
//! cargo run -p astrodyn_runner --example apollo
//! ```

use std::path::Path;

use astrodyn::recipes::scenarios::apollo;
use astrodyn::recipes::Mission;
use astrodyn::{EphemerisBody, TranslationalState};
use astrodyn_dynamics::MassProperties;
use astrodyn_runner::SimulationBuilderExt;
use glam::{DMat3, DVec3};

const MU_EARTH: f64 = 3.986_004_418e14;
const R_EARTH: f64 = 6_371_000.0;

const PARKING_ALT: f64 = 185_000.0;
const INCLINATION_DEG: f64 = 28.5;

const MASS_CSM: f64 = 28_800.0;
const MASS_SIVB: f64 = 13_300.0;

const TLI_DELTA_V: f64 = 3_130.0;
const PARKING_ORBITS: f64 = 2.5;
const DT: f64 = 60.0;
const TOTAL_DURATION: f64 = 3.0 * 86_400.0;

/// Parse `--steps N` from CLI args; fall back to `default` when absent.
/// Panics with a clear message on a malformed value (per fail-loudly policy).
fn parse_steps_arg(default: usize) -> usize {
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
    default
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Override scenario state with Apollo parking orbit, attach DE421
    // ephemeris, then build.
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let bsp_path = data_dir.join("de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    let ephemeris = astrodyn::Ephemeris::from_bsp(&bsp_path)?;

    let r_park = R_EARTH + PARKING_ALT;
    let v_circ = (MU_EARTH / r_park).sqrt();
    let inc = INCLINATION_DEG.to_radians();
    let init_pos = DVec3::new(r_park, 0.0, 0.0);
    let init_vel = DVec3::new(0.0, v_circ * inc.cos(), v_circ * inc.sin());

    let mut sb = Mission::apollo_translunar().into_builder();
    sb.dt = DT;
    // Override the scenario's default vehicle state with the parking-orbit IC.
    sb.bodies[0].trans = TranslationalState {
        position: init_pos,
        velocity: init_vel,
    };
    // Initialize the body with CSM-only mass; the S-IVB is added separately
    // below as a child in the mass tree, and `add_body_to_tree` will snapshot
    // the body's current mass into the tree node. Initializing with
    // CSM+S-IVB here would double-count the S-IVB once it's attached.
    sb.bodies[0].mass = Some(MassProperties::new(MASS_CSM));
    // Wire DE421 ephemeris on the Moon/Sun sources (indices exposed by
    // the scenario as named constants — robust against any future
    // reordering inside `apollo_translunar`).
    sb.set_source_ephemeris(apollo::MOON_IDX, EphemerisBody::Moon, EphemerisBody::Earth);
    sb.set_source_ephemeris(apollo::SUN_IDX, EphemerisBody::Sun, EphemerisBody::Earth);
    sb = sb.ephemeris(ephemeris);

    let mut sim = sb.build().expect("apollo scenario must validate");

    // Mass tree: CSM (root) + S-IVB child for stage separation.
    let csm_tree_id = sim.add_body_to_tree(0, "CSM");
    let tree = sim.mass_tree.as_mut().unwrap();
    let sivb_tree_id = tree.add_body("S-IVB".to_string(), MassProperties::new(MASS_SIVB));
    tree.attach(sivb_tree_id, csm_tree_id, DVec3::ZERO, DMat3::IDENTITY);
    sim.sync_body_mass_from_tree(0);

    let parking_period = 2.0 * std::f64::consts::PI * (r_park.powi(3) / MU_EARTH).sqrt();
    let tli_time = PARKING_ORBITS * parking_period;
    let separation_time = tli_time + 600.0;

    let total_steps = parse_steps_arg((TOTAL_DURATION / DT).round() as usize);
    let print_interval = 7_200.0;
    let steps_per_print = (print_interval / DT).round() as usize;
    // Derive the printed duration from `total_steps * DT` so headings/summary
    // stay accurate when `--steps` overrides the nominal 3-day run length.
    let elapsed_days = total_steps as f64 * DT / 86_400.0;

    println!("Apollo trans-lunar injection — {elapsed_days:.2}-day trajectory");
    println!(
        "  Parking orbit: {:.0} km circular, {INCLINATION_DEG}° inclination",
        PARKING_ALT / 1000.0
    );
    println!("  Vehicle: CSM ({MASS_CSM} kg) + S-IVB ({MASS_SIVB} kg)");
    println!(
        "  TLI Δv: {TLI_DELTA_V} m/s prograde at t={:.1} h",
        tli_time / 3600.0
    );
    println!("  Earth + Moon + Sun point-mass gravity | dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}  {:>8}",
        "Time (h)", "Alt (km)", "Dist Moon", "Speed (m/s)", "Mass (kg)", "Phase"
    );
    println!("{}", "-".repeat(76));
    let mut tli_applied = false;
    let mut separated = false;

    for step in 1..=total_steps {
        let t = (step as f64) * DT;
        if !tli_applied && t >= tli_time {
            let body = sim.body(0);
            let vel_hat = body.trans.velocity.normalize();
            sim.set_body_velocity(0, body.trans.velocity + vel_hat * TLI_DELTA_V);
            tli_applied = true;
            println!(
                "{:10.1}  {:>12}  {:>12}  {:>12}  {:>10}  TLI BURN",
                t / 3600.0,
                "---",
                "---",
                "---",
                "---"
            );
        }
        if !separated && t >= separation_time {
            let tree = sim.mass_tree.as_mut().unwrap();
            tree.detach(sivb_tree_id);
            sim.sync_body_mass_from_tree(0);
            separated = true;
            println!(
                "{:10.1}  {:>12}  {:>12}  {:>12}  {:>10}  SEPARATE",
                t / 3600.0,
                "---",
                "---",
                "---",
                "---"
            );
        }
        sim.step().expect("step failed");
        if step % steps_per_print == 0 {
            let body = sim.body(0);
            let pos = body.trans.position;
            let vel = body.trans.velocity;
            let moon_pos = sim.source_position(apollo::MOON_IDX);
            let alt_km = (pos.length() - R_EARTH) / 1000.0;
            let dist_moon_km = (pos - moon_pos).length() / 1000.0;
            let speed = vel.length();
            let mass = if separated {
                MASS_CSM
            } else {
                MASS_CSM + MASS_SIVB
            };
            let hours = t / 3600.0;
            let phase = if !tli_applied {
                "PARK"
            } else if dist_moon_km < 50_000.0 {
                "LUNAR"
            } else {
                "COAST"
            };
            println!(
                "{hours:10.1}  {alt_km:12.1}  {dist_moon_km:12.1}  {speed:12.1}  {mass:10.0}  {phase}"
            );
        }
    }

    let final_body = sim.body(0);
    let final_moon_dist =
        (final_body.trans.position - sim.source_position(apollo::MOON_IDX)).length();
    println!();
    println!(
        "Final distance to Moon: {:.0} km after {elapsed_days:.2} days",
        final_moon_dist / 1000.0,
    );
    Ok(())
}
