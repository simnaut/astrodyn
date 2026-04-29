//! Dawn spacecraft at Mars: high-fidelity spherical harmonics gravity.
//!
//! Verification-style example exercising:
//! - Mars MRO110B2 110×110 spherical harmonics gravity (loaded from
//!   `$JEOD_HOME` via [`recipes::verification::reference_data`])
//! - Mars IAU rotation model
//! - Sun as 3rd-body perturbation with DE421 ephemeris
//!   (`test_data/de421.bsp`)
//!
//! Mission code that wants high-fidelity gravity / ephemeris without a
//! JEOD checkout is tracked in #144; until that lands the example is
//! verification-grade by necessity.
//!
//! ```bash
//! cargo run -p jeod_runner --example mars_orbit
//! ```

use std::path::Path;

use glam::DVec3;
use jeod_runner::SimulationBuilderExt;
use jeod_sim::recipes::{epoch, sun, vehicle, verification::reference_data};
use jeod_sim::vehicle_builder::VehicleBuilder;
use jeod_sim::{EphemerisBody, GravityControl, SimulationBuilder, TranslationalState};

// Dawn spacecraft initial state at Mars (from JEOD SIM_Mars RUN_dawn, t=0).
const INIT_POS: DVec3 = DVec3::new(11_563_355.680_2, -14_356_668.897_7, 6_293_704.616_9);
const INIT_VEL: DVec3 = DVec3::new(-2_273.107_8, 2_380.132_4, -22.911);

const DT: f64 = 10.0;
const DURATION: f64 = 10_800.0;
const R_MARS_EQ: f64 = 3_396_190.0;

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
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let bsp_path = data_dir.join("de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );

    let time = epoch::dawn_mars_2009();
    let epoch_tdb_jd = time.tdb_julian_date();

    let ephemeris = jeod_sim::Ephemeris::from_bsp(&bsp_path)?;
    let (sun_pos_typed, _) =
        ephemeris.get_state_typed(EphemerisBody::Sun, EphemerisBody::Mars, epoch_tdb_jd)?;
    let sun_pos = sun_pos_typed.raw_si();

    // Mars central body with MRO110B2 SH gravity (verification reference data).
    let mars_source = reference_data::mars_mro110b2();
    let mars_mu = mars_source.source.mu;

    let mut sb = SimulationBuilder::new(time, DT);
    let mars = sb.add_source("Mars", mars_source);
    let sun_idx = sb.add_source("Sun", sun::third_body(sun_pos));
    sb.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Mars);
    sb = sb.ephemeris(ephemeris).sun(sun_idx);

    let trans = TranslationalState {
        position: INIT_POS,
        velocity: INIT_VEL,
    };
    let dawn = VehicleBuilder::new()
        .with_state(trans)
        .three_dof_point_mass(vehicle::dawn_mass())
        .rk4()
        .gravity(GravityControl::new_nonspherical(mars, 110, 110, false))
        .gravity(GravityControl::new_third_body(sun_idx))
        .build();
    sb.add_body(dawn);

    let mut sim = sb.build().expect("mars_orbit scenario must validate");

    println!("Dawn spacecraft at Mars — 3-hour propagation");
    println!("  Mars: MRO110B2 110×110 SH | IAU rotation | Sun 3rd-body");
    println!("  Integrator: RK4 at dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "Time (min)", "Alt (km)", "Speed (m/s)", "SMA (km)", "Ecc"
    );
    println!("{}", "-".repeat(62));

    let print_interval = 900.0;
    let total_steps = parse_steps_arg((DURATION / DT).round() as usize);
    let steps_per_print = (print_interval / DT).round() as usize;

    for step in 1..=total_steps {
        sim.step().expect("step failed");
        if step % steps_per_print == 0 {
            let body = sim.body(0);
            let pos = body.trans.position;
            let vel = body.trans.velocity;
            let r = pos.length();
            let v = vel.length();
            let altitude_km = (r - R_MARS_EQ) / 1000.0;
            let minutes = (step as f64) * DT / 60.0;
            let energy = 0.5 * v * v - mars_mu / r;
            let sma_km = -mars_mu / (2.0 * energy) / 1000.0;
            let h_vec = pos.cross(vel);
            let ecc_vec = vel.cross(h_vec) / mars_mu - pos / r;
            let ecc = ecc_vec.length();
            println!("{minutes:10.1}  {altitude_km:12.1}  {v:12.1}  {sma_km:12.1}  {ecc:10.6}");
        }
    }

    println!();
    let final_body = sim.body(0);
    let final_alt = (final_body.trans.position.length() - R_MARS_EQ) / 1000.0;
    // Derive elapsed time from `total_steps * DT` so the summary stays
    // accurate when `--steps` overrides the nominal duration.
    let elapsed_min = total_steps as f64 * DT / 60.0;
    println!("Final altitude: {final_alt:.1} km after {elapsed_min:.1} minutes");
    Ok(())
}
