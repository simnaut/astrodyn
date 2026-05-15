//! Clementine lunar orbit: multi-body gravity with Earth, Moon, and Sun.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "example step counts (hours of propagation) fit exactly in f64 mantissa and usize"
)]
//!
//! Verification-style example exercising:
//! - Moon LP150Q 60×60 spherical-harmonics gravity
//! - Earth and Sun as 3rd-body point-mass perturbations
//! - DE421 ephemeris for per-step source position updates
//! - Moon libration rotation from DE421 BPC data
//! - Cannonball solar radiation pressure (no shadow body — illumination
//!   stays at 1.0; matches the original example's behaviour)
//!
//! Scenario construction lives in
//! [`astrodyn_verif_jeod::setups::earth_moon_clem`], shared with the
//! `tier3_simulation_earth_moon_clem` Tier 3 test and the
//! `tier3_perf_runner` binary so that all three callers stay in sync.
//!
//! ```bash
//! cargo run -p astrodyn_verif_jeod --example earth_moon
//! ```

#![forbid(unsafe_code)]

use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::setups::earth_moon_clem::{earth_moon_clem, moon_mu};

const DT: f64 = 1.0;
const DURATION: f64 = 86_400.0;
const R_MOON: f64 = 1_737_400.0;

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
    let mut sim = earth_moon_clem(DT, None)
        .build()
        .expect("earth_moon_clem scenario must validate");

    // Mu of the central Moon source — needed for the orbital-metrics
    // print loop below. Pulled from the same LP150Q fixture loader the
    // setup module uses, so the orbital bookkeeping stays consistent
    // with the integrator.
    let moon_mu = moon_mu();

    let print_interval = 7_200.0;
    let total_steps = parse_steps_arg((DURATION / DT).round() as usize);
    let steps_per_print = (print_interval / DT).round() as usize;
    // Derive the printed duration from `total_steps * DT` so headings stay
    // accurate when `--steps` overrides the nominal 1-day run length.
    let elapsed_days = total_steps as f64 * DT / 86_400.0;

    println!("Clementine lunar orbit — {elapsed_days:.2}-day propagation");
    println!("  Moon: LP150Q 60×60 SH | Earth + Sun 3rd-body | Cannonball SRP");
    println!("  Integrator: RK4 at dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Time (h)", "Alt (km)", "Speed (m/s)", "Period (h)", "Ecc"
    );
    println!("{}", "-".repeat(62));

    for step in 1..=total_steps {
        sim.step().expect("step failed");
        if step % steps_per_print == 0 {
            let body = sim.body(0);
            let pos = body.trans.position.raw_si();
            let vel = body.trans.velocity.raw_si();
            let r = pos.length();
            let v = vel.length();
            let altitude_km = (r - R_MOON) / 1000.0;
            let hours = (step as f64) * DT / 3600.0;
            let energy = 0.5 * v * v - moon_mu / r;
            let sma = -moon_mu / (2.0 * energy);
            let period_h = 2.0 * std::f64::consts::PI * (sma.powi(3) / moon_mu).sqrt() / 3600.0;
            let h_vec = pos.cross(vel);
            let ecc_vec = vel.cross(h_vec) / moon_mu - pos / r;
            let ecc = ecc_vec.length();
            println!("{hours:10.1}  {altitude_km:12.1}  {v:12.1}  {period_h:10.2}  {ecc:10.6}");
        }
    }

    println!();
    let final_body = sim.body(0);
    let final_alt = (final_body.trans.position.raw_si().length() - R_MOON) / 1000.0;
    println!("Final altitude: {final_alt:.1} km after {elapsed_days:.2} days");

    #[cfg(feature = "phase_timing")]
    {
        println!();
        print!("{}", sim.phase_timings_summary());
    }

    Ok(())
}
