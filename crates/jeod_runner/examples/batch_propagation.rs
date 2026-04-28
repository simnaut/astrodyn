//! Standalone batch Kepler propagation using the recipes module.
//!
//! Propagates a circular LEO orbit for 10 periods, printing eccentricity
//! and energy drift at regular intervals. Uses
//! [`Mission::iss_leo`](jeod_sim::recipes::Mission::iss_leo) as the
//! starting scenario.

use glam::DVec3;
use jeod_runner::SimulationBuilderExt;
use jeod_sim::recipes::{constants, Mission};

fn specific_energy(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    0.5 * velocity.length_squared() - mu / position.length()
}

fn eccentricity(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    let h = position.cross(velocity);
    let e_vec = velocity.cross(h) / mu - position.normalize();
    e_vec.length()
}

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

fn main() {
    let mu_earth = constants::mu_ggm05c().value;

    // 10 orbits of an ISS-altitude circular LEO. The recipe defaults to
    // a 60 s step (good enough for verification accuracy); override to
    // 10 s here so the energy-drift demo passes its <1e-8 threshold.
    let mut sb = Mission::iss_leo().into_builder();
    sb.dt = 10.0;
    let mut sim = sb.build().expect("iss_leo must validate");

    let r0 = sim.body(0).trans.position.length();
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu_earth).sqrt();
    let nominal_orbits = 10;
    let dt = sim.dt;
    let steps = parse_steps_arg((nominal_orbits as f64 * period / dt).ceil() as usize);
    // Derive the actual orbit count from `steps * dt / period` so the heading
    // stays accurate when `--steps` overrides the nominal 10-orbit length.
    let actual_orbits = steps as f64 * dt / period;

    let initial = sim.body(0).trans;
    let initial_energy = specific_energy(mu_earth, initial.position, initial.velocity);

    println!("Batch Kepler Orbit Propagation (no Bevy)");
    println!("=========================================");
    println!("Initial altitude: {:.1} km", (r0 - 6_378_137.0) / 1000.0);
    println!("Orbital period:   {period:.1} s");
    println!("Timestep:         {dt:.1} s");
    println!("Propagating for:  {actual_orbits:.2} orbits ({steps} steps)");
    println!();

    let report_interval = (steps / 10).max(1);

    for step in 0..=steps {
        if step > 0 {
            sim.step().expect("step failed");
        }
        if step % report_interval == 0 || step == steps {
            let s = sim.body(0).trans;
            let energy = specific_energy(mu_earth, s.position, s.velocity);
            let drift = energy - initial_energy;
            let alt_km = (s.position.length() - 6_378_137.0) / 1000.0;
            let e_mag = eccentricity(mu_earth, s.position, s.velocity);
            println!(
                "t={:8.0}s  alt={:7.1}km  e={:.2e}  energy_drift={:.2e} J/kg",
                step as f64 * dt,
                alt_km,
                e_mag,
                drift
            );
        }
    }

    let final_state = sim.body(0).trans;
    let final_energy = specific_energy(mu_earth, final_state.position, final_state.velocity);
    let drift = (final_energy - initial_energy).abs();
    let relative_drift = drift / final_energy.abs();
    println!();
    println!("Final energy drift: {:.2e} J/kg", drift);
    println!("Relative energy drift: {:.2e}", relative_drift);
    if relative_drift < 1e-8 {
        println!("PASS: Energy conservation excellent (relative drift < 1e-8)");
    } else {
        println!("WARN: Relative energy drift exceeds 1e-8");
    }
}
