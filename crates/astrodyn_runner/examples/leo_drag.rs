//! LEO orbit with atmospheric drag using the recipes module.
//!
//! Propagates an ISS-like orbit (400 km, i=51.6 deg) with the MET
//! atmosphere model and shows altitude decay over 24 hours. Uses
//! [`Mission::iss_leo_drag`](astrodyn::recipes::Mission::iss_leo_drag).

use astrodyn::recipes::{constants, Mission};
use astrodyn_runner::SimulationBuilderExt;
use glam::DVec3;

fn specific_energy(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    0.5 * velocity.length_squared() - mu / position.length()
}

fn semi_major_axis(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    -mu / (2.0 * specific_energy(mu, position, velocity))
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
    let r_eq = 6_378_137.0;

    let mut sim = Mission::iss_leo_drag()
        .into_builder()
        .build()
        .expect("iss_leo_drag must validate");
    let body_idx = 0;

    let initial = sim.body(body_idx).trans;
    let initial_pos = initial.position.raw_si();
    let initial_vel = initial.velocity.raw_si();
    let initial_e = eccentricity(mu_earth, initial_pos, initial_vel);
    let initial_a = semi_major_axis(mu_earth, initial_pos, initial_vel);

    println!("=== LEO Orbit with Atmospheric Drag (MET Jacchia 1971) ===");
    println!(
        "Initial: alt={:.1} km, e={:.6}, a={:.1} km",
        (initial_pos.length() - r_eq) / 1000.0,
        initial_e,
        initial_a / 1000.0,
    );
    println!("Atmosphere: MET solar mean (recipes::atmosphere::met_solar_mean())");
    println!();
    println!(
        "{:>8}  {:>10}  {:>12}  {:>10}",
        "Time(h)", "Alt(km)", "a(km)", "e"
    );
    println!("{}", "-".repeat(48));

    let dt = sim.dt;
    let total_time = 86_400.0;
    let steps = parse_steps_arg((total_time / dt) as usize);
    // Guard against `steps < 24` (e.g. CI smoke `--steps 10`) producing a
    // zero divisor in `step % print_interval`.
    let print_interval = (steps / 24).max(1);

    for step in 0..steps {
        sim.step().expect("step failed");
        let sim_time = (step + 1) as f64 * dt;
        if (step + 1) % print_interval == 0 {
            let time_h = sim_time / 3600.0;
            let s = sim.body(body_idx).trans;
            let pos = s.position.raw_si();
            let vel = s.velocity.raw_si();
            let alt_km = (pos.length() - r_eq) / 1000.0;
            let e_mag = eccentricity(mu_earth, pos, vel);
            let a_km = semi_major_axis(mu_earth, pos, vel) / 1000.0;
            println!(
                "{:>8.1}  {:>10.3}  {:>12.3}  {:>10.6}",
                time_h, alt_km, a_km, e_mag,
            );
        }
    }

    let final_state = sim.body(body_idx).trans;
    let final_pos = final_state.position.raw_si();
    let final_vel = final_state.velocity.raw_si();
    let final_a = semi_major_axis(mu_earth, final_pos, final_vel);
    let final_e = eccentricity(mu_earth, final_pos, final_vel);
    let sma_decay = initial_a - final_a;

    // Derive elapsed time from `steps * dt` so the summary stays accurate
    // when `--steps` overrides the nominal 24-hour run length.
    let elapsed_h = steps as f64 * dt / 3600.0;

    println!();
    println!("Final: a={:.3} km, e={:.6}", final_a / 1000.0, final_e);
    println!("SMA decay: {sma_decay:.1} m over {elapsed_h:.2}h");
}
