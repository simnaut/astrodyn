//! LEO orbit with atmospheric drag using the recipes module.
//!
//! Propagates an ISS-like orbit (400 km, i=51.6 deg) with the MET
//! atmosphere model and shows altitude decay over 24 hours. Uses
//! [`scenarios::iss_leo_drag`](jeod_sim::recipes::scenarios::iss_leo_drag).

use glam::DVec3;
use jeod_runner::SimulationBuilderExt;
use jeod_sim::recipes::{constants, scenarios};

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

fn main() {
    let mu_earth = constants::mu_ggm05c().value;
    let r_eq = 6_378_137.0;

    let mut sim = scenarios::iss_leo_drag()
        .build()
        .expect("iss_leo_drag() must validate");
    let body_idx = 0;

    let initial = sim.body(body_idx).trans;
    let initial_e = eccentricity(mu_earth, initial.position, initial.velocity);
    let initial_a = semi_major_axis(mu_earth, initial.position, initial.velocity);

    println!("=== LEO Orbit with Atmospheric Drag (MET Jacchia 1971) ===");
    println!(
        "Initial: alt={:.1} km, e={:.6}, a={:.1} km",
        (initial.position.length() - r_eq) / 1000.0,
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
    let steps = (total_time / dt) as usize;
    let print_interval = steps / 24;

    for step in 0..steps {
        sim.step();
        let sim_time = (step + 1) as f64 * dt;
        if (step + 1) % print_interval == 0 {
            let time_h = sim_time / 3600.0;
            let s = sim.body(body_idx).trans;
            let alt_km = (s.position.length() - r_eq) / 1000.0;
            let e_mag = eccentricity(mu_earth, s.position, s.velocity);
            let a_km = semi_major_axis(mu_earth, s.position, s.velocity) / 1000.0;
            println!(
                "{:>8.1}  {:>10.3}  {:>12.3}  {:>10.6}",
                time_h, alt_km, a_km, e_mag,
            );
        }
    }

    let final_state = sim.body(body_idx).trans;
    let final_a = semi_major_axis(mu_earth, final_state.position, final_state.velocity);
    let final_e = eccentricity(mu_earth, final_state.position, final_state.velocity);
    let sma_decay = initial_a - final_a;

    println!();
    println!("Final: a={:.3} km, e={:.6}", final_a / 1000.0, final_e);
    println!("SMA decay: {:.1} m over 24h", sma_decay);
}
