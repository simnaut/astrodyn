//! Standalone batch Kepler propagation using `jeod_sim` (no Bevy).
//!
//! Propagates a circular LEO orbit for 10 periods with the `Simulation`
//! runner, printing eccentricity and energy drift at regular intervals.

use glam::DVec3;
use jeod_sim::{
    default_leap_second_table, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, SimBody, Simulation, SimulationTime, TranslationalState,
};

fn specific_energy(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    0.5 * velocity.length_squared() - mu / position.length()
}

fn eccentricity(mu: f64, position: DVec3, velocity: DVec3) -> f64 {
    let h = position.cross(velocity);
    let e_vec = velocity.cross(h) / mu - position.normalize();
    e_vec.length()
}

fn main() {
    let mu_earth: f64 = 3.986004418e14; // m^3/s^2
    let r0: f64 = 6_778_137.0; // m (400 km altitude)
    let v0 = (mu_earth / r0).sqrt(); // circular velocity

    let state0 = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let dt = 10.0; // seconds
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu_earth).sqrt();
    let n_orbits = 10;
    let steps = (n_orbits as f64 * period / dt).ceil() as usize;

    let initial_energy = specific_energy(mu_earth, state0.position, state0.velocity);

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mu_earth,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });
    let body_idx = sim.add_body(SimBody {
        trans: state0,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().expect("valid batch propagation setup");

    println!("Batch Kepler Orbit Propagation (no Bevy)");
    println!("=========================================");
    println!("Initial altitude: {:.1} km", (r0 - 6_378_137.0) / 1000.0);
    println!("Orbital period:   {:.1} s", period);
    println!("Timestep:         {:.1} s", dt);
    println!("Propagating for:  {} orbits ({} steps)", n_orbits, steps);
    println!();

    let report_interval = (steps / 10).max(1);

    for step in 0..=steps {
        if step > 0 {
            sim.step();
        }

        if step % report_interval == 0 || step == steps {
            let state = sim.body(body_idx).trans;
            let energy = specific_energy(mu_earth, state.position, state.velocity);
            let energy_drift = energy - initial_energy;
            let alt_km = (state.position.length() - 6_378_137.0) / 1000.0;
            let e_mag = eccentricity(mu_earth, state.position, state.velocity);

            println!(
                "t={:8.0}s  alt={:7.1}km  e={:.2e}  energy_drift={:.2e} J/kg",
                step as f64 * dt,
                alt_km,
                e_mag,
                energy_drift
            );
        }
    }

    let final_state = sim.body(body_idx).trans;
    let final_energy = specific_energy(mu_earth, final_state.position, final_state.velocity);
    let drift = (final_energy - initial_energy).abs();
    println!();
    println!("Final energy drift: {:.2e} J/kg", drift);
    let specific_energy = final_energy.abs();
    let relative_drift = drift / specific_energy;
    println!("Relative energy drift: {:.2e}", relative_drift);
    if relative_drift < 1e-8 {
        println!("PASS: Energy conservation excellent (relative drift < 1e-8)");
    } else {
        println!("WARN: Relative energy drift exceeds 1e-8");
    }
}
