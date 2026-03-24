//! Standalone batch Kepler orbit propagation using jeod_* crates (no Bevy).
//!
//! Propagates a circular LEO orbit for 10 periods using RK4, printing orbital
//! elements and energy drift at regular intervals. Validates that RK4 energy
//! conservation is within expected bounds.

use jeod_dynamics::{rk4_translational_step, TranslationalState};
use jeod_gravity::calc_spherical;
use jeod_math::{DVec3, OrbitalElements};

fn main() {
    let mu_earth: f64 = 3.986004418e14; // m^3/s^2
    let r0: f64 = 6_778_137.0; // m (400 km altitude)
    let v0 = (mu_earth / r0).sqrt(); // circular velocity

    let mut state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let dt = 10.0; // seconds
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu_earth).sqrt();
    let n_orbits = 10;
    let steps = (n_orbits as f64 * period / dt).ceil() as usize;

    let initial_energy =
        0.5 * state.velocity.length_squared() - mu_earth / state.position.length();

    println!("Batch Kepler Orbit Propagation (no Bevy)");
    println!("=========================================");
    println!("Initial altitude: {:.1} km", (r0 - 6_378_137.0) / 1000.0);
    println!("Orbital period:   {:.1} s", period);
    println!("Timestep:         {:.1} s", dt);
    println!(
        "Propagating for:  {} orbits ({} steps)",
        n_orbits, steps
    );
    println!();

    let report_interval = (steps / 10).max(1);

    for step in 0..=steps {
        if step > 0 {
            state = rk4_translational_step(
                &state,
                |s| calc_spherical(mu_earth, s.position).grav_accel,
                dt,
            );
        }

        if step % report_interval == 0 || step == steps {
            let energy =
                0.5 * state.velocity.length_squared() - mu_earth / state.position.length();
            let energy_drift = energy - initial_energy;
            let alt_km = (state.position.length() - 6_378_137.0) / 1000.0;

            match OrbitalElements::from_cartesian(mu_earth, state.position, state.velocity) {
                Ok(elems) => {
                    println!(
                        "t={:8.0}s  alt={:7.1}km  e={:.2e}  energy_drift={:.2e} J/kg",
                        step as f64 * dt,
                        alt_km,
                        elems.e_mag,
                        energy_drift
                    );
                }
                Err(e) => println!("Error computing elements: {}", e),
            }
        }
    }

    let final_energy =
        0.5 * state.velocity.length_squared() - mu_earth / state.position.length();
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
