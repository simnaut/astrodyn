//! LEO orbit with atmospheric drag — Phase 4 example.
//!
//! Propagates an ISS-like orbit (400 km, i=51.6°) with exponential atmosphere
//! drag and shows altitude decay over 24 hours.
//!
//! Uses only `jeod_*` crates (no Bevy dependency) to demonstrate portability
//! of the new Phase 4 interaction physics.
//!
//! ```bash
//! cargo run --example leo_drag
//! ```

use glam::{DMat3, DVec3};
use jeod_atmosphere::exponential::ExponentialAtmosphere;
use jeod_dynamics::{rk4_translational_step, TranslationalState};
use jeod_interactions::{compute_ballistic_drag, DragConfig};
use jeod_math::OrbitalElements;

const MU_EARTH: f64 = 3.986004418e14;
const R_EARTH: f64 = 6_378_137.0;

fn main() {
    // ISS-like orbit: 400 km circular, 51.6° inclination
    let altitude = 400_000.0; // m
    let r0 = R_EARTH + altitude;
    let v0 = (MU_EARTH / r0).sqrt();
    let inc = 51.6_f64.to_radians();

    let mut state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0 * inc.cos(), v0 * inc.sin()),
    };

    // ISS-like drag properties
    let drag_config = DragConfig {
        cd: 2.2,
        area: 1900.0, // m^2 (cross-sectional area)
    };
    let mass = 420_000.0; // kg
    let atmos = ExponentialAtmosphere::default();

    let dt = 60.0; // 1-minute steps
    let total_time = 86400.0; // 24 hours
    let steps = (total_time / dt) as usize;
    let print_interval = steps / 24; // Print once per hour

    let initial_elements = OrbitalElements::from_cartesian(MU_EARTH, state.position, state.velocity).unwrap();
    println!("=== LEO Orbit with Atmospheric Drag ===");
    println!("Initial: alt={:.1} km, e={:.6}, a={:.1} km",
        altitude / 1000.0,
        initial_elements.e_mag,
        initial_elements.semi_major_axis / 1000.0,
    );
    println!();
    println!("{:>8}  {:>10}  {:>12}  {:>10}  {:>12}",
        "Time(h)", "Alt(km)", "a(km)", "e", "DragF(mN)");
    println!("{}", "-".repeat(60));

    for step in 0..steps {
        let new_state = rk4_translational_step(&state, |s| {
            // Point-mass gravity
            let r = s.position.length();
            let grav = -MU_EARTH / (r * r * r) * s.position;

            // Atmospheric drag
            let alt = s.position.length() - R_EARTH;
            let atmos_state = atmos.density(alt);
            let drag = compute_ballistic_drag(
                &drag_config,
                &atmos_state,
                s.velocity,
                &DMat3::IDENTITY, // ballistic model: frame doesn't matter
            );

            grav + drag.force / mass
        }, dt);
        state = new_state;

        if (step + 1) % print_interval == 0 {
            let time_h = (step + 1) as f64 * dt / 3600.0;
            let alt_km = (state.position.length() - R_EARTH) / 1000.0;
            let elements = OrbitalElements::from_cartesian(MU_EARTH, state.position, state.velocity).unwrap();

            // Current drag force magnitude
            let alt = state.position.length() - R_EARTH;
            let atmos_state = atmos.density(alt);
            let drag = compute_ballistic_drag(
                &drag_config,
                &atmos_state,
                state.velocity,
                &DMat3::IDENTITY,
            );

            println!("{:>8.1}  {:>10.3}  {:>12.3}  {:>10.6}  {:>12.6}",
                time_h,
                alt_km,
                elements.semi_major_axis / 1000.0,
                elements.e_mag,
                drag.force.length() * 1000.0, // mN
            );
        }
    }

    let final_elements = OrbitalElements::from_cartesian(MU_EARTH, state.position, state.velocity).unwrap();
    let sma_decay = initial_elements.semi_major_axis - final_elements.semi_major_axis;

    println!();
    println!("Final: a={:.3} km, e={:.6}", final_elements.semi_major_axis / 1000.0, final_elements.e_mag);
    println!("SMA decay: {:.1} m over 24h", sma_decay);
}
