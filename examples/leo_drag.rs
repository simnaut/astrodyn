//! LEO orbit with atmospheric drag — Phase 4 example.
//!
//! Propagates an ISS-like orbit (400 km, i=51.6°) with the MET (Jacchia 1971)
//! atmosphere model and shows altitude decay over 24 hours.
//!
//! Uses only `jeod_*` crates (no Bevy dependency) to demonstrate portability
//! of the new Phase 4 interaction physics.
//!
//! ```bash
//! cargo run --example leo_drag
//! ```

use glam::{DMat3, DVec3};
use jeod_atmosphere::met;
use jeod_dynamics::{rk4_translational_step, TranslationalState};
use jeod_interactions::{compute_ballistic_drag, DragConfig};
use jeod_math::geodetic::cartesian_to_spherical;
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

    // MET atmosphere (Jacchia 1971) at solar mean conditions
    let atmos = met::SOLAR_MEAN;

    // TJT for J2000 epoch (2000-01-01 12:00 TAI)
    let tjt_start = jeod_time::epoch::J2000_TAI_TJT;

    let dt = 60.0; // 1-minute steps
    let total_time = 86400.0; // 24 hours
    let steps = (total_time / dt) as usize;
    let print_interval = steps / 24; // Print once per hour

    let initial_elements = OrbitalElements::from_cartesian(MU_EARTH, state.position, state.velocity).unwrap();
    println!("=== LEO Orbit with Atmospheric Drag (MET Jacchia 1971) ===");
    println!("Initial: alt={:.1} km, e={:.6}, a={:.1} km",
        altitude / 1000.0,
        initial_elements.e_mag,
        initial_elements.semi_major_axis / 1000.0,
    );
    println!("Atmosphere: MET solar mean (F10.7={}, F10B={})", atmos.f10, atmos.f10b);
    println!();
    println!("{:>8}  {:>10}  {:>12}  {:>10}  {:>14}  {:>12}",
        "Time(h)", "Alt(km)", "a(km)", "e", "Density(kg/m3)", "DragF(mN)");
    println!("{}", "-".repeat(78));

    for step in 0..steps {
        let sim_time = (step + 1) as f64 * dt;
        let tjt = tjt_start + sim_time / 86400.0;

        let new_state = rk4_translational_step(&state, |s| {
            // Point-mass gravity
            let r = s.position.length();
            let grav = -MU_EARTH / (r * r * r) * s.position;

            // Spherical coordinates for atmosphere (no RNP in this example)
            let sph = cartesian_to_spherical(s.position, R_EARTH);
            let alt_km = sph.altitude / 1000.0;

            // MET atmosphere density
            let atmos_state = atmos.density(alt_km, sph.latitude, sph.longitude, tjt);
            let drag = compute_ballistic_drag(
                &drag_config,
                &atmos_state,
                s.velocity,
                &DMat3::IDENTITY,
            );

            grav + drag.force / mass
        }, dt);
        state = new_state;

        if (step + 1) % print_interval == 0 {
            let time_h = sim_time / 3600.0;
            let sph = cartesian_to_spherical(state.position, R_EARTH);
            let alt_km = sph.altitude / 1000.0;
            let elements = OrbitalElements::from_cartesian(MU_EARTH, state.position, state.velocity).unwrap();

            let atmos_state = atmos.density(alt_km, sph.latitude, sph.longitude, tjt);
            let drag = compute_ballistic_drag(
                &drag_config,
                &atmos_state,
                state.velocity,
                &DMat3::IDENTITY,
            );

            println!("{:>8.1}  {:>10.3}  {:>12.3}  {:>10.6}  {:>14.6e}  {:>12.6}",
                time_h,
                alt_km,
                elements.semi_major_axis / 1000.0,
                elements.e_mag,
                atmos_state.density,
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
