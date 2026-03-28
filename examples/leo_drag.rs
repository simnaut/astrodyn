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
use jeod_math::geodetic::{cartesian_to_geodetic, cartesian_to_spherical};
use jeod_math::OrbitalElements;

const MU_EARTH: f64 = 3.986004418e14;
const R_EARTH_EQ: f64 = 6_378_137.0;
const R_EARTH_POL: f64 = 6_356_752.3142;

/// Compute GMST in radians (same formula as MET model / JEOD atmos_MET_TME.cc).
fn compute_gmst(tjt: f64) -> f64 {
    let tjt_prev_midnight = tjt.floor();
    let fraction_of_day = tjt - tjt_prev_midnight;
    let century_days = tjt_prev_midnight + 24980.0;
    let century_frac = (century_days + 0.5) / 36525.0;
    let minutes_of_day = fraction_of_day * 1440.0;
    let greenwich_mean_position =
        (99.6909833 + 36000.76892 * century_frac + 0.00038708 * century_frac * century_frac
            + 0.250684477 * minutes_of_day)
            .rem_euclid(360.0);
    greenwich_mean_position * 0.017453293
}

fn main() {
    // ISS-like orbit: 400 km circular, 51.6° inclination
    let altitude = 400_000.0; // m
    let r0 = R_EARTH_EQ + altitude;
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

            // Rotate inertial → planet-fixed via GMST, then geodetic coords.
            // Matches JEOD's PlanetFixedPosition → MET pipeline.
            let gmst = compute_gmst(tjt);
            let (cos_g, sin_g) = (gmst.cos(), gmst.sin());
            let pfix = DVec3::new(
                cos_g * s.position.x + sin_g * s.position.y,
                -sin_g * s.position.x + cos_g * s.position.y,
                s.position.z,
            );
            let geo = cartesian_to_geodetic(pfix, R_EARTH_EQ, R_EARTH_POL);
            let atmos_state = atmos.density(geo.altitude / 1000.0, geo.latitude, geo.longitude, tjt);
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
            let sph = cartesian_to_spherical(state.position, R_EARTH_EQ);
            let alt_km = sph.altitude / 1000.0;
            let elements = OrbitalElements::from_cartesian(MU_EARTH, state.position, state.velocity).unwrap();

            let gmst_p = compute_gmst(tjt);
            let (cg, sg) = (gmst_p.cos(), gmst_p.sin());
            let pfix_p = DVec3::new(
                cg * state.position.x + sg * state.position.y,
                -sg * state.position.x + cg * state.position.y,
                state.position.z,
            );
            let geo_p = cartesian_to_geodetic(pfix_p, R_EARTH_EQ, R_EARTH_POL);
            let atmos_state = atmos.density(geo_p.altitude / 1000.0, geo_p.latitude, geo_p.longitude, tjt);
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
