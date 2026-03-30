//! LEO orbit with atmospheric drag using `jeod_sim` (no Bevy).
//!
//! Propagates an ISS-like orbit (400 km, i=51.6 deg) with the MET atmosphere
//! model and shows altitude decay over 24 hours.
//!
//! ```bash
//! cargo run --example leo_drag
//! ```

use glam::{DMat3, DVec3};
use jeod_sim::{
    default_leap_second_table, met_atmosphere, AtmosphereConfig, AtmosphereModel, DragConfig,
    DynamicsConfig, GravityAcceleration, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, MassProperties, SimBody, Simulation, SimulationTime,
    TranslationalState,
};

const MU_EARTH: f64 = 3.986004418e14;
const R_EARTH_EQ: f64 = 6_378_137.0;
const R_EARTH_POL: f64 = 6_356_752.3142;
/// Earth angular velocity (rad/s), from JEOD RNPJ2000 data.
const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5;

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
    // ISS-like orbit: 400 km circular, 51.6° inclination
    let altitude = 400_000.0; // m
    let r0 = R_EARTH_EQ + altitude;
    let v0 = (MU_EARTH / r0).sqrt();
    let inc = 51.6_f64.to_radians();

    let state0 = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0 * inc.cos(), v0 * inc.sin()),
    };

    // ISS-like drag properties
    let drag_config = DragConfig {
        cd: 2.2,
        area: 1900.0, // m^2 (cross-sectional area)
    };
    let mass = 420_000.0; // kg

    // MET atmosphere (Jacchia 1971) at solar mean conditions.
    let met_model = met_atmosphere::SOLAR_MEAN;
    let f10 = met_model.f10;
    let f10b = met_model.f10b;

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sim = Simulation::new(time, 60.0);
    let earth_idx = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        // Set to Some so Simulation updates this with GMST each step.
        t_inertial_pfix: Some(DMat3::IDENTITY),
    });
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met_model),
        r_eq: R_EARTH_EQ,
        r_pol: R_EARTH_POL,
        planet_omega: OMEGA_EARTH,
    });
    sim.atmosphere_planet_source = Some(earth_idx);

    let body_idx = sim.add_body(SimBody {
        trans: state0,
        rot: None,
        mass: Some(MassProperties::new(mass)),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        drag: Some(drag_config),
        flat_plates: None,
        plate_temperatures: vec![],
        plate_t_pow4_cached: vec![],
        shadow_body: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: Some(Default::default()),
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });
    sim.validate().expect("valid LEO drag setup");

    let dt = 60.0; // 1-minute steps
    let total_time = 86400.0; // 24 hours
    let steps = (total_time / dt) as usize;
    let print_interval = steps / 24; // Print once per hour

    let initial = sim.body(body_idx).trans;
    let initial_e = eccentricity(MU_EARTH, initial.position, initial.velocity);
    let initial_a = semi_major_axis(MU_EARTH, initial.position, initial.velocity);
    println!("=== LEO Orbit with Atmospheric Drag (MET Jacchia 1971) ===");
    println!(
        "Initial: alt={:.1} km, e={:.6}, a={:.1} km",
        altitude / 1000.0,
        initial_e,
        initial_a / 1000.0,
    );
    println!(
        "Atmosphere: MET solar mean (F10.7={}, F10B={})",
        f10, f10b
    );
    println!();
    println!(
        "{:>8}  {:>10}  {:>12}  {:>10}  {:>14}  {:>12}",
        "Time(h)", "Alt(km)", "a(km)", "e", "Density(kg/m3)", "DragF(mN)"
    );
    println!("{}", "-".repeat(78));

    for step in 0..steps {
        sim.step();
        let sim_time = (step + 1) as f64 * dt;

        if (step + 1) % print_interval == 0 {
            let time_h = sim_time / 3600.0;
            let body = sim.body(body_idx);
            let state = body.trans;
            let alt_km = (state.position.length() - R_EARTH_EQ) / 1000.0;
            let e_mag = eccentricity(MU_EARTH, state.position, state.velocity);
            let a_km = semi_major_axis(MU_EARTH, state.position, state.velocity) / 1000.0;
            let atmos_state = body
                .atmospheric_state
                .as_ref()
                .expect("atmospheric state enabled");
            let drag = body.aero_force.expect("drag force should be computed");

            println!(
                "{:>8.1}  {:>10.3}  {:>12.3}  {:>10.6}  {:>14.6e}  {:>12.6}",
                time_h,
                alt_km,
                a_km,
                e_mag,
                atmos_state.density,
                drag.force.length() * 1000.0, // mN
            );
        }
    }

    let final_state = sim.body(body_idx).trans;
    let final_a = semi_major_axis(MU_EARTH, final_state.position, final_state.velocity);
    let final_e = eccentricity(MU_EARTH, final_state.position, final_state.velocity);
    let sma_decay = initial_a - final_a;

    println!();
    println!(
        "Final: a={:.3} km, e={:.6}",
        final_a / 1000.0,
        final_e
    );
    println!("SMA decay: {:.1} m over 24h", sma_decay);
}
