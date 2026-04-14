//! Dawn spacecraft at Mars: high-fidelity spherical harmonics gravity.
//!
//! Demonstrates:
//! - Mars MRO110B2 110x110 spherical harmonics gravity
//! - Mars IAU rotation model
//! - Sun as 3rd-body perturbation with DE421 ephemeris
//!
//! Shows how Mars's lumpy gravity field perturbs the orbit over 3 hours.
//!
//! Requires JEOD source checkout (JEOD_HOME or JEOD_PATH) for gravity
//! coefficients and `test_data/de421.bsp` for ephemeris data.
//!
//! ```bash
//! cargo run -p jeod_runner --example mars_orbit
//! ```

use std::path::Path;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};

// Dawn spacecraft initial state at Mars (from JEOD SIM_Mars RUN_dawn, t=0).
// Mars-centered inertial frame.
const INIT_POS: DVec3 = DVec3::new(11_563_355.6802, -14_356_668.8977, 6_293_704.6169);
const INIT_VEL: DVec3 = DVec3::new(-2273.1078, 2380.1324, -22.911);

// Epoch: 2009-02-17 23:00:00 UTC (TAI-UTC = 34 s)
const EPOCH_TAI_TJT: f64 = 14_879.958_727;

// Propagation: 3 hours at 10 s (error insensitive to dt)
const DT: f64 = 10.0;
const DURATION: f64 = 10_800.0; // 3 hours

// Mars equatorial radius for altitude computation
const R_MARS_EQ: f64 = 3_396_190.0; // m

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let grav_dir = jeod_root.join("models/environment/gravity/data/src");
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let bsp_path = data_dir.join("de421.bsp");

    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );

    // Load Mars MRO110B2 110x110 spherical harmonics
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&grav_dir.join("mars_MRO110B2.cc"))?;
    let mars_mu = sh_data.mu;

    // Load Sun mu
    let mu_sun = jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_dir.join("sun_spherical.cc"))?;

    // Ephemeris for Sun position relative to Mars
    let ephemeris = jeod_sim::Ephemeris::from_bsp(&bsp_path)?;
    let time = SimulationTime::new(EPOCH_TAI_TJT, jeod_sim::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();

    let (sun_pos, _) = ephemeris.get_state(
        jeod_sim::EphemerisBody::Sun,
        jeod_sim::EphemerisBody::Mars,
        epoch_tdb_jd,
    )?;

    let mut sim = Simulation::new(time, DT);

    // Mars: central body with MRO110B2 SH gravity + IAU rotation
    let mars = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: mars_mu,
                model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(DMat3::IDENTITY),
            rotation_model: RotationModel::MarsIAU,
            delta_c20: 0.0,
            tidal_config: None,
            central: true,
        },
    );

    // Sun: 3rd-body with per-step DE421 ephemeris
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.set_source_ephemeris(
        sun,
        jeod_sim::EphemerisBody::Sun,
        jeod_sim::EphemerisBody::Mars,
    );
    sim.ephemeris = Some(ephemeris);

    // Dawn spacecraft (no SRP, no drag — gravity-only propagation)
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: INIT_POS,
            velocity: INIT_VEL,
        },
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(mars, 110, 110, false),
                GravityControl::new_third_body(sun),
            ],
        },
        ..Default::default()
    });

    sim.validate().expect("validation failed");

    println!("Dawn spacecraft at Mars — 3-hour propagation");
    println!("  Mars: MRO110B2 110x110 SH | IAU rotation | Sun 3rd-body");
    println!("  Integrator: RK4 at dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "Time (min)", "Alt (km)", "Speed (m/s)", "SMA (km)", "Ecc"
    );
    println!("{}", "-".repeat(62));

    let print_interval = 900.0; // 15 minutes
    let total_steps = (DURATION / DT).round() as usize;
    let steps_per_print = (print_interval / DT).round() as usize;

    for step in 1..=total_steps {
        sim.step();

        if step % steps_per_print == 0 {
            let body = sim.body(0);
            let pos = body.trans.position;
            let vel = body.trans.velocity;

            let r = pos.length();
            let v = vel.length();
            let altitude_km = (r - R_MARS_EQ) / 1000.0;
            let minutes = (step as f64) * DT / 60.0;

            // Vis-viva
            let energy = 0.5 * v * v - mars_mu / r;
            let sma = -mars_mu / (2.0 * energy);
            let sma_km = sma / 1000.0;
            let h_vec = pos.cross(vel);
            let ecc_vec = vel.cross(h_vec) / mars_mu - pos / r;
            let ecc = ecc_vec.length();

            println!("{minutes:10.1}  {altitude_km:12.1}  {v:12.1}  {sma_km:12.1}  {ecc:10.6}");
        }
    }

    println!();
    let final_body = sim.body(0);
    let final_alt = (final_body.trans.position.length() - R_MARS_EQ) / 1000.0;
    println!(
        "Final altitude: {final_alt:.1} km after {:.0} minutes",
        DURATION / 60.0
    );

    Ok(())
}
