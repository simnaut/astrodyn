//! Dawn spacecraft at Mars: high-fidelity Mars-centered trajectory.
//!
//! Demonstrates:
//!
//! - Mars MRO110B2 110x110 spherical harmonics gravity (coefficients loaded
//!   from the JEOD source checkout at runtime).
//! - Mars body-fixed rotation via the IAU pole + spin + nutation Fourier
//!   series (`RotationModel::MarsIAU`), matching JEOD's `RNPMars`.
//! - Sun as a third-body point-mass, position refreshed every step from DE421.
//!
//! Initial conditions match JEOD SIM_Mars RUN_dawn: a Dawn-like hyperbolic
//! Mars flyby at TAI TJT = 14879.958727 days (2009-02-17 23:00 UTC + 34 s
//! TAI-UTC offset).
//!
//! Requires a JEOD source checkout (`JEOD_HOME` or `JEOD_PATH`) for gravity
//! coefficients and `test_data/de421.bsp` (committed) for ephemeris data.
//!
//! Run with:
//! ```bash
//! cargo run --example mars_orbit
//! ```

use std::path::Path;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};

// Dawn spacecraft initial state at Mars (from JEOD SIM_Mars RUN_dawn, t=0).
// Mars-centered inertial frame.
const INIT_POS: DVec3 = DVec3::new(11_563_355.680_2, -14_356_668.897_7, 6_293_704.616_9);
const INIT_VEL: DVec3 = DVec3::new(-2273.1078, 2380.1324, -22.911);

// Epoch: 2009-02-17 23:00:00 UTC (TAI-UTC = 34 s).
const EPOCH_TAI_TJT: f64 = 14_879.958_727;

// Integration configuration.
const DT: f64 = 10.0;
const DURATION: f64 = 10_800.0; // 3 hours

// Mars equatorial radius (m) — display only.
const R_MARS_EQ: f64 = 3_396_190.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH to point at \
         a JEOD 5.4 checkout.",
        jeod_root.display()
    );

    let grav_dir = jeod_root.join("models/environment/gravity/data/src");
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let bsp_path = data_dir.join("de421.bsp");

    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );

    // Load Mars MRO110B2 110x110 spherical harmonics and the Sun mu.
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&grav_dir.join("mars_MRO110B2.cc"))?;
    let mars_mu = sh_data.mu;
    let mu_sun = jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_dir.join("sun_spherical.cc"))?;

    // DE421 for Sun position relative to Mars each step.
    let ephemeris = jeod_sim::Ephemeris::from_bsp(&bsp_path)?;
    let time = SimulationTime::new(EPOCH_TAI_TJT, jeod_sim::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();

    let (sun_pos, _) =
        ephemeris.get_state(EphemerisBody::Sun, EphemerisBody::Mars, epoch_tdb_jd)?;

    let mut sim = Simulation::new(time, DT);

    // Mars: central body with MRO110B2 SH gravity and IAU rotation.
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
            planet_omega: 0.0,
            central: true,
        },
    );

    // Sun: third-body with per-step DE421 updates.
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
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Mars);
    sim.ephemeris = Some(ephemeris);

    // Dawn spacecraft: gravity-only propagation (no SRP/drag in this scenario).
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
        "{:>10}  {:>12}  {:>12}  {:>10}  {:>12}  {:>10}",
        "Time (min)", "Alt (km)", "|v| (m/s)", "|r| (km)", "SMA (km)", "Ecc"
    );
    println!("{}", "-".repeat(72));

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
            let r_km = r / 1000.0;
            let minutes = (step as f64) * DT / 60.0;

            // Vis-viva: energy < 0 is bound, > 0 is hyperbolic.
            let energy = 0.5 * v * v - mars_mu / r;
            let sma_km = -mars_mu / (2.0 * energy) / 1000.0;
            let h_vec = pos.cross(vel);
            let ecc_vec = vel.cross(h_vec) / mars_mu - pos / r;
            let ecc = ecc_vec.length();

            println!(
                "{minutes:10.1}  {altitude_km:12.1}  {v:12.1}  {r_km:10.1}  {sma_km:12.1}  {ecc:10.6}"
            );
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
