//! Clementine lunar orbit: high-fidelity Moon-centered trajectory with all
//! interaction forces available in JEOD today.
//!
//! Demonstrates:
//!
//! - Moon LP150Q 60x60 spherical harmonics gravity (JEOD coefficient file
//!   loaded at runtime from the JEOD source checkout).
//! - Earth and Sun as third-body point-mass perturbations, positions refreshed
//!   from DE421 every step.
//! - Moon body-fixed rotation driven by the DE421 BPC libration kernel
//!   (`moon_pa_de421_1900-2050.bpc`), the high-fidelity path used by JEOD's
//!   SIM_Earth_Moon reference.
//! - Cannonball solar radiation pressure with Earth shadow eclipse geometry.
//!
//! Initial conditions match JEOD SIM_Earth_Moon RUN_clem: a Clementine-like
//! lunar orbit at TAI TJT = 9412.000324 days (1994-03-01 00:00 UTC epoch +
//! 28 s TAI-UTC offset).
//!
//! Requires a JEOD source checkout (set `JEOD_HOME` or `JEOD_PATH`) for
//! coefficient data, and `test_data/de421.bsp` + `test_data/moon_pa_de421_1900-2050.bpc`
//! (both committed) for ephemeris/libration.
//!
//! Run with:
//! ```bash
//! cargo run --example earth_moon
//! ```

use std::path::Path;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, SrpModel, VehicleConfig};
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    MassProperties, SimulationTime, TranslationalState,
};

// Clementine orbital state at t=0 (from JEOD SIM_Earth_Moon RUN_clem reference
// trajectory). Moon-centered inertial frame.
const INIT_POS: DVec3 = DVec3::new(1_296_944.012, -1_060_824.45, 2_522_289.146);
const INIT_VEL: DVec3 = DVec3::new(-930.578, -439.312, 862.075);

// Clementine-like spacecraft SRP configuration.
const MASS_KG: f64 = 424.0;
const SRP_CX_AREA: f64 = 2.1432; // m^2
const SRP_ALBEDO: f64 = 1.0;
const SRP_DIFFUSE: f64 = 0.27;

// Epoch: 1994-03-01 00:00:00 UTC, TAI-UTC = 28 s.
// TAI TJT = MJD - 40000 + 28/86400 days.
const EPOCH_TAI_TJT: f64 = 9412.0 + 28.0 / 86400.0;

// Integration configuration. The Tier 3 cross-validation uses dt = 0.03125 s
// to chase sub-meter parity with JEOD over seven days; this showcase uses a
// larger step so the example finishes in a reasonable interactive runtime.
const DT: f64 = 1.0;
const DURATION: f64 = 86_400.0; // 1 day

// Moon equatorial radius (m) — display only.
const R_MOON: f64 = 1_737_400.0;

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
    let bpc_path = data_dir.join("moon_pa_de421_1900-2050.bpc");

    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    assert!(
        bpc_path.exists(),
        "Moon BPC not found at {}",
        bpc_path.display()
    );

    // Load Moon LP150Q coefficients (60x60 usage level, truncated internally
    // by GravityControl::new_nonspherical below). The file itself is 150x150.
    let lp150q = jeod_sim::coefficients::load_from_jeod_cc(&grav_dir.join("moon_LP150Q.cc"))?;
    let moon_mu = lp150q.mu;
    let mu_earth = jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_dir.join("earth_GGM05C.cc"))?;
    let mu_sun = jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_dir.join("sun_spherical.cc"))?;

    // Load the planetary ephemeris and the Moon PA libration kernel.
    let mut ephemeris = Ephemeris::from_bsp(&bsp_path)?;
    ephemeris.load_bpc(&bpc_path)?;

    let time = SimulationTime::new(EPOCH_TAI_TJT, jeod_sim::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();

    let mut sim = Simulation::new(time, DT);

    // Moon: central body. Gravity is 60x60 spherical harmonics on top of the
    // LP150Q file; rotation is driven from DE421 BPC each step
    // (RotationModel::MoonDE421 queries sim.ephemeris in step()).
    let moon_rotation = ephemeris.get_body_rotation(EphemerisBody::Moon, epoch_tdb_jd)?;

    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: moon_mu,
                model: GravityModel::SphericalHarmonics(Box::new(lp150q)),
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(moon_rotation),
            rotation_model: RotationModel::MoonDE421,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Earth as third body. Position refreshed every step from DE421
    // (Moon-centered frame).
    let (earth_pos, _) =
        ephemeris.get_state(EphemerisBody::Earth, EphemerisBody::Moon, epoch_tdb_jd)?;
    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry::new(
            GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            earth_pos,
            None,
        ),
    );
    sim.set_source_ephemeris(earth, EphemerisBody::Earth, EphemerisBody::Moon);

    // Sun as third body + SRP source.
    let (sun_pos, _) =
        ephemeris.get_state(EphemerisBody::Sun, EphemerisBody::Moon, epoch_tdb_jd)?;
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
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Moon);
    sim.sun_source = Some(sun);
    sim.ephemeris = Some(ephemeris);

    // Clementine spacecraft: cannonball SRP, no drag (no lunar atmosphere),
    // no gravity torque (point-mass inertia left default).
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: INIT_POS,
            velocity: INIT_VEL,
        },
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(moon, 60, 60, false),
                GravityControl::new_third_body(earth),
                GravityControl::new_third_body(sun),
            ],
        },
        mass: Some(MassProperties::new(MASS_KG)),
        srp: Some(SrpModel::Cannonball {
            cx_area: SRP_CX_AREA,
            albedo: SRP_ALBEDO,
            diffuse: SRP_DIFFUSE,
        }),
        ..Default::default()
    });

    sim.validate().expect("validation failed");

    println!(
        "Clementine lunar orbit — {:.1}-day propagation",
        DURATION / 86400.0
    );
    println!("  Moon: LP150Q 60x60 SH | Earth + Sun 3rd-body | Cannonball SRP");
    println!("  Moon rotation: DE421 BPC libration");
    println!("  Integrator: RK4 at dt={DT} s | Mass: {MASS_KG} kg");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>10}  {:>12}  {:>10}",
        "Time (h)", "Alt (km)", "|v| (m/s)", "|r| (km)", "Energy (J/kg)", "Ecc"
    );
    println!("{}", "-".repeat(72));

    let print_interval = 7_200.0; // 2 hours
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
            let altitude_km = (r - R_MOON) / 1000.0;
            let r_km = r / 1000.0;
            let hours = (step as f64) * DT / 3600.0;

            // Specific orbital energy (J/kg) and eccentricity magnitude.
            let energy = 0.5 * v * v - moon_mu / r;
            let h_vec = pos.cross(vel);
            let ecc_vec = vel.cross(h_vec) / moon_mu - pos / r;
            let ecc = ecc_vec.length();

            println!(
                "{hours:10.1}  {altitude_km:12.1}  {v:12.1}  {r_km:10.1}  {energy:12.2}  {ecc:10.6}"
            );
        }
    }

    println!();
    let final_body = sim.body(0);
    let final_alt = (final_body.trans.position.length() - R_MOON) / 1000.0;
    println!(
        "Final altitude: {final_alt:.1} km after {:.1} days",
        DURATION / 86400.0
    );

    Ok(())
}
