//! Tier 3: SIM_Earth_Moon — Clementine lunar orbit cross-validation.
//!
//! Validates multi-body gravity (Earth + Moon LP150Q 60×60 spherical harmonics,
//! Sun 3rd-body, DE421 BPC libration, cannonball SRP) against the JEOD
//! reference trajectory. Clementine-like orbit, 7 days (604,800 s).
//!
//! Matches JEOD SIM_Earth_Moon RUN_clem configuration:
//! - Integrator: RK4 at 0.03125 s (32 Hz)
//! - Moon gravity: LP150Q 60×60
//! - Moon rotation: DE421 BPC libration (per-step update)
//! - Earth/Sun: point-mass 3rd-body with per-step DE421 ephemeris (JEOD uses DE405)
//! - SRP: cannonball (cx_area=2.1432 m², albedo=1.0, diffuse=0.27)
//! - No drag, no gravity torque

use jeod_test_data::tier3_csv::test_data_path;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, SrpModel, VehicleConfig};
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

fn load_mu_earth() -> f64 {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C")
}

fn load_mu_sun() -> f64 {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/sun_spherical.cc"),
    )
    .expect("load Sun mu from sun_spherical")
}

/// Load a state CSV with interleaved columns: time, pos[0], vel[0], pos[1], vel[1], pos[2], vel[2].
fn load_interleaved_csv(path: &std::path::Path, sim_name: &str) -> Vec<StateLog> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {sim_name} CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 7,
            "line {}: expected >=7 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(StateLog {
            time: p(0),
            position: Some(DVec3::new(p(1), p(3), p(5))),
            velocity: Some(DVec3::new(p(2), p(4), p(6))),
            ..Default::default()
        });
    }
    records
}

/// Clementine lunar orbit: Moon LP150Q 60×60 + Earth 3rd-body + Sun 3rd-body
/// + cannonball SRP, matching JEOD SIM_Earth_Moon RUN_clem.
#[test]
fn tier3_simulation_earth_moon_clem() {
    let mu_earth = load_mu_earth();
    let mu_sun = load_mu_sun();
    let csv_path = test_data_path("earth_moon_clem_earth_moon.csv");
    let ref_states = load_interleaved_csv(&csv_path, "SIM_Earth_Moon RUN_clem");
    assert!(
        !ref_states.is_empty(),
        "No reference data for SIM_Earth_Moon RUN_clem"
    );

    // Use JEOD's initial state from CSV
    let init = &ref_states[0];
    let init_pos = init.position.unwrap();
    let init_vel = init.velocity.unwrap();

    // JEOD uses DE405; we use DE421 (no LE DE405 BSP available for Anise).
    // DE405/DE421 difference is sub-meter for Moon-centered 7-day orbits.
    let bsp_path = test_data_path("de421.bsp");
    let mut ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");
    let bpc_path = test_data_path("moon_pa_de421_1900-2050.bpc");
    ephemeris
        .load_bpc(&bpc_path)
        .expect("load Moon BPC for libration");

    // Clementine epoch: 1994-03-01 00:00:00 UTC
    // JD = 2449412.5; MJD = 49412.0; TJT = MJD - 40000 = 9412.0
    // TAI-UTC = 28 s at 1994-03-01 (29th leap second added 1994-07-01)
    let clem_tai_tjt = 9412.0 + 28.0 / 86400.0;
    let leap_table = jeod_sim::default_leap_second_table();
    let time = SimulationTime::new(clem_tai_tjt, leap_table);
    let mut sim = Simulation::new(time, 0.03125); // 32 Hz, matching JEOD S_define

    // Load LP150Q spherical harmonics for Moon (matching JEOD's SIM_Earth_Moon)
    let jeod_root = jeod_test_data::jeod_path();
    let lp150q_path = jeod_root.join("models/environment/gravity/data/src/moon_LP150Q.cc");
    let sh_data =
        jeod_sim::coefficients::load_from_jeod_cc(&lp150q_path).expect("load LP150Q coefficients");
    let moon_mu = sh_data.mu;

    // Moon rotation from DE421 BPC libration data, updated per step.
    let epoch_tdb_jd = sim.time.tdb_julian_date();
    let moon_rotation = ephemeris
        .get_body_rotation(EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Moon DE421 libration rotation");

    // Moon at origin with LP150Q SH gravity + per-step DE421 BPC rotation.
    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: moon_mu,
                model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
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

    // Earth as 3rd-body with per-step ephemeris updates
    let epoch_tdb_jd = sim.time.tdb_julian_date();
    let (earth_pos_typed, _earth_vel) = ephemeris
        .get_state_typed(EphemerisBody::Earth, EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Earth-Moon state from DE421");
    let earth_pos_from_moon = earth_pos_typed.raw_si();

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry::new(
            GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            earth_pos_from_moon,
            None,
        ),
    );
    sim.set_source_ephemeris(earth, EphemerisBody::Earth, EphemerisBody::Moon);

    // Sun as 3rd-body with per-step ephemeris updates (also SRP source)
    let (sun_pos_typed, _) = ephemeris
        .get_state_typed(EphemerisBody::Sun, EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Sun-Moon state from DE421");
    let sun_pos_from_moon = sun_pos_typed.raw_si();
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            sun_pos_from_moon,
            None,
        ),
    );
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Moon);
    sim.sun_source = Some(sun);

    // Store ephemeris for per-step updates
    sim.ephemeris = Some(ephemeris);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init_pos,
            velocity: init_vel,
        },
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(moon, 60, 60, false),
                GravityControl::new_third_body(earth),
                GravityControl::new_third_body(sun),
            ],
        },
        // Clementine mass: 424 kg (from Modified_data/mass.py)
        mass: Some(jeod_sim::MassProperties::new(424.0)),
        // Cannonball SRP matching JEOD Clementine: cx_area=2.1432 m²,
        // albedo=1.0, diffuse=0.27 (from Modified_data/radiation_pressure.py)
        srp: Some(SrpModel::Cannonball {
            cx_area: 2.1432,
            albedo: 1.0,
            diffuse: 0.27,
        }),
        ..Default::default()
    });

    sim.validate().unwrap();

    let mut our_states = vec![StateLog {
        time: 0.0,
        position: Some(init_pos),
        velocity: Some(init_vel),
        ..Default::default()
    }];

    for (i, record) in ref_states[1..].iter().enumerate() {
        sim.step_until(record.time).expect("step_until failed");
        let body = sim.body(0);
        if i == 0 {
            let jeod_pos = record.position.unwrap();
            println!(
                "  t={}: ours=[{:.1}, {:.1}, {:.1}]",
                record.time, body.trans.position.x, body.trans.position.y, body.trans.position.z
            );
            println!(
                "  t={}: JEOD=[{:.1}, {:.1}, {:.1}]",
                record.time, jeod_pos.x, jeod_pos.y, jeod_pos.z
            );
            let err = (body.trans.position - jeod_pos).length();
            println!("  t={}: error={:.1} m", record.time, err);
        }
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute(
        "tier3_earth_moon_clem",
        &our_states,
        &ref_states[..our_states.len()],
    );
    report.write();

    let max_pos = report.max_position_component();
    println!(
        "  Earth-Moon Clem: max position error = {:.2} m \
         (LP150Q 60x60 + DE421 BPC + cannonball SRP, dt=0.03125s, 7 days)",
        max_pos
    );
    // Residual from DE405/DE421 difference (JEOD uses DE405, we use DE421).
    // Tolerance: observed max × 1.05.
    report.assert_position([0.832, 0.331, 0.972]);
}
