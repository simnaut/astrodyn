//! Tier 3: SIM_Earth_Moon — Clementine lunar orbit cross-validation.
//!
//! Validates multi-body gravity (Earth + Moon LP150Q 60×60 spherical harmonics,
//! Sun 3rd-body, DE421 BPC libration) against the JEOD reference trajectory.
//! Clementine-like orbit, 7 days (604,800 s at 60 s intervals).

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, RotationModel, SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

const MU_EARTH: f64 = 3.986_004_415e14;

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

/// Clementine lunar orbit: Moon LP150Q 60×60 + Earth 3rd-body + Sun 3rd-body.
///
/// Loads DE421 ephemeris for third-body positions and DE421 BPC libration
/// for lunar orientation.
#[test]
fn tier3_simulation_earth_moon_clem() {
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

    // Load ephemeris for 3rd-body positions + Moon orientation
    let bsp_path = test_data_path("de421.bsp");
    let mut ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");
    let bpc_path = test_data_path("moon_pa_de421_1900-2050.bpc");
    ephemeris
        .load_bpc(&bpc_path)
        .expect("load Moon BPC for libration");

    // Clementine epoch: 1994-03-01 00:00:00 UTC
    // JD = 2449412.5; MJD = 49412.0; TJT = MJD - 40000 = 9412.0
    // TAI-UTC = 29s at 1994-03-01; TAI TJT = UTC TJT + 29/86400
    let clem_tai_tjt = 9412.0 + 29.0 / 86400.0;
    let leap_table = jeod_sim::default_leap_second_table();
    let time = SimulationTime::new(clem_tai_tjt, leap_table);
    let mut sim = Simulation::new(time, 10.0); // 10s timestep

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
    let moon = sim.add_source(GravitySourceEntry {
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
    });

    // Earth as 3rd-body with per-step ephemeris updates
    let epoch_tdb_jd = sim.time.tdb_julian_date();
    let (earth_pos_from_moon, _earth_vel) = ephemeris
        .get_state(EphemerisBody::Earth, EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Earth-Moon state from DE421");

    let earth = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        earth_pos_from_moon,
        None,
    ));
    // Enable per-step ephemeris update for Earth position relative to Moon
    sim.set_source_ephemeris(earth, EphemerisBody::Earth, EphemerisBody::Moon);

    // Sun as 3rd-body with per-step ephemeris updates
    let (sun_pos_from_moon, _) = ephemeris
        .get_state(EphemerisBody::Sun, EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Sun-Moon state from DE421");
    let sun = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: 1.327_124_40e20,
            model: GravityModel::PointMass,
        },
        sun_pos_from_moon,
        None,
    ));
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Moon);

    // Store ephemeris for per-step updates
    sim.ephemeris = Some(ephemeris);

    sim.add_body(SimBody {
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
        sim.step_until(record.time);
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
            println!("  mu={:.6e}", sim.sources[moon].source.mu);
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
        "  Earth-Moon Clem: max position error = {:.1} m (LP150Q 60x60 + DE421 BPC libration)",
        max_pos
    );
    // 7-day lunar orbit with LP150Q 60x60, per-step DE421 BPC libration,
    // Earth+Sun 3rd-body with per-step ephemeris. Residual from DE421
    // ephemeris drift (~10 arcsec Sun direction offset over 7 days).
    // Tolerance: observed max × 1.05.
    report.assert_position([222.0, 133.0, 314.0]);
}
