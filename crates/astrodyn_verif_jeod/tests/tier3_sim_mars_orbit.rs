//! Tier 3: SIM_Mars — Dawn spacecraft Mars orbit cross-validation.
//!
//! Validates Mars MRO110B2 110×110 spherical harmonics gravity with Mars IAU
//! rotation model and Sun 3rd-body gravity.
//! Achieved parity: ~3.8 m position error over 3 hours.

use astrodyn::Vec3Ext;
use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::{RotationModel, Simulation};
use astrodyn_verif_jeod::crossval::{CrossvalReport, StateLog};
use glam::{DMat3, DVec3};

fn load_mu_sun() -> f64 {
    astrodyn::gravity_fixtures::load_sun_spherical_mu()
}

/// Load a state CSV with interleaved columns: time, pos[0], vel[0], pos[1], vel[1], pos[2], vel[2].
/// This is the default DRAscii layout when position and velocity are logged per-component.
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
        // Interleaved: col1=pos[0], col2=vel[0], col3=pos[1], col4=vel[1], col5=pos[2], col6=vel[2]
        records.push(StateLog {
            time: p(0),
            position: Some(DVec3::new(p(1), p(3), p(5))),
            velocity: Some(DVec3::new(p(2), p(4), p(6))),
            ..Default::default()
        });
    }
    records
}

/// Dawn at Mars: Mars point-mass + Sun 3rd-body + Mars IAU rotation.
///
/// Point-mass baseline — upgrading to MRO110B2 spherical harmonics will
/// tighten the tolerance significantly.
#[test]
fn tier3_simulation_mars_dawn() {
    let mu_sun = load_mu_sun();
    let csv_path = test_data_path("mars_dawn_mars.csv");
    let ref_states = load_interleaved_csv(&csv_path, "SIM_Mars RUN_dawn");
    assert!(
        !ref_states.is_empty(),
        "No reference data for SIM_Mars RUN_dawn"
    );

    let init = &ref_states[0];
    let init_pos = init.position.unwrap();
    let init_vel = init.velocity.unwrap();

    // Load MRO110B2 spherical harmonics coefficients from the committed fixture.
    let sh_data = astrodyn::gravity_fixtures::load_mars_mro110b2();
    let mars_mu = sh_data.mu;

    // Dawn epoch: 2009-02-17 23:00:00 UTC
    // TAI-UTC = 34s at this epoch; TAI TJT = MJD - 40000, MJD = JD - 2400000.5
    // JD(2009-02-17 23:00 UTC) = 2454880.4583
    // MJD = 2454880.4583 - 2400000.5 = 54879.9583
    // UTC TJT = 54879.9583 - 40000 = 14879.9583
    // TAI TJT = UTC TJT + 34/86400 = 14879.9583 + 0.000394 = 14879.958727
    let dawn_tai_tjt = 14_879.958_727;
    let leap_table = astrodyn::default_leap_second_table();
    let time = SimulationTime::new(dawn_tai_tjt, leap_table);

    // Load DE421 for Sun position relative to Mars at Dawn epoch
    let bsp_path = astrodyn::ephemeris_assets::de421_path();
    let ephemeris = astrodyn::Ephemeris::from_bsp(&bsp_path).expect("load DE421");
    let epoch_tdb_jd = time.tdb_julian_date();
    let (sun_pos_typed, _sun_vel) = ephemeris
        .get_state_typed(
            astrodyn::EphemerisBody::Sun,
            astrodyn::EphemerisBody::Mars,
            epoch_tdb_jd,
        )
        .expect("Sun-Mars state from DE421");
    let sun_pos_from_mars = sun_pos_typed.raw_si();

    // JEOD SIM_Mars uses RK4 at 1 Hz (DYNAMICS = 1.0 in S_define).
    // Error is insensitive to dt (dt=1 and dt=10 give same result); use 10s for speed.
    let mut sim = Simulation::new(time, 10.0);

    // Mars at origin with IAU rotation + MRO110B2 SH gravity
    let mars = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: mars_mu,
                model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY), // Triggers Mars rotation update
            rotation_model: RotationModel::MarsIAU,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    // Sun as 3rd-body with per-step DE421 ephemeris updates
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            sun_pos_from_mars.m_at::<astrodyn::RootInertial>(),
            None,
        ),
    );
    sim.set_source_ephemeris(
        sun,
        astrodyn::EphemerisBody::Sun,
        astrodyn::EphemerisBody::Mars,
    );
    sim.ephemeris = Some(ephemeris);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init_pos,
            velocity: init_vel,
        }),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(mars, 110, 110, GravityGradient::Skip),
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
        sim.step_until(record.time).expect("step_until failed");
        let body = sim.body(0);
        if i < 3 || i == ref_states.len() - 2 {
            let err = (body.trans.position.raw_si() - record.position.unwrap()).length();
            println!("  t={:.0}: error={:.1} m", record.time, err);
        }
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            acceleration: Some(body.trans_accel),
            ang_accel: body.rot_accel,
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute(
        "tier3_mars_dawn",
        &our_states,
        &ref_states[..our_states.len()],
    );
    report.write();

    let max_pos = report.max_position_component();
    println!("  Mars Dawn: max position error = {max_pos:.1} m (MRO110B2 SH 110x110)");
    // After fixing the Mars rotation transpose and the 12-hour epoch offset,
    // the error dropped from 25 km to ~3.8 m. Residual is from SH evaluation
    // sensitivity and Lyapunov divergence in the 110x110 gravity field.
    // Observed max per-component: 3.8 m × 1.05 = 3.99 ≈ 4.0 m.
    report.assert_position([4.0, 4.0, 4.0]);
}
