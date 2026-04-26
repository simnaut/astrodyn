//! Tier 3: SIM_SolarBeta edge-case cross-validation via Simulation pipeline
//!
//! RUN_incl_0:    Equatorial orbit (i=0) — beta tracks Sun declination (~23.4 deg).
//! RUN_incl_23_4: Inclination = Earth obliquity (23.44 deg).
//!
//! Uses `Simulation::step()` with 8x8 spherical harmonics gravity + DE421
//! ephemeris for Sun. Propagates from CSV initial conditions and compares
//! solar beta at each checkpoint.

use jeod_test_data::tier3_csv::{load_solar_beta_csv, test_data_path};

use glam::DVec3;
use jeod_runner::{
    DerivedStateConfig, GravitySourceEntry, RotationModel, Simulation, VehicleConfig,
};
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_SolarBeta epoch: 1991-01-01 00:00:00 UTC
/// JD = 2448257.5; TAI-UTC = 26 s at this date; TT = TAI + 32.184 s
const EPOCH_TDB_JD: f64 = 2_448_257.5 + 58.184 / 86_400.0;
/// TAI TJT = MJD(TAI) - 40000 = (JD + TAI-UTC/86400 - 2400000.5) - 40000
/// = 2448257.5 + 26/86400 - 2400000.5 - 40000 = 8257.000300925926
const EPOCH_TAI_TJT: f64 = 2_448_257.5 + 26.0 / 86_400.0 - 2_400_000.5 - 40_000.0;

fn run_solar_beta_test(
    csv_filename: &str,
    label: &str,
    test_name: &str,
    beta_tol: f64,
    use_sh_gravity: bool,
) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_SolarBeta CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );

    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");
    let ggm05c_path = grav_data_dir.join("earth_GGM05C.cc");
    let sh_data = if use_sh_gravity {
        Some(jeod_sim::coefficients::load_from_jeod_cc(&ggm05c_path).expect("load GGM05C"))
    } else {
        None
    };
    let mu_earth = if let Some(ref sh) = sh_data {
        sh.mu
    } else {
        jeod_sim::coefficients::load_mu_from_jeod_cc(&ggm05c_path).expect("load Earth mu")
    };

    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");
    let records = load_solar_beta_csv(&csv_path);
    assert!(
        records.len() > 2,
        "Expected at least 3 records in {csv_filename}, got {}",
        records.len()
    );

    let init = &records[0];

    // Load dt from SIM_SolarBeta S_define (matches JEOD's integration step)
    let sim_dir = jeod_root.join("models/dynamics/derived_state/verif/SIM_SolarBeta");
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Build Simulation at the SIM_SolarBeta epoch with UT1-TAI offset
    // from JEOD time config (affects GMST → Earth RNP → SH gravity evaluation)
    let leap_table = jeod_sim::default_leap_second_table();
    let mut time = SimulationTime::new(EPOCH_TAI_TJT, leap_table);
    let time_cfg = jeod_test_data::time_config::load_time_config(
        &sim_dir.join("Modified_data/date_and_time.py"),
    );
    if let Some(ut1_tai) = time_cfg.ut1_tai_offset() {
        time.set_ut1_tai_offset(ut1_tai);
    }
    let mut sim = Simulation::new(time, dt);

    let gravity_model = if let Some(sh) = sh_data {
        GravityModel::SphericalHarmonics(Box::new(sh))
    } else {
        GravityModel::PointMass
    };
    // For SH gravity, initialize t_inertial_pfix to the correct RNP rotation
    // at the epoch (not IDENTITY). IDENTITY is only valid near J2000; at 1991
    // the precession/nutation offset is significant.
    let initial_rotation = if use_sh_gravity {
        Some(jeod_sim::compute_t_parent_this_from_tjt_with_polar(
            sim.time.gmst_seconds,
            sim.time.tt_tjt(),
            None,
        ))
    } else {
        None
    };
    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: gravity_model,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: initial_rotation,
            delta_c20: 0.0,
            rotation_model: if use_sh_gravity {
                RotationModel::EarthRNP
            } else {
                RotationModel::default()
            },
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Sun source — position from DE421 at epoch
    let (initial_sun_typed, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, EPOCH_TDB_JD)
        .expect("Sun position at epoch");
    let initial_sun = initial_sun_typed.raw_si();
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: initial_sun,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun);
    sim.ephemeris = Some(ephemeris);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![if use_sh_gravity {
                GravityControl::new_nonspherical(earth, 8, 8, false)
            } else {
                GravityControl::new_spherical(earth, false)
            }],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_SolarBeta {label}, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len());
    let mut ref_states = Vec::with_capacity(records.len());
    let mut max_beta_err = 0.0_f64;
    let mut max_pos_err = 0.0_f64;

    // Record t=0 state
    {
        let body = sim.body(0);
        let our_beta = body.solar_beta.unwrap_or(0.0);
        let _beta_err = (our_beta - init.solar_beta).abs();
        // solar_beta not computed before first step; skip t=0 in error tracking
    }

    for rec in &records[1..] {
        sim.step_until(rec.time);

        let body = sim.body(0);
        let our_beta = body.solar_beta.unwrap_or(0.0);
        let beta_err = (our_beta - rec.solar_beta).abs();
        let pos_err = (body.trans.position - rec.position).length();

        max_beta_err = max_beta_err.max(beta_err);
        max_pos_err = max_pos_err.max(pos_err);

        our_states.push(StateLog {
            time: rec.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: rec.time,
            position: Some(rec.position),
            velocity: Some(rec.velocity),
            ..Default::default()
        });

        if (rec.time % 86400.0).abs() < dt + 1.0 {
            println!(
                "  t={:8.0}s: jeod_beta={:.4} deg  our_beta={:.4} deg  beta_err={:.4e} rad  pos_err={:.2} m",
                rec.time,
                rec.solar_beta.to_degrees(),
                our_beta.to_degrees(),
                beta_err,
                pos_err,
            );
        }
    }

    println!(
        "  Max beta error: {:.6e} rad  Max position error: {:.2} m",
        max_beta_err, max_pos_err
    );

    let mut report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.add_extra("beta", max_beta_err, "rad");
    report.write();

    assert!(
        max_beta_err < beta_tol,
        "{label}: beta error {max_beta_err:.3e} rad exceeds {beta_tol:.3e} rad"
    );
}

#[test]
fn tier3_simulation_solar_beta_equ() {
    // Equatorial orbit: no J2 RAAN drift, point-mass is sufficient
    run_solar_beta_test(
        "solarbeta_incl_0_solarbeta.csv",
        "RUN_incl_0 (equatorial)",
        "tier3_simulation_solar_beta_equ",
        1.892e-5,
        false,
    );
}

#[test]
fn tier3_simulation_solar_beta_obliquity() {
    // Inclined orbit: 8x8 SH gravity captures J2 RAAN drift that changes
    // orbital plane orientation vs Sun, directly affecting solar beta.
    run_solar_beta_test(
        "solarbeta_incl_23_4_solarbeta.csv",
        "RUN_incl_23_4 (obliquity)",
        "tier3_simulation_solar_beta_obliquity",
        3.446e-5,
        true,
    );
}
