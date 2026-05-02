//! `VerificationCase` constructor for SIM_tide_verif RUN_01.
//!
//! Validates the tidal ΔC20 computation against JEOD's SIM_tide_verif:
//! GGM05C 8×8 + solid body tides + Sun/Moon 3rd-body, ISS highly
//! elliptical orbit, 8 h at 60 s logging.
//!
//! Two assertions per record:
//! 1. Trajectory (position, velocity) against JEOD's logged state.
//! 2. ΔC20 (`Simulation::source_delta_c20(earth)`) against JEOD's
//!    `earth.sb_tide.dC20` column via [`ExtrasComparator::TideDc20`].
//!
//! The pre-step hook updates Sun/Moon source positions *and* the
//! Earth source's tidal-bodies positions before each `step_until`
//! call, mirroring the bespoke loop's update pattern exactly so
//! baselines stay bit-stable.

use jeod_sim::Vec3Ext;
use std::path::PathBuf;

use glam::{DMat3, DVec3};
use jeod_gravity::tides::{TidalBody, TidalConfig, EARTH_K2};
use jeod_sim::recipes::verification::{
    CsvReference, ExtrasComparator, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use jeod_sim::{
    default_leap_second_table, Ephemeris, EphemerisBody, GravityControl, GravityControls,
    GravityModel, GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationModel,
    RotationalState, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";
const OMEGA_EARTH: f64 = jeod_sim::planet_config::EARTH.omega;

const EARTH_IDX: usize = 0;
const SUN_IDX: usize = 1;
const MOON_IDX: usize = 2;

fn bsp_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(p.exists(), "DE421 ephemeris not found at {}", p.display());
    p
}

fn dyncomp_time() -> SimulationTime {
    let time_cfg = jeod_test_data::time_config::load_time_config(
        &jeod_test_data::jeod_inputs::path(SIM_DYNCOMP).join("Modified_data/time.py"),
    );
    let mut time = SimulationTime::new(time_cfg.tai_tjt(), default_leap_second_table());
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");
    time.set_ut1_tai_offset(ut1_tai_offset);
    time
}

fn third_body(mu: f64, initial_pos: DVec3) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: initial_pos.m_at::<jeod_sim::RootInertial>(),
        velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

fn build_tide_run01(init: &InitialConditions) -> SimulationBuilder {
    let sim_dir = jeod_test_data::jeod_inputs::path(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Earth GGM05C SH, Sun mu, and Moon GRAIL150 mu all from committed
    // gravity fixtures (#249).
    let earth_grav = jeod_test_data::gravity_fixtures::load_ggm05c();
    let earth_mu = earth_grav.mu;
    let earth_radius = earth_grav.radius;
    let mu_sun = jeod_test_data::gravity_fixtures::load_sun_spherical_mu();
    let mu_moon = jeod_test_data::gravity_fixtures::load_moon_grail150_mu();

    let time = dyncomp_time();
    let epoch_tdb_jd = time.tdb_julian_date();
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, epoch_tdb_jd)
        .expect("Sun at epoch");
    let (moon_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Moon at epoch");
    let initial_sun = sun_t0.raw_si();
    let initial_moon = moon_t0.raw_si();

    let tidal_config = TidalConfig {
        k2: EARTH_K2,
        mu_primary: earth_mu,
        radius_primary: earth_radius,
        // Order matters: bespoke registers Moon at index 0 and Sun at
        // index 1, and the pre-step hook indexes by [0]/[1] to refresh
        // them. Keep that order.
        tidal_bodies: vec![
            TidalBody {
                mu: mu_moon,
                position_inertial: initial_moon,
            },
            TidalBody {
                mu: mu_sun,
                position_inertial: initial_sun,
            },
        ],
    };

    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: earth_mu,
                model: GravityModel::SphericalHarmonics(Box::new(earth_grav)),
            },
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            rotation_model: RotationModel::EarthRNP,
            delta_c20: 0.0,
            tidal_config: Some(tidal_config),
            planet_omega: OMEGA_EARTH,
            central: true,
        },
    );
    let sun = sb.add_source("Sun", third_body(mu_sun, initial_sun));
    let moon = sb.add_source("Moon", third_body(mu_moon, initial_moon));
    debug_assert_eq!(
        earth, EARTH_IDX,
        "Earth source index drifted; update EARTH_IDX or keep add_source order in sync"
    );
    debug_assert_eq!(
        sun, SUN_IDX,
        "Sun source index drifted; update SUN_IDX or keep add_source order in sync with pre_step"
    );
    debug_assert_eq!(
        moon, MOON_IDX,
        "Moon source index drifted; update MOON_IDX or keep add_source order in sync with pre_step"
    );

    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        // Bespoke uses identity quaternion + zero ω for the body — the
        // tidal CSV doesn't log attitude (8 columns: time, pos, vel,
        // dC20), so we mirror that here.
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        // Bespoke ISS mass tensor (literal in the original test).
        mass: Some(MassProperties::with_inertia(
            400_000.0,
            DMat3::from_cols(
                DVec3::new(1.02e8, -6.96e6, -5.48e6),
                DVec3::new(-6.96e6, 0.91e8, 5.90e5),
                DVec3::new(-5.48e6, 5.90e5, 1.64e8),
            ),
            DVec3::new(-3.0, -1.5, 4.0),
        )),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(earth, 8, 8, true),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        compute_gravity_gradient: true,
        ..Default::default()
    });
    sb
}

/// Pre-step factory: capture DE421 + the epoch TDB once, then push
/// Sun/Moon source positions to the upcoming step's TDB *and* update
/// Earth's `tidal_config.tidal_bodies` positions in lockstep so the
/// per-step ΔC20 computation sees consistent inputs. Mirrors the
/// bespoke loop's per-record update pattern exactly.
fn tide_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let epoch_tdb_jd = dyncomp_time().tdb_julian_date();
    Box::new(move |sim, time_s: f64| {
        let target_tdb_jd = epoch_tdb_jd + time_s / 86_400.0;
        let (sun_pos_typed, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Sun, target_tdb_jd)
            .expect("Sun");
        let (moon_pos_typed, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Moon, target_tdb_jd)
            .expect("Moon");
        let sun_pos = sun_pos_typed.raw_si();
        let moon_pos = moon_pos_typed.raw_si();

        sim.set_source_position(SUN_IDX, sun_pos);
        sim.set_source_position(MOON_IDX, moon_pos);
        // Earth's `TidalConfig::tidal_bodies` registers Moon at
        // index 0 and Sun at index 1 (see `build_tide_run01`); update
        // both in lockstep with the source positions so the per-step
        // ΔC20 sees consistent inputs.
        sim.set_tidal_body_position(EARTH_IDX, 0, moon_pos);
        sim.set_tidal_body_position(EARTH_IDX, 1, sun_pos);
    })
}

/// SIM_tide_verif RUN_01 — solid body tides + 3rd-body. Validates
/// trajectory (pos/vel) and per-step ΔC20 against JEOD's logged values.
pub fn run01() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_tide_run01",
        scenario: build_tide_run01,
        reference: CsvReference::Tide("tide_run01_tide.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            // Position / velocity tolerances inherited from the bespoke
            // test (5 % above observed). dC20 matches at machine
            // precision (the bespoke asserted < 1e-14).
            position_m: [2.117, 1.786, 0.582],
            velocity_m_s: [2.452e-3, 2.001e-3, 6.305e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[("dc20", 1.0e-14)],
        },
        extras: Some(ExtrasComparator::TideDc20 {
            earth_source_idx: EARTH_IDX,
        }),
        pre_step: Some(tide_pre_step),
    }
}
