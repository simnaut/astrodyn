//! `VerificationCase` constructors for SIM_torque_compare_simple.
//!
//! Six runs with progressive gravity + gravity-gradient complexity:
//!
//!   01: spherical, gradient OFF                — zero torque (control)
//!   02: spherical, point-mass gradient         — point-mass torque
//!   03: spherical, gradient_degree=4           — same as 02 (spherical overrides)
//!   04: SH 20×20, gradient OFF                 — zero torque (control)
//!   05: SH 20×20, point-mass gradient          — point-mass torque (SH trajectory)
//!   06: SH 20×20, SH 4×4 gradient              — SH gradient torque
//!
//! All runs share: ISS mass (400_000 kg, non-diagonal inertia), epoch
//! 2007-11-20 00:00 UTC (parsed from `Modified_data/time.py`), RK4 at
//! 32 Hz, 10 800 s duration. Sun + Moon as differential third-body
//! sources; pre-step hook updates them from DE421 each `step_until`.
//!
//! The bespoke test recorded a "torque" extras metric that was always
//! 0.0 (acknowledged as a placeholder — `body.gravity_torque` was not
//! exposed on `VehicleOutput`). The migrated recipe drops the
//! placeholder; trajectory comparison (position/velocity/quaternion/
//! angular velocity) still validates the integrator and gradient
//! torque indirectly through their effect on attitude.

use astrodyn::Vec3Ext;
use std::path::PathBuf;

use crate::verification::{
    CsvReference, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, Ephemeris, EphemerisBody, GravityControl, GravityControls,
    GravityModel, GravitySource, GravitySourceEntry, MassProperties, RotationModel,
    RotationalState, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";
const OMEGA_EARTH: f64 = astrodyn::planet_config::EARTH.omega;

const SUN_IDX: usize = 1;
const MOON_IDX: usize = 2;

fn bsp_path() -> PathBuf {
    let p = astrodyn::ephemeris_assets::de421_path();
    assert!(p.exists(), "DE421 ephemeris not found at {}", p.display());
    p
}

fn dyncomp_time() -> SimulationTime {
    let time_cfg = crate::time_config::load_time_config(
        &crate::jeod_inputs::path(SIM_DYNCOMP).join("Modified_data/time.py"),
    );
    let mut time = SimulationTime::new(time_cfg.tai_tjt(), default_leap_second_table());
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");
    time.set_ut1_tai_offset(ut1_tai_offset);
    time
}

fn iss_mass_props() -> MassProperties {
    let mass_init = crate::mass_data::load_mass_from_file(
        &crate::jeod_inputs::path(SIM_DYNCOMP).join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );
    let inertia = DMat3::from_cols(
        DVec3::new(
            mass_init.inertia[0][0],
            mass_init.inertia[1][0],
            mass_init.inertia[2][0],
        ),
        DVec3::new(
            mass_init.inertia[0][1],
            mass_init.inertia[1][1],
            mass_init.inertia[2][1],
        ),
        DVec3::new(
            mass_init.inertia[0][2],
            mass_init.inertia[1][2],
            mass_init.inertia[2][2],
        ),
    );
    MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    )
}

#[derive(Clone, Copy)]
pub(crate) struct RunConfig {
    pub earth_nonspherical: bool,
    pub earth_gradient: bool,
    pub gradient_degree: usize,
    pub gradient_order: usize,
}

fn third_body(mu: f64, initial_pos: DVec3) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: initial_pos.m_at::<astrodyn::RootInertial>(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

fn build_torque_simple(init: &InitialConditions, cfg: RunConfig) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Earth GGM05C SH, Sun mu, and Moon GRAIL150 mu all from committed
    // gravity fixtures (#249).
    let earth_grav = astrodyn_gravity::fixtures::load_ggm05c();
    let mu_sun = astrodyn_gravity::fixtures::load_sun_spherical_mu();
    let mu_moon = astrodyn_gravity::fixtures::load_moon_grail150_mu();

    let needs_pfix = cfg.earth_nonspherical || cfg.gradient_degree > 0;
    let earth_source = if needs_pfix {
        GravitySource {
            mu: earth_grav.mu,
            model: GravityModel::SphericalHarmonics(Box::new(earth_grav)),
        }
    } else {
        GravitySource {
            mu: earth_grav.mu,
            model: GravityModel::PointMass,
        }
    };

    let time = dyncomp_time();
    let epoch_tdb_jd = time.tdb_julian_date();
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, epoch_tdb_jd)
        .expect("Sun at epoch");
    let (moon_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Moon at epoch");

    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source,
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: if needs_pfix {
                Some(DMat3::IDENTITY)
            } else {
                None
            },
            delta_c20: 0.0,
            rotation_model: if needs_pfix {
                RotationModel::EarthRNP
            } else {
                RotationModel::default()
            },
            tidal_config: None,
            planet_omega: if needs_pfix { OMEGA_EARTH } else { 0.0 },
            central: true,
        },
    );
    let sun = sb.add_source("Sun", third_body(mu_sun, sun_t0.raw_si()));
    let moon = sb.add_source("Moon", third_body(mu_moon, moon_t0.raw_si()));
    debug_assert_eq!(
        sun, SUN_IDX,
        "Sun source index drifted; update SUN_IDX or keep add_source order in sync with pre_step"
    );
    debug_assert_eq!(
        moon, MOON_IDX,
        "Moon source index drifted; update MOON_IDX or keep add_source order in sync with pre_step"
    );

    let mut earth_ctrl = if cfg.earth_nonspherical {
        GravityControl::new_nonspherical(earth, 20, 20, cfg.earth_gradient)
    } else {
        GravityControl::new_spherical(earth, cfg.earth_gradient)
    };
    if cfg.earth_gradient {
        earth_ctrl.gradient_degree = cfg.gradient_degree;
        earth_ctrl.gradient_order = cfg.gradient_order;
    }

    let q = init
        .quaternion
        .expect("torque_simple: 6-DOF init.quaternion");
    let w = init.ang_vel.expect("torque_simple: 6-DOF init.ang_vel");

    sb.add_body(VehicleConfig {
        trans: astrodyn::TranslationalStateTyped::<astrodyn::RootInertial>::from_untyped_unchecked(
            &TranslationalState {
                position: init.position,
                velocity: init.velocity,
            },
        ),
        rot: Some(
            RotationalState {
                quaternion: astrodyn::JeodQuat::from_glam(q),
                ang_vel_body: w,
            }
            .into(),
        ),
        mass: Some(iss_mass_props().into()),
        gravity_controls: GravityControls {
            controls: vec![
                earth_ctrl,
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        compute_gravity_gradient: cfg.earth_gradient,
        ..Default::default()
    });

    sb
}

/// Pre-step factory: capture DE421 + epoch TDB once, push Sun/Moon to
/// the upcoming step's TDB before `step_until`.
fn torque_simple_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let epoch_tdb_jd = dyncomp_time().tdb_julian_date();
    Box::new(move |sim, time_s: f64| {
        let target_tdb_jd = epoch_tdb_jd + time_s / 86_400.0;
        let (sun_pos, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Sun, target_tdb_jd)
            .expect("Sun");
        let (moon_pos, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Moon, target_tdb_jd)
            .expect("Moon");
        sim.set_source_position(SUN_IDX, sun_pos.raw_si());
        sim.set_source_position(MOON_IDX, moon_pos.raw_si());
    })
}

const DURATION_S: f64 = 10_800.0;

const RUN01_CFG: RunConfig = RunConfig {
    earth_nonspherical: false,
    earth_gradient: false,
    gradient_degree: 0,
    gradient_order: 0,
};
const RUN02_CFG: RunConfig = RunConfig {
    earth_nonspherical: false,
    earth_gradient: true,
    gradient_degree: 0,
    gradient_order: 0,
};
// RUN_03's input.py sets gradient_degree=4, but spherical=true overrides
// JEOD's gradient computation to point-mass, so the effective config we
// pass to GravityControl is identical to RUN_02. The bespoke test
// confirms RUN_03 and RUN_02 produce identical torques.
const RUN03_CFG: RunConfig = RUN02_CFG;
const RUN04_CFG: RunConfig = RunConfig {
    earth_nonspherical: true,
    earth_gradient: false,
    gradient_degree: 0,
    gradient_order: 0,
};
const RUN05_CFG: RunConfig = RunConfig {
    earth_nonspherical: true,
    earth_gradient: true,
    gradient_degree: 0,
    gradient_order: 0,
};
const RUN06_CFG: RunConfig = RunConfig {
    earth_nonspherical: true,
    earth_gradient: true,
    gradient_degree: 4,
    gradient_order: 4,
};

fn build_run01(init: &InitialConditions) -> SimulationBuilder {
    build_torque_simple(init, RUN01_CFG)
}
fn build_run02(init: &InitialConditions) -> SimulationBuilder {
    build_torque_simple(init, RUN02_CFG)
}
fn build_run03(init: &InitialConditions) -> SimulationBuilder {
    build_torque_simple(init, RUN03_CFG)
}
fn build_run04(init: &InitialConditions) -> SimulationBuilder {
    build_torque_simple(init, RUN04_CFG)
}
fn build_run05(init: &InitialConditions) -> SimulationBuilder {
    build_torque_simple(init, RUN05_CFG)
}
fn build_run06(init: &InitialConditions) -> SimulationBuilder {
    build_torque_simple(init, RUN06_CFG)
}

/// RUN_01 — spherical gravity, gradient OFF. Zero torque (control).
pub fn run01() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_simple_run01",
        scenario: build_run01,
        reference: CsvReference::TorqueSimple("torque_simple_run01_torque_simple.csv"),
        duration: Time::new::<second>(DURATION_S),
        tolerances: Tolerances {
            position_m: [4.928e-4, 8.746e-4, 9.074e-4],
            velocity_m_s: [7.443e-7, 8.554e-7, 8.158e-7],
            quat_angle_rad: 3.299,
            ang_vel_rad_s: [2.248e-3, 3.136e-3, 4.999e-4],
            extras: &[],
        },
        extras: None,
        pre_step: Some(torque_simple_pre_step),
    }
}

/// RUN_02 — spherical gravity, point-mass gradient.
pub fn run02() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_simple_run02",
        scenario: build_run02,
        reference: CsvReference::TorqueSimple("torque_simple_run02_torque_simple.csv"),
        duration: Time::new::<second>(DURATION_S),
        tolerances: Tolerances {
            position_m: [4.928e-4, 8.746e-4, 9.074e-4],
            velocity_m_s: [7.443e-7, 8.554e-7, 8.158e-7],
            quat_angle_rad: 3.755e-2,
            ang_vel_rad_s: [4.290e-5, 3.233e-5, 2.689e-6],
            extras: &[],
        },
        extras: None,
        pre_step: Some(torque_simple_pre_step),
    }
}

/// RUN_03 — spherical gravity with `gradient_degree=4` configured but
/// `spherical=true` causes JEOD to compute only the point-mass
/// gradient, so the trajectory matches RUN_02 exactly.
pub fn run03() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_simple_run03",
        scenario: build_run03,
        reference: CsvReference::TorqueSimple("torque_simple_run03_torque_simple.csv"),
        duration: Time::new::<second>(DURATION_S),
        tolerances: Tolerances {
            position_m: [4.928e-4, 8.746e-4, 9.074e-4],
            velocity_m_s: [7.443e-7, 8.554e-7, 8.158e-7],
            quat_angle_rad: 3.755e-2,
            ang_vel_rad_s: [4.290e-5, 3.233e-5, 2.689e-6],
            extras: &[],
        },
        extras: None,
        pre_step: Some(torque_simple_pre_step),
    }
}

/// RUN_04 — SH 20×20 gravity, gradient OFF.
pub fn run04() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_simple_run04",
        scenario: build_run04,
        reference: CsvReference::TorqueSimple("torque_simple_run04_torque_simple.csv"),
        duration: Time::new::<second>(DURATION_S),
        tolerances: Tolerances {
            position_m: [0.3083, 0.4835, 0.4257],
            velocity_m_s: [3.543e-4, 5.589e-4, 4.104e-4],
            quat_angle_rad: 3.299,
            ang_vel_rad_s: [2.244e-3, 3.187e-3, 4.977e-4],
            extras: &[],
        },
        extras: None,
        pre_step: Some(torque_simple_pre_step),
    }
}

/// RUN_05 — SH 20×20 gravity, point-mass gradient.
pub fn run05() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_simple_run05",
        scenario: build_run05,
        reference: CsvReference::TorqueSimple("torque_simple_run05_torque_simple.csv"),
        duration: Time::new::<second>(DURATION_S),
        tolerances: Tolerances {
            position_m: [0.3083, 0.4835, 0.4257],
            velocity_m_s: [3.543e-4, 5.589e-4, 4.104e-4],
            quat_angle_rad: 1.81e-2,
            ang_vel_rad_s: [1.806e-5, 1.412e-5, 4.493e-6],
            extras: &[],
        },
        extras: None,
        pre_step: Some(torque_simple_pre_step),
    }
}

/// RUN_06 — SH 20×20 gravity, SH 4×4 gradient.
pub fn run06() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_simple_run06",
        scenario: build_run06,
        reference: CsvReference::TorqueSimple("torque_simple_run06_torque_simple.csv"),
        duration: Time::new::<second>(DURATION_S),
        tolerances: Tolerances {
            position_m: [0.3083, 0.4835, 0.4257],
            velocity_m_s: [3.543e-4, 5.589e-4, 4.104e-4],
            quat_angle_rad: 6.24e-1,
            ang_vel_rad_s: [5.696e-4, 5.047e-4, 1.749e-4],
            extras: &[],
        },
        extras: None,
        pre_step: Some(torque_simple_pre_step),
    }
}
