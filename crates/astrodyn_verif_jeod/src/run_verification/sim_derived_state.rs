// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! `VerificationCase` constructors for the derived-state Tier 3 family
//! (SIM_OrbElem, SIM_LVLH, SIM_NED, SIM_Euler).
//!
//! All four sims share the same per-step physics — point-mass Earth
//! gravity, RK4 at the matching `S_define`'s `DT`, started from the
//! t=0 row of the reference CSV — and only differ in which
//! `DerivedStateConfig` flag is enabled and which CSV columns the
//! extras comparator reads.

use super::fixtures::load_mu_earth;
use crate::verification::{
    CsvReference, ExtrasComparator, InitialConditions, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, DerivedStateConfig, EulerSequence, GeodeticConfig, GravityControl,
    GravityControls, GravityGradient, GravityModel, GravitySource, GravitySourceEntry, JeodQuat,
    MassProperties, RotationModel, RotationalState, SimulationBuilder, SimulationTime,
    TranslationalState, VehicleConfig, EARTH,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

const DERIVED_STATE_VERIF: &str = "models/dynamics/derived_state/verif";
const SIM_ORBELEM_DIR: &str = "models/dynamics/derived_state/verif/SIM_OrbElem";
const SIM_LVLH_DIR: &str = "models/dynamics/derived_state/verif/SIM_LVLH";
const SIM_NED_DIR: &str = "models/dynamics/derived_state/verif/SIM_NED";
const SIM_EULER_DIR: &str = "models/dynamics/derived_state/verif/SIM_Euler";

/// UT1-TAI from JEOD `tai_to_ut1.cc` at 1991-01-01 (index 10592).
/// SIM_NED's epoch comes from `Modified_data/date_and_time.py` and the
/// UT1-TAI offset is read from JEOD's internal table at that epoch
/// rather than a sim config file — keep this constant in sync if the
/// upstream verif sim's epoch ever changes.
const NED_UT1_TAI_S: f64 = -25.381_221_5;

fn point_mass_earth(mu: f64, with_rnp: bool) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        // `t_inertial_pfix: Some(IDENTITY)` triggers the RNP update each
        // step, which the geodetic conversion needs. Other derived
        // states don't need it, so leave it `None` to match the JEOD
        // configurations bit-for-bit.
        t_inertial_pfix: if with_rnp {
            Some(DMat3::IDENTITY)
        } else {
            None
        },
        delta_c20: 0.0,
        rotation_model: if with_rnp {
            RotationModel::EarthRNP
        } else {
            RotationModel::default()
        },
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
        marker_only: false,
    }
}

// ── SIM_OrbElem ────────────────────────────────────────────────────────────

fn build_orbelem_ecc(init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let dt = crate::s_define::load_dynamics_dt(
        &crate::jeod_inputs::path(SIM_ORBELEM_DIR).join("S_define"),
    );

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth(mu, false));
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            orbital_elements_source: Some(earth),
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// SIM_OrbElem RUN_ecc — eccentric orbit (e=0.36), 24h, point-mass.
/// Validates classical orbital element extraction at every step.
pub fn orbelem_ecc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_orbelem",
        scenario: build_orbelem_ecc,
        reference: CsvReference::Orbelem("orbelem_ecc_orbelem.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [6.556e-5, 5.15e-5, 5.478e-8],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("sma", 2.613e-6),
                ("eccentricity", 1.496e-13),
                ("inclination", 8.436e-17),
                ("arg_periapsis", 1.78e-12),
                ("long_asc_node", 9.513e-14),
                ("true_anom", 1.136e-11),
                ("mean_anom", 5.642e-12),
            ],
        },
        extras: Some(ExtrasComparator::Orbelem),
        pre_step: None,
    }
}

// ── SIM_LVLH ───────────────────────────────────────────────────────────────

fn build_lvlh(init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let dt =
        crate::s_define::load_dynamics_dt(&crate::jeod_inputs::path(SIM_LVLH_DIR).join("S_define"));

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth(mu, false));
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            lvlh: true,
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// SIM_LVLH RUN_inc — inclined LEO (i=45°), 24h, point-mass.
pub fn lvlh_inc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_lvlh",
        scenario: build_lvlh,
        reference: CsvReference::Lvlh("lvlh_inc_lvlh.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [6.96e-5, 9.448e-5, 6.874e-5],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[("t_parent_this", 1.42e-11), ("ang_vel", 3.68e-16)],
        },
        extras: Some(ExtrasComparator::Lvlh),
        pre_step: None,
    }
}

/// SIM_LVLH RUN_ecc — eccentric (400 km × 8000 km), 24h.
pub fn lvlh_ecc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_lvlh_ecc",
        scenario: build_lvlh,
        reference: CsvReference::Lvlh("lvlh_ecc_lvlh.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [6.556e-5, 5.15e-5, 5.478e-8],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[("t_parent_this", 9.71e-12), ("ang_vel", 4.81e-15)],
        },
        extras: Some(ExtrasComparator::Lvlh),
        pre_step: None,
    }
}

/// SIM_LVLH RUN_equ — equatorial (i=0), near-singular LVLH.
pub fn lvlh_equ() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_lvlh_equ",
        scenario: build_lvlh,
        reference: CsvReference::Lvlh("lvlh_equ_lvlh.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [1.486e-4, 1.466e-4, 1.261e-7],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[("t_parent_this", 2.192e-11), ("ang_vel", 4.704e-16)],
        },
        extras: Some(ExtrasComparator::Lvlh),
        pre_step: None,
    }
}

// ── SIM_NED ────────────────────────────────────────────────────────────────

fn ned_time() -> SimulationTime {
    let verif_dir = crate::jeod_inputs::path(DERIVED_STATE_VERIF);
    let time_cfg =
        crate::time_config::load_time_config(&verif_dir.join("Modified_data/date_and_time.py"));
    let leap_table = default_leap_second_table();
    let tai_utc_s = leap_table.tai_utc_at_utc_tjt(time_cfg.utc_tjt());
    let epoch_tai_tjt = time_cfg.tai_tjt_with_offset(tai_utc_s);
    let mut time = SimulationTime::new(epoch_tai_tjt, leap_table);
    time.set_ut1_tai_offset(NED_UT1_TAI_S);
    time
}

fn build_ned(init: &InitialConditions, spherical: bool) -> SimulationBuilder {
    let mu = load_mu_earth();
    let dt =
        crate::s_define::load_dynamics_dt(&crate::jeod_inputs::path(SIM_NED_DIR).join("S_define"));

    let mut sb = SimulationBuilder::new(ned_time(), dt);
    let earth = sb.add_source("Earth", point_mass_earth(mu, true));

    let (r_eq, r_pol) = if spherical {
        (EARTH.shape.r_eq(), EARTH.shape.r_eq()) // spherical: r_pol = r_eq
    } else {
        (EARTH.shape.r_eq(), EARTH.shape.r_pol()) // ellipsoidal (WGS84)
    };

    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            geodetic: Some(GeodeticConfig {
                source_idx: earth,
                r_eq,
                r_pol,
            }),
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

fn build_ned_ell(init: &InitialConditions) -> SimulationBuilder {
    build_ned(init, false)
}
fn build_ned_sph(init: &InitialConditions) -> SimulationBuilder {
    build_ned(init, true)
}

/// SIM_NED RUN_ell_inc — ellipsoidal Earth, inclined orbit, 24h.
pub fn ned_ell_inc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_geodetic",
        scenario: build_ned_ell,
        reference: CsvReference::Ned("ned_ell_inc_ned.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [3.78e-6, 5.155e-6, 3.717e-6],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("altitude", 8.938e-4),
                ("latitude", 4.182e-8),
                ("longitude", 6.493e-8),
            ],
        },
        extras: Some(ExtrasComparator::Ned { spherical: false }),
        pre_step: None,
    }
}

/// SIM_NED RUN_ell_polar — ellipsoidal Earth, polar orbit, 24h.
/// Polar singularity: longitude becomes hypersensitive to sub-mm
/// position drift; tolerance widened per CLAUDE.md. See the
/// `# Numerical stability at the poles` section on
/// `astrodyn_math::GeodeticState` for the geometric rationale and the
/// fixed `~3.3e-5 rad` polar vs `~6.5e-8 rad` inclined tolerance ratio.
pub fn ned_ell_polar() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_ned_polar",
        scenario: build_ned_ell,
        reference: CsvReference::Ned("ned_ell_polar_ned.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [3.464e-6, 1.911e-5, 1.967e-5],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("altitude", 2.123e-4),
                ("latitude", 1.089e-8),
                ("longitude", 3.349e-5),
            ],
        },
        extras: Some(ExtrasComparator::Ned { spherical: false }),
        pre_step: None,
    }
}

/// SIM_NED RUN_sph_inc — spherical Earth, inclined orbit, 24h.
pub fn ned_sph_inc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_ned_sph_inc",
        scenario: build_ned_sph,
        reference: CsvReference::Ned("ned_sph_inc_ned.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [3.78e-6, 5.155e-6, 3.717e-6],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("altitude", 4.02e-7),
                ("latitude", 4.181e-8),
                ("longitude", 6.493e-8),
            ],
        },
        extras: Some(ExtrasComparator::Ned { spherical: true }),
        pre_step: None,
    }
}

/// SIM_NED RUN_sph_polar — spherical Earth, polar orbit, 24h.
pub fn ned_sph_polar() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_ned_sph_polar",
        scenario: build_ned_sph,
        reference: CsvReference::Ned("ned_sph_polar_ned.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [3.464e-6, 1.911e-5, 1.967e-5],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("altitude", 3.984e-7),
                ("latitude", 1.083e-8),
                ("longitude", 3.349e-5),
            ],
        },
        extras: Some(ExtrasComparator::Ned { spherical: true }),
        pre_step: None,
    }
}

// ── SIM_Euler ──────────────────────────────────────────────────────────────

/// ISS-like mass properties hardcoded by the existing Euler tests
/// (`Modified_data/mass.py` `set_mass_iss` defaults). Faithful copy of
/// the values the JEOD verif sim drives the body with.
fn iss_euler_mass_properties() -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0))
}

fn build_euler_run2(init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    // SIM_Euler reuses the SIM_dyncomp S_define dt (1s). The Euler
    // verif sim shares its time/integrator config with the dyncomp
    // RUN_2 trajectory it's driven from, so we read the same file the
    // existing test reads.
    let dt =
        crate::s_define::load_dynamics_dt(&crate::jeod_inputs::path("verif/SIM_dyncomp/S_define"));

    let q = init
        .quaternion
        .expect("euler_run2: 6-DOF init must include quaternion");
    let w = init
        .ang_vel
        .expect("euler_run2: 6-DOF init must include ang_vel");

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth(mu, false));
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        rot: Some(super::typed_helpers::rot_typed(
            &(RotationalState {
                quaternion: JeodQuat::from_glam(q),
                ang_vel_body: w,
            }),
        )),
        mass: Some(super::typed_helpers::mass_typed(
            &(iss_euler_mass_properties()),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            euler_sequence: Some(EulerSequence::XYZ),
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

fn build_euler_edge(init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let dt = crate::s_define::load_dynamics_dt(
        &crate::jeod_inputs::path(SIM_EULER_DIR).join("S_define"),
    );

    // SIM_Euler edge cases (ecc / equ) load the reference quaternion
    // from the CSV row at t=0 and force ang_vel = 0 — the JEOD verif
    // sim drives those runs from a static attitude.
    let init_q = init
        .quaternion
        .expect("euler_edge: SIM_Euler reference must populate quaternion at t=0");
    let init_quat = JeodQuat::from_glam(init_q);

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth(mu, false));
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        rot: Some(super::typed_helpers::rot_typed(
            &(RotationalState {
                quaternion: init_quat,
                ang_vel_body: DVec3::ZERO, // SIM_Euler initializes with zero ang vel
            }),
        )),
        mass: Some(super::typed_helpers::mass_typed(
            &(iss_euler_mass_properties()),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            euler_sequence: Some(EulerSequence::XYZ),
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// SIM_Euler reusing dyncomp RUN_2 — point-mass + ISS mass, 8h.
pub fn euler_run2() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_euler",
        scenario: build_euler_run2,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [0.0; 3], // 6-DOF rotational test; pos/vel asserts skipped
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("euler_roll", 1.846e-13),
                ("euler_pitch", 8.674e-14),
                ("euler_yaw", 1.103e-13),
            ],
        },
        extras: Some(ExtrasComparator::DyncompEuler),
        pre_step: None,
    }
}

/// SIM_Euler RUN_ecc — eccentric orbit, 24h.
pub fn euler_ecc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_euler_ecc",
        scenario: build_euler_edge,
        reference: CsvReference::Euler("euler_ecc_euler.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [0.0; 3],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 1e-10,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("euler_roll", 1e-10),
                ("euler_pitch", 1e-10),
                ("euler_yaw", 1e-10),
            ],
        },
        extras: Some(ExtrasComparator::Euler),
        pre_step: None,
    }
}

/// SIM_Euler RUN_equ — equatorial orbit, 24h.
pub fn euler_equ() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_euler_equ",
        scenario: build_euler_edge,
        reference: CsvReference::Euler("euler_equ_euler.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: Tolerances {
            position_m: [0.0; 3],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 1e-10,
            ang_vel_rad_s: [0.0; 3],
            extras: &[
                ("euler_roll", 1e-10),
                ("euler_pitch", 1e-10),
                ("euler_yaw", 1e-10),
            ],
        },
        extras: Some(ExtrasComparator::Euler),
        pre_step: None,
    }
}
