// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! `VerificationCase` constructors for the SIM_dyncomp Tier 3 family.
//!
//! Each constructor returns a fully-populated
//! [`crate::verification::VerificationCase`]
//! whose scenario closure loads its initial conditions from JEOD source
//! files (Modified_data/*.py, S_define, gravity coefficient files, plus
//! the t=0 row of the matching reference CSV — all "JEOD source data"
//! per CLAUDE.md). The scenario builds a [`SimulationBuilder`] that the
//! `run_and_assert` machinery materializes into a runtime
//! [`astrodyn_runner::Simulation`].

use std::path::PathBuf;

use crate::verification::{
    CsvReference, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use astrodyn::GeoIndexType;
use astrodyn::{
    default_leap_second_table, AtmosphereConfig, AtmosphereModel, BodyAction, DragConfig,
    Ephemeris, EphemerisBody, EulerSequence, GravityControl, GravityControls, GravityGradient,
    GravityModel, GravitySource, GravitySourceEntry, JeodQuat, LvlhAngularVelocityFrame,
    MassProperties, MetAtmosphere, RotationModel, RotationalState, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig, EARTH,
};
use glam::{DMat3, DVec3};
use uom::si::angle::degree;
use uom::si::f64::Angle;
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

/// Earth's sidereal angular velocity, sourced from
/// [`PlanetConfig::omega`](astrodyn::PlanetConfig).
const OMEGA_EARTH: f64 = astrodyn::planet_config::EARTH.omega;

fn point_mass_earth_source(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
        marker_only: false,
    }
}

// ── RUN_2: Point-mass 3-DOF / 6-DOF ────────────────────────────────────────

/// Translational state from the t=0 [`InitialConditions`] passed by
/// `run_and_assert`. Centralises the common builder pattern shared by
/// every dyncomp scenario.
fn trans_from(
    init: &InitialConditions,
) -> astrodyn::TranslationalStateTyped<astrodyn::RootInertial> {
    super::typed_helpers::trans_typed(&TranslationalState {
        position: init.position,
        velocity: init.velocity,
    })
}

/// Rotational state from a 6-DOF [`InitialConditions`]. Panics with a
/// descriptive error if either field is `None` (a 3-DOF init was passed
/// to a 6-DOF scenario constructor).
fn rot_from(init: &InitialConditions, case: &str) -> RotationalState {
    let q = init
        .quaternion
        .unwrap_or_else(|| panic!("{case}: 6-DOF scenario expected init.quaternion"));
    let w = init
        .ang_vel
        .unwrap_or_else(|| panic!("{case}: 6-DOF scenario expected init.ang_vel"));
    RotationalState {
        quaternion: JeodQuat::from_glam(q),
        ang_vel_body: w,
    }
}

fn build_run2_3dof(init: &InitialConditions) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_mu));
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        ..Default::default()
    });
    sb
}

fn build_run2_6dof(init: &InitialConditions) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mass_init = crate::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
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
    let mass_props = MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    );

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_mu));
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(
            &(rot_from(init, "run2_6dof")),
        )),
        mass: Some(super::typed_helpers::mass_typed(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        ..Default::default()
    });
    sb
}

/// SIM_dyncomp RUN_2 — point-mass 3-DOF (translational only).
pub fn run2_3dof() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run2_3dof",
        scenario: build_run2_3dof,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_2 — point-mass 6-DOF (with ISS mass properties).
pub fn run2_6dof() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run2_6dof",
        scenario: build_run2_6dof,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [2.619e-18, 1.367e-18, 7.969e-19],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── RUN_2 with InitLvlhRot post-init propagation ───────────────────────────
//
// Cross-validates both the `BodyAction::InitLvlhRot` initializer and
// the rotational integrator against the existing
// `dyncomp_run2_state.csv` reference. Where the sibling `run2_6dof`
// reads the t=0 quaternion straight off the reference CSV, this
// scenario reads the JEOD `Modified_data/state.py` Yaw-Pitch-Roll
// Euler triple + LVLH-relative angular velocity, runs them through
// `BodyAction::InitLvlhRot.apply_rotational()` (which delegates to
// the shipped `init_rot_from_lvlh` kernel), and uses the result as
// the initial rotational state. The reference CSV is suitable
// because every SIM_dyncomp run already initializes its rotational
// state through `set_orientation_lvlh`, which calls
// `DynBodyInitLvlhRotState` upstream — see
// `verif/SIM_dyncomp/Modified_data/state.py:set_orientation_lvlh`.

/// Yaw-Pitch-Roll (ZYX) Euler triple from a JEOD `state.py` LVLH-init
/// function name, converted to a JEOD scalar-first left-transformation
/// quaternion via [`astrodyn::compute_quaternion_from_euler_angles_typed`].
///
/// Maps the source's `trick.Orientation.<sequence>` string to the
/// matching [`EulerSequence`]. Only the sequences that appear in
/// SIM_dyncomp `Modified_data/state.py` are wired today; an unknown
/// sequence panics with a fail-loudly diagnostic rather than silently
/// substituting a default.
fn lvlh_init_from_state_py(
    state_py: &std::path::Path,
    function_name: &str,
) -> (JeodQuat, DVec3, LvlhAngularVelocityFrame) {
    let lvlh = crate::lvlh_init_data::load_lvlh_init_function(state_py, function_name);
    let sequence = match lvlh.euler_sequence.as_str() {
        // JEOD `trick.Orientation.Yaw_Pitch_Roll = 5`, ZYX axis order.
        // models/utils/orientation/include/orientation.hh:130
        "Yaw_Pitch_Roll" => EulerSequence::ZYX,
        other => panic!(
            "lvlh_init_from_state_py: unsupported euler sequence {other:?} in {} \
             (function {function_name}); add a mapping in sim_dyncomp.rs",
            state_py.display()
        ),
    };
    let angles = [
        Angle::new::<degree>(lvlh.euler_angles_deg[0]),
        Angle::new::<degree>(lvlh.euler_angles_deg[1]),
        Angle::new::<degree>(lvlh.euler_angles_deg[2]),
    ];
    let q_lvlh_body =
        astrodyn::compute_quaternion_from_euler_angles_typed(angles, sequence).inner();
    let ang_vel = DVec3::from_array(lvlh.ang_velocity);
    // SIM_dyncomp `set_orientation_lvlh` does not set `rate_in_parent`,
    // so JEOD's default `rate_in_parent = false` applies — the
    // user-supplied ang_velocity is in the body frame.
    (q_lvlh_body, ang_vel, LvlhAngularVelocityFrame::Body)
}

/// `RotationalState` reproduced by running `BodyAction::InitLvlhRot`
/// from JEOD source data (`Modified_data/state.py`) — *not* from the
/// reference CSV's t=0 quaternion. Exercises the LVLH-rot-init path
/// end-to-end through the public `BodyAction` API.
fn init_lvlh_rot_state(state_py: &std::path::Path) -> RotationalState {
    let trans = crate::lvlh_init_data::load_trans_init_function(state_py, "set_trans_init_typical");
    let (q_lvlh_body, ang_vel_lvlh_to_body, ang_vel_frame) =
        lvlh_init_from_state_py(state_py, "set_orientation_lvlh");
    let action = BodyAction::InitLvlhRot {
        q_lvlh_body,
        ang_vel_lvlh_to_body,
        ang_vel_frame,
        reference_position: DVec3::from_array(trans.position),
        reference_velocity: DVec3::from_array(trans.velocity),
    };
    action.apply_rotational().expect(
        "BodyAction::InitLvlhRot.apply_rotational must yield Some(RotationalState); \
         the variant is rotational by construction",
    )
}

/// Builder: same point-mass 6-DOF sim as `build_run2_6dof`, but the
/// initial *rotational* state is computed by
/// `BodyAction::InitLvlhRot` applied to the JEOD `state.py` Euler
/// triple + LVLH-relative ang velocity instead of being read from
/// the reference CSV's t=0 quaternion. The translational state is
/// also taken from `state.py:set_trans_init_typical`, since that's
/// the same JEOD source that produced the CSV's t=0 position /
/// velocity row — keeping both halves on the source-Python path
/// avoids any reliance on the reference output.
fn build_run2_lvlh_rot_init(_init: &InitialConditions) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mass_props = iss_mass_properties();
    let state_py = sim_dir.join("Modified_data/state.py");

    let trans =
        crate::lvlh_init_data::load_trans_init_function(&state_py, "set_trans_init_typical");
    let trans_state = super::typed_helpers::trans_typed(&TranslationalState {
        position: DVec3::from_array(trans.position),
        velocity: DVec3::from_array(trans.velocity),
    });
    let rot_state = init_lvlh_rot_state(&state_py);

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_mu));
    sb.add_body(VehicleConfig {
        trans: trans_state,
        rot: Some(super::typed_helpers::rot_typed(&(rot_state))),
        mass: Some(super::typed_helpers::mass_typed(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        ..Default::default()
    });
    sb
}

/// SIM_dyncomp RUN_2 — InitLvlhRot post-init propagation.
///
/// The initial rotational state is computed by our
/// `BodyAction::InitLvlhRot` from the JEOD `Modified_data/state.py`
/// Yaw-Pitch-Roll Euler triple `[0, -11.6, 0]` deg + zero LVLH-frame
/// angular velocity, then the simulation propagates 8 hours under
/// point-mass gravity. The propagated trajectory is compared against
/// the existing `dyncomp_run2_state.csv` reference (whose t=0 row
/// JEOD itself produced through `DynBodyInitLvlhRotState`), so the
/// assertion exercises *both* the LVLH-rot-init kernel *and* the
/// rotational integrator under SIM_dyncomp's mass properties.
pub fn run2_lvlh_rot_init_propagation() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run2_lvlh_rot_init_propagation",
        scenario: build_run2_lvlh_rot_init,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            // Tolerances are 5% above the observed max errors on
            // a clean run, per the standard CLAUDE.md policy. They
            // come out essentially identical to `run2_6dof`'s
            // baseline because the propagator path is the same and
            // the InitLvlhRot kernel reproduces the JEOD-side
            // quaternion to floating-point precision; the tiny
            // residuals are integration-step round-off, not
            // initialization error.
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.388e-9, 1.814e-9],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [2.619e-18, 1.367e-18, 7.969e-19],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── Shared helpers used by RUN_3+ (SH gravity, RNP rotation, time/UT1) ─────

fn iss_mass_properties() -> MassProperties {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let mass_init = crate::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
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

fn sphere_mass_properties() -> MassProperties {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let mass_init = crate::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_sphere"),
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

/// Simulation time anchored at the SIM_dyncomp epoch (parsed from
/// `Modified_data/time.py`), with the UT1-TAI offset applied.
fn dyncomp_time() -> SimulationTime {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let time_cfg = crate::time_config::load_time_config(&sim_dir.join("Modified_data/time.py"));
    let mut time = SimulationTime::new(time_cfg.tai_tjt(), default_leap_second_table());
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");
    time.set_ut1_tai_offset(ut1_tai_offset);
    time
}

/// Earth gravity source backed by GGM02C spherical-harmonics coefficients,
/// with the EarthRNP rotation model so the planet-fixed frame updates each
/// step. Used by the RUN_3A / RUN_3B / RUN_5* / RUN_6* configurations.
fn earth_sh_with_rnp() -> GravitySourceEntry {
    // GGM02C SH coefficients from the committed fixture (Wave 1 of #232).
    let sh_data = astrodyn::gravity_fixtures::load_ggm02c();
    GravitySourceEntry {
        source: GravitySource {
            mu: sh_data.mu,
            model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
        planet_omega: OMEGA_EARTH,
        central: true,
        marker_only: false,
    }
}

/// Earth point-mass with the EarthRNP rotation model (used by drag /
/// atmosphere runs that need geodetic coordinates each step).
fn earth_pm_with_rnp(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
        planet_omega: OMEGA_EARTH,
        central: true,
        marker_only: false,
    }
}

// ── RUN_3A / RUN_3B: spherical-harmonics gravity (4x4 / 8x8) + RNP ─────────

fn build_run3_3dof(init: &InitialConditions, run_dir: &str) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Gravity controls degree/order from RUN input chain.
    let mut grav_files: Vec<PathBuf> = vec![sim_dir.join("Modified_data/grav_controls.py")];
    grav_files.push(sim_dir.join("SET_test/RUN_3A/input.py"));
    if run_dir != "RUN_3A" {
        grav_files.push(sim_dir.join(format!("SET_test/{run_dir}/input.py")));
    }
    let grav_file_refs: Vec<&std::path::Path> = grav_files.iter().map(|p| p.as_path()).collect();
    let grav_cfg = crate::gravity_control::load_gravity_control(&grav_file_refs);

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_sh_with_rnp());
    let earth_gradient = if grav_cfg.gradient {
        GravityGradient::Compute
    } else {
        GravityGradient::Skip
    };
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth,
                grav_cfg.degree,
                grav_cfg.order,
                earth_gradient,
            )],
        },
        ..Default::default()
    });
    sb
}

fn build_run3a(init: &InitialConditions) -> SimulationBuilder {
    build_run3_3dof(init, "RUN_3A")
}

fn build_run3b(init: &InitialConditions) -> SimulationBuilder {
    build_run3_3dof(init, "RUN_3B")
}

/// SIM_dyncomp RUN_3A — 4×4 spherical-harmonics gravity + RNP rotation.
pub fn run3a_sh4x4() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run3a_sh4x4",
        scenario: build_run3a,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run3a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.3e-2, 1.344e-1, 1.026e-1],
            velocity_m_s: [6.151e-5, 1.246e-4, 1.24e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_3B — 8×8 spherical-harmonics gravity + RNP rotation.
pub fn run3b_sh8x8() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run3b_sh8x8",
        scenario: build_run3b,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run3b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.325e-1, 2.3e-1, 1.646e-1],
            velocity_m_s: [1.478e-4, 2.329e-4, 1.892e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── RUN_4: spherical Earth + Sun/Moon third-body (DE421) ──────────────────

/// Path to the bundled DE421 BSP kernel that drives Sun/Moon position
/// updates for third-body and tidal scenarios. Panics with the expected
/// path if missing — the file is committed to `test_data/` per CLAUDE.md.
fn bsp_path() -> PathBuf {
    let p = astrodyn::ephemeris_assets::de421_path();
    assert!(p.exists(), "DE421 ephemeris not found at {}", p.display());
    p
}

/// Index of the Sun gravity source inside the RUN_4 scenario. The
/// scenario builder `debug_assert!`s this against the actual returned
/// index so a future reorder doesn't silently desync the `pre_step`
/// hook from the source registry.
const RUN4_SUN_IDX: usize = 1;
/// Index of the Moon gravity source inside the RUN_4 scenario.
const RUN4_MOON_IDX: usize = 2;

fn third_body_source(mu: f64, initial_pos: DVec3) -> GravitySourceEntry {
    use astrodyn::Vec3Ext;
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
        marker_only: false,
    }
}

fn build_run4_3rd_body(init: &InitialConditions) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Earth GGM05C mu, Sun mu, and Moon GRAIL150 mu all from committed
    // gravity fixtures (#249).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mu_sun = astrodyn::gravity_fixtures::load_sun_spherical_mu();
    let mu_moon = astrodyn::gravity_fixtures::load_moon_grail150_mu();

    // Initial Sun/Moon positions at the dyncomp epoch — re-queried each
    // step by the `pre_step` hook below.
    let time = dyncomp_time();
    let epoch_tdb_jd = time.tdb_julian_date();
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, epoch_tdb_jd)
        .expect("Sun position at epoch");
    let (moon_t0, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Moon position at epoch");

    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_mu));
    let sun = sb.add_source("Sun", third_body_source(mu_sun, sun_t0.raw_si()));
    let moon = sb.add_source("Moon", third_body_source(mu_moon, moon_t0.raw_si()));
    debug_assert_eq!(
        sun, RUN4_SUN_IDX,
        "Sun source index drift: run4_pre_step assumes Sun is at \
         RUN4_SUN_IDX={RUN4_SUN_IDX}, but add_source returned {sun}."
    );
    debug_assert_eq!(
        moon, RUN4_MOON_IDX,
        "Moon source index drift: run4_pre_step assumes Moon is at \
         RUN4_MOON_IDX={RUN4_MOON_IDX}, but add_source returned {moon}."
    );

    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(
            &(rot_from(init, "run4_3rd_body")),
        )),
        mass: Some(super::typed_helpers::mass_typed(&(iss_mass_properties()))),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth, GravityGradient::Skip),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        ..Default::default()
    });
    sb
}

/// Pre-step factory for RUN_4: capture a DE421 ephemeris + the
/// epoch-anchored TDB JD once, then update Sun/Moon source positions to
/// the upcoming step's TDB before `step_until`. Mirrors the bespoke
/// test's per-record loop exactly so baselines stay bit-stable.
fn run4_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let epoch_tdb_jd = dyncomp_time().tdb_julian_date();
    Box::new(move |sim, time_s: f64| {
        let target_tdb_jd = epoch_tdb_jd + time_s / 86_400.0;
        let (sun_pos, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Sun, target_tdb_jd)
            .expect("Sun position query");
        let (moon_pos, _) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Moon, target_tdb_jd)
            .expect("Moon position query");
        sim.set_source_position(RUN4_SUN_IDX, sun_pos.raw_si());
        sim.set_source_position(RUN4_MOON_IDX, moon_pos.raw_si());
    })
}

/// SIM_dyncomp RUN_4 — spherical Earth + Sun/Moon third-body
/// gravity. Validates differential 3rd-body acceleration with DE421.
pub fn run4_3rd_body() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run4_3rd_body",
        scenario: build_run4_3rd_body,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run4_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            // Inherited from the bespoke assertion (5 % above observed).
            position_m: [1.644e-3, 2.098e-3, 2.025e-3],
            velocity_m_s: [1.762e-6, 2.082e-6, 2.400e-6],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [2.619e-18, 1.367e-18, 7.969e-19],
            extras: &[],
        },
        extras: None,
        pre_step: Some(run4_pre_step),
    }
}

// ── Battin's method vs direct subtraction (third-body cross-compare) ──────

/// Step size for the Battin cross-compare scenario.
///
/// SIM_dyncomp's S_define specifies `DYNAMICS = 0.03125` s (32 Hz). The
/// cross-compare doesn't validate against a JEOD CSV, so it uses a
/// coarser 10 s step that still resolves the fp-rounding divergence
/// between the two methods (~5 digits lost in direct subtraction of
/// LEO + Sun accelerations) without paying for ~921 600 integration
/// steps over 8 hours.
const BATTIN_DT: f64 = 10.0;

/// Output of [`build_battin_3rd_body`]: the configured simulation
/// builder plus the runtime source indices for Sun and Moon, captured
/// from `add_source` so [`battin_pre_step`] can update the right
/// entries without depending on registration order being stable
/// across future edits.
pub struct BattinScenario {
    /// Configured simulation builder, ready for `.build()`.
    pub builder: SimulationBuilder,
    /// Source index for Sun (third-body).
    pub sun_idx: usize,
    /// Source index for Moon (third-body).
    pub moon_idx: usize,
}

fn third_body_source_with_state(mu: f64, position: DVec3, velocity: DVec3) -> GravitySourceEntry {
    use astrodyn::Vec3Ext;
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: position.m_at::<astrodyn::RootInertial>(),
        velocity: velocity.m_per_s_at::<astrodyn::RootInertial>(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
        marker_only: false,
    }
}

/// Build a Battin/direct cross-compare scenario with Earth (central) +
/// Sun + Moon third-body gravity. The two sibling sims share initial
/// conditions; only the `battin_method` flag on the Sun/Moon
/// `GravityControl`s differs.
///
/// Used by the `tier3_sim_battin` test to verify that Battin's
/// reformulation produces the same trajectory as direct subtraction up
/// to fp rounding. There is no JEOD CSV reference for the comparison —
/// it is internal between the two sibling sims — so this returns a
/// [`BattinScenario`] (builder + source indices) rather than a
/// [`VerificationCase`].
pub fn build_battin_3rd_body(init: &InitialConditions, battin: bool) -> BattinScenario {
    // Earth GGM05C mu, Sun mu, and Moon GRAIL150 mu all from committed
    // gravity fixtures (#249).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mu_sun = astrodyn::gravity_fixtures::load_sun_spherical_mu();
    let mu_moon = astrodyn::gravity_fixtures::load_moon_grail150_mu();

    // Initial Sun/Moon state at the dyncomp epoch — refreshed each step
    // by the `pre_step` hook returned from [`battin_pre_step`].
    let time = dyncomp_time();
    let epoch_tdb_jd = time.tdb_julian_date();
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let (sun_pos_t0, sun_vel_t0) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, epoch_tdb_jd)
        .expect("Sun state at epoch");
    let (moon_pos_t0, moon_vel_t0) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Moon state at epoch");

    let mut sb = SimulationBuilder::new(time, BATTIN_DT);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_mu));
    let sun_idx = sb.add_source(
        "Sun",
        third_body_source_with_state(mu_sun, sun_pos_t0.raw_si(), sun_vel_t0.raw_si()),
    );
    let moon_idx = sb.add_source(
        "Moon",
        third_body_source_with_state(mu_moon, moon_pos_t0.raw_si(), moon_vel_t0.raw_si()),
    );

    let mut sun_control = GravityControl::new_third_body(sun_idx);
    sun_control.battin_method = battin;
    let mut moon_control = GravityControl::new_third_body(moon_idx);
    moon_control.battin_method = battin;

    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(
            &(rot_from(init, "battin_3rd_body")),
        )),
        mass: Some(super::typed_helpers::mass_typed(&(iss_mass_properties()))),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth, GravityGradient::Skip),
                sun_control,
                moon_control,
            ],
        },
        ..Default::default()
    });
    BattinScenario {
        builder: sb,
        sun_idx,
        moon_idx,
    }
}

/// Pre-step factory for the Battin cross-compare scenario: capture a
/// DE421 ephemeris and the dyncomp epoch's TDB JD once, then update the
/// Sun and Moon source position+velocity to the upcoming step's TDB
/// before `step_until`.
///
/// `sun_idx` and `moon_idx` come from the [`BattinScenario`] returned
/// by [`build_battin_3rd_body`] — capturing them at construction time
/// (rather than relying on hard-coded constants validated by
/// `debug_assert!`) prevents a future source-registration reorder from
/// silently updating the wrong entries in release builds.
///
/// Sets state (position **and** velocity), not just position, to match
/// the bespoke pre-recipe test exactly so the cross-compare baselines
/// stay bit-stable. Velocity does not affect third-body point-mass
/// gravity, but propagating it explicitly preserves the original
/// behavior with no reasoning required about which fields the gravity
/// pipeline reads.
pub fn battin_pre_step(sun_idx: usize, moon_idx: usize) -> PreStepClosure {
    let ephemeris = Ephemeris::from_bsp(&bsp_path()).expect("load DE421");
    let epoch_tdb_jd = dyncomp_time().tdb_julian_date();
    Box::new(move |sim, time_s: f64| {
        let target_tdb_jd = epoch_tdb_jd + time_s / 86_400.0;
        let (sun_pos, sun_vel) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Sun, target_tdb_jd)
            .expect("Sun state query");
        let (moon_pos, moon_vel) = ephemeris
            .get_earth_centered_state_typed(EphemerisBody::Moon, target_tdb_jd)
            .expect("Moon state query");
        sim.set_source_state(sun_idx, sun_pos.raw_si(), sun_vel.raw_si());
        sim.set_source_state(moon_idx, moon_pos.raw_si(), moon_vel.raw_si());
    })
}

// ── RUN_7A–7D: SH gravity + Sun/Moon third-body (± MET drag) ──────────────

const RUN7_SUN_IDX: usize = 1;
const RUN7_MOON_IDX: usize = 2;

fn build_run7(
    init: &InitialConditions,
    run_dir: &str,
    with_drag: bool,
    case: &str,
) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Gravity controls degree/order from RUN input chain (RUN_7A is always
    // in the chain; RUN_7B/7D add an 8x8 override; 7C/7D add their own
    // input.py for drag wiring without gravity changes).
    let mut grav_files: Vec<PathBuf> = vec![sim_dir.join("Modified_data/grav_controls.py")];
    grav_files.push(sim_dir.join("SET_test/RUN_7A/input.py"));
    if run_dir == "RUN_7B" || run_dir == "RUN_7D" {
        grav_files.push(sim_dir.join("SET_test/RUN_7B/input.py"));
    }
    if run_dir == "RUN_7C" || run_dir == "RUN_7D" {
        grav_files.push(sim_dir.join(format!("SET_test/{run_dir}/input.py")));
    }
    let grav_file_refs: Vec<&std::path::Path> = grav_files.iter().map(|p| p.as_path()).collect();
    let grav_cfg = crate::gravity_control::load_gravity_control(&grav_file_refs);

    // Earth GGM02C SH, Sun mu, and Moon GRAIL150 mu all from committed
    // gravity fixtures (#249).
    let earth_grav = astrodyn::gravity_fixtures::load_ggm02c();
    let mu_sun = astrodyn::gravity_fixtures::load_sun_spherical_mu();
    let mu_moon = astrodyn::gravity_fixtures::load_moon_grail150_mu();

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
            source: GravitySource {
                mu: earth_grav.mu,
                model: GravityModel::SphericalHarmonics(Box::new(earth_grav)),
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: OMEGA_EARTH,
            central: true,
            marker_only: false,
        },
    );
    let sun = sb.add_source("Sun", third_body_source(mu_sun, sun_t0.raw_si()));
    let moon = sb.add_source("Moon", third_body_source(mu_moon, moon_t0.raw_si()));
    debug_assert_eq!(sun, RUN7_SUN_IDX);
    debug_assert_eq!(moon, RUN7_MOON_IDX);

    if with_drag {
        let met_model = MetAtmosphere {
            f10: 128.8,
            f10b: 128.8,
            geo_index: 15.7,
            geo_index_type: GeoIndexType::Ap,
        };
        sb = sb.atmosphere(
            AtmosphereConfig {
                model: AtmosphereModel::Met(met_model),
                r_eq: EARTH.shape.r_eq,
                r_pol: EARTH.shape.r_pol,
                planet_omega: OMEGA_EARTH,
            },
            earth,
        );
    }

    let drag = if with_drag {
        Some(DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: None,
        })
    } else {
        None
    };

    // Drag requires a RotationalState (JEOD_INV: IN.15). 7A/7B (no drag)
    // run as 3-DOF; 7C/7D (drag) carry the reference quaternion.
    let rot = if with_drag {
        Some(super::typed_helpers::rot_typed(&rot_from(init, case)))
    } else {
        None
    };

    let earth_gradient = if grav_cfg.gradient {
        GravityGradient::Compute
    } else {
        GravityGradient::Skip
    };
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot,
        mass: Some(super::typed_helpers::mass_typed(
            &(sphere_mass_properties()),
        )),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(
                    earth,
                    grav_cfg.degree,
                    grav_cfg.order,
                    earth_gradient,
                ),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        drag,
        ..Default::default()
    });
    sb
}

fn build_run7a(init: &InitialConditions) -> SimulationBuilder {
    build_run7(init, "RUN_7A", false, "run7a")
}
fn build_run7b(init: &InitialConditions) -> SimulationBuilder {
    build_run7(init, "RUN_7B", false, "run7b")
}
fn build_run7c(init: &InitialConditions) -> SimulationBuilder {
    build_run7(init, "RUN_7C", true, "run7c")
}
fn build_run7d(init: &InitialConditions) -> SimulationBuilder {
    build_run7(init, "RUN_7D", true, "run7d")
}

/// Pre-step factory for RUN_7*: capture DE421 + epoch TDB once, push
/// Sun/Moon positions to the upcoming step's TDB before `step_until`.
fn run7_pre_step(_init: &InitialConditions) -> PreStepClosure {
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
        sim.set_source_position(RUN7_SUN_IDX, sun_pos.raw_si());
        sim.set_source_position(RUN7_MOON_IDX, moon_pos.raw_si());
    })
}

/// SIM_dyncomp RUN_7A — 4×4 SH + Sun/Moon, no drag.
pub fn run7a_sh4x4_3rd_body() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run7a",
        scenario: build_run7a,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run7a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.13e-2, 1.316e-1, 9.986e-2],
            velocity_m_s: [6.041e-5, 1.206e-4, 1.218e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(run7_pre_step),
    }
}

/// SIM_dyncomp RUN_7B — 8×8 SH + Sun/Moon, no drag.
pub fn run7b_sh8x8_3rd_body() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run7b",
        scenario: build_run7b,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run7b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.28e-1, 2.25e-1, 1.597e-1],
            velocity_m_s: [1.447e-4, 2.25e-4, 1.856e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(run7_pre_step),
    }
}

/// SIM_dyncomp RUN_7C — 4×4 SH + Sun/Moon + MET drag.
pub fn run7c_sh4x4_3rd_body_drag() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run7c",
        scenario: build_run7c,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run7c_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [6.988e-1, 1.038, 8.523e-1],
            velocity_m_s: [7.06e-4, 1.156e-3, 9.565e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(run7_pre_step),
    }
}

/// SIM_dyncomp RUN_7D — 8×8 SH + Sun/Moon + MET drag.
pub fn run7d_sh8x8_3rd_body_drag() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run7d",
        scenario: build_run7d,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run7d_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [7.735e-1, 1.126, 9.118e-1],
            velocity_m_s: [7.898e-4, 1.259e-3, 1.03e-3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(run7_pre_step),
    }
}

// ── RUN_5B / RUN_5C: elliptical, no drag, gradient=true ────────────────────

fn build_run5(init: &InitialConditions, case: &str) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let mu_earth = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mass_props = iss_mass_properties();

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(mu_earth));
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(&(rot_from(init, case)))),
        mass: Some(super::typed_helpers::mass_typed(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth,
                GravityGradient::Compute,
            )],
        },
        ..Default::default()
    });
    sb
}

fn build_run5b(init: &InitialConditions) -> SimulationBuilder {
    build_run5(init, "run5b")
}

fn build_run5c(init: &InitialConditions) -> SimulationBuilder {
    build_run5(init, "run5c")
}

/// SIM_dyncomp RUN_5B — elliptical orbit, point-mass + gravity gradient,
/// 6-DOF (drag off, MET solar mean configuration without effect).
pub fn run5b_atmosphere_mean() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run5b_atmosphere_mean",
        scenario: build_run5b,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run5b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_5C — same physics as RUN_5B with the solar-max MET
/// configuration (no effect on dynamics; drag is off).
pub fn run5c_atmosphere_max() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run5c_atmosphere_max",
        scenario: build_run5c,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run5c_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── RUN_6A / RUN_6B: drag (constant-density / MET) ─────────────────────────

fn met_solar_mean() -> MetAtmosphere {
    MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: GeoIndexType::Ap,
    }
}

fn build_run6_drag(
    init: &InitialConditions,
    case: &str,
    drag_config: DragConfig,
) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mass_props = sphere_mass_properties();

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_pm_with_rnp(earth_mu));
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(met_solar_mean()),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(&(rot_from(init, case)))),
        mass: Some(super::typed_helpers::mass_typed(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        drag: Some(drag_config),
        ..Default::default()
    });
    sb
}

fn build_run6a(init: &InitialConditions) -> SimulationBuilder {
    build_run6_drag(
        init,
        "run6a",
        DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: Some(1.4e-12),
        },
    )
}

fn build_run6b(init: &InitialConditions) -> SimulationBuilder {
    build_run6_drag(
        init,
        "run6b",
        DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: None,
        },
    )
}

/// SIM_dyncomp RUN_6A — sphere mass with constant-density drag (1.4e-12
/// kg/m³). Atmosphere pipeline still runs for wind/co-rotation, but the
/// constant-density override drives the drag force.
pub fn run6a_const_density_drag() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run6a_const_density_drag",
        scenario: build_run6a,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run6a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [4.366e-4, 6.84e-4, 5.325e-4],
            velocity_m_s: [4.942e-7, 7.482e-7, 6.155e-7],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_6B — sphere mass with MET-solar-mean drag.
pub fn run6b_drag() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run6b_drag",
        scenario: build_run6b,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run6b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [7.971e-1, 1.114, 8.945e-1],
            velocity_m_s: [7.861e-4, 1.254e-3, 1.003e-3],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── RUN_10A / RUN_10C / RUN_10D: gravity-gradient torque ───────────────────

fn cylinder_mass_properties() -> MassProperties {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let mass_init = crate::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_cylinder"),
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

fn build_run10(init: &InitialConditions, case: &str) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let earth_mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mass_props = cylinder_mass_properties();

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_mu));
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(&(rot_from(init, case)))),
        mass: Some(super::typed_helpers::mass_typed(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth,
                GravityGradient::Compute,
            )],
        },
        compute_gravity_gradient: true,
        ..Default::default()
    });
    sb
}

fn build_run10a(init: &InitialConditions) -> SimulationBuilder {
    build_run10(init, "run10a")
}

fn build_run10c(init: &InitialConditions) -> SimulationBuilder {
    build_run10(init, "run10c")
}

fn build_run10d(init: &InitialConditions) -> SimulationBuilder {
    build_run10(init, "run10d")
}

/// SIM_dyncomp RUN_10A — gravity-gradient torque, cylinder mass, 6-DOF.
/// Initial attitude is 85° pitch + 1° yaw from LVLH; tests gravity
/// gradient libration.
pub fn run10a_gravity_torque() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run10a_gravity_torque",
        scenario: build_run10a,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run10a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 7.556e-5,
            ang_vel_rad_s: [1e-15, 1.172e-7, 9.301e-8],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_10C — gravity-gradient torque, elliptical orbit,
/// zero initial body rate.
pub fn run10c_gravity_torque_elliptical() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run10c_gravity_torque_elliptical",
        scenario: build_run10c,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run10c_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 7.978e-5,
            ang_vel_rad_s: [1e-15, 1.243e-7, 9.646e-8],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── RUN_5A: MET atmosphere validation (drag off, atmosphere live) ─────────

fn build_run5a_met(init: &InitialConditions) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let mu_earth = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mass_props = iss_mass_properties();

    // RUN_5A: minimum solar activity (F10.7 = 70, Ap = 0)
    let met_model = MetAtmosphere {
        f10: 70.0,
        f10b: 70.0,
        geo_index: 0.0,
        geo_index_type: GeoIndexType::Ap,
    };

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_pm_with_rnp(mu_earth));
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(met_model),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(super::typed_helpers::mass_typed(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth,
                GravityGradient::Compute,
            )],
        },
        ..Default::default()
    });
    sb
}

/// SIM_dyncomp RUN_5A — MET atmosphere live (minimum solar activity),
/// drag off, ISS mass with identity attitude. Validates atmosphere
/// pipeline matches JEOD reference trajectory (atmosphere has no
/// dynamic effect since drag is off).
pub fn run5a_met() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_met_run5a",
        scenario: build_run5a_met,
        reference: CsvReference::AtmosTraj("dyncomp_run5a_atmos_atmos_traj.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [2.5e-6, 2.5e-6, 2.5e-6],
            velocity_m_s: [0.0; 3], // verbatim from existing test (no velocity assert)
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

// ── RUN_6B aero-trajectory variants (sphere + drag, position+velocity only) ─

/// Build a RUN_6B-style sphere/drag scenario reading initial conditions
/// from an aero-trajectory CSV (`dyncomp_run6b_aero_aero_traj.csv` or the
/// rotated-frame variant). Uses `MassProperties::new(1.0)` (1 kg sphere
/// with no explicit inertia) per JEOD's RUN_6B sphere replacement.
fn build_run6b_aero_traj(init: &InitialConditions, t_struct_body: DMat3) -> SimulationBuilder {
    let sim_dir = crate::jeod_inputs::path(SIM_DYNCOMP);
    let dt = crate::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    let mu_earth = astrodyn::gravity_fixtures::load_ggm05c().mu;

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_pm_with_rnp(mu_earth));
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(met_solar_mean()),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: trans_from(init),
        rot: Some(super::typed_helpers::rot_typed(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(super::typed_helpers::mass_typed(
            &(MassProperties::new(1.0)),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        drag: Some(DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: None,
        }),
        t_struct_body,
        ..Default::default()
    });
    sb
}

fn build_run6b_aero_identity(init: &InitialConditions) -> SimulationBuilder {
    build_run6b_aero_traj(init, DMat3::IDENTITY)
}

fn build_run6b_aero_rotated(init: &InitialConditions) -> SimulationBuilder {
    // 15° rotation about [1,1,1]/√3 — matches DYNCOMP_AERO_ROT_SNIPPET in
    // generate_references.sh. For ballistic drag on a sphere the inertial
    // force is rotation-invariant, so the trajectory must match the
    // identity case; any divergence indicates a frame-transform bug.
    let eigen_angle = 15.0_f64.to_radians();
    let eigen_axis = DVec3::new(1.0, 1.0, 1.0).normalize();
    let q_struct_body = JeodQuat::left_quat_from_eigen_rotation(eigen_angle, eigen_axis);
    build_run6b_aero_traj(init, q_struct_body.left_quat_to_transformation())
}

/// SIM_dyncomp RUN_6B — sphere + drag, identity structural frame.
/// Validates the drag pipeline against the aero-trajectory CSV
/// (position+velocity only; aero_force is not exposed on
/// `VehicleOutput`).
pub fn run6b_drag_aero_traj() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_drag_run6b",
        scenario: build_run6b_aero_identity,
        reference: CsvReference::AeroTraj("dyncomp_run6b_aero_aero_traj.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.12, 1.12, 1.12],
            velocity_m_s: [1.254e-3, 1.254e-3, 1.254e-3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_6B with a 15° rotation about `[1,1,1]/√3` applied to
/// the structural-to-body transform. Should match the identity case
/// since ballistic drag on a sphere is rotation-invariant.
pub fn run6b_drag_rotated_struct() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_drag_run6b_rotated",
        scenario: build_run6b_aero_rotated,
        reference: CsvReference::AeroTraj("dyncomp_run6b_rot_aero_traj.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [0.798, 1.114, 0.895],
            velocity_m_s: [7.861e-4, 1.254e-3, 1.003e-3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_dyncomp RUN_10D — gravity-gradient torque, elliptical orbit,
/// non-zero initial body rate.
pub fn run10d_gravity_torque_elliptical_rate() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run10d_gravity_torque_elliptical_rate",
        scenario: build_run10d,
        reference: CsvReference::Dyncomp6Dof("dyncomp_run10d_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 1.106e-4,
            ang_vel_rad_s: [1e-15, 1.825e-7, 1.196e-7],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}
