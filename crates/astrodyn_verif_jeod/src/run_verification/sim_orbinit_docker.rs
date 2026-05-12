//! `VerificationCase` constructors for the SIM_orbinit Docker scenarios
//! (`tier3_sim_orbinit_docker`).
//!
//! Each RUN reproduces JEOD's orbital-element-to-inertial initialization
//! pipeline: read the body-init fixture (originally extracted from
//! `Modified_data/<vehicle>/<init_name>.py`), convert to an inertial
//! `TranslationalState` via [`astrodyn::init_from_mean_anomaly`], and
//! (for the pfix variants) rotate through the inertial↔pfix matrix at
//! the SIM epoch (2005-07-28 10:09:59 UT1). The resulting state is fed
//! to a point-mass-Earth `Simulation` configured with the same `dt` as
//! the orbinit_edge family; the synthetic-cadence checkpoint exercises
//! the integrator/frame-propagation stages end-to-end so the bevy
//! parity wrapper drives bit-identity at the same propagation depth as
//! the matching tier3 runner test.
//!
//! Five recipes correspond to JEOD's RUN list:
//!   * RUN_0001 — ISS orbital elements (set01) in `Earth.inertial`;
//!   * RUN_0101 — STS-114 orbital elements (set01) in `Earth.inertial`;
//!   * RUN_0201 — ISS orbital elements (set01) in `Earth.pfix`;
//!   * RUN_0301 — STS-114 orbital elements (set01) in `Earth.pfix`;
//!   * RUN_0401 — STS-114 direct Cartesian state in `Earth.inertial`.
//!
//! The orbital-element-to-Cartesian conversion is the substance of this
//! test; it runs inside every scenario factory rather than being
//! pre-baked, so the parity wrapper exercises `init_from_mean_anomaly`
//! (and the inertial↔pfix RNP rotation for the pfix variants) on both
//! the runner and Bevy paths. Bit-identity at the synthetic checkpoint
//! implies the conversion produced the same f64 bit-pattern on both
//! sides.
//!
//! Each recipe pairs with [`CsvReference::SyntheticTimes`]; the tier3
//! sibling compares the recipe's pre-propagation initial state (the
//! orbital-element-to-Cartesian conversion's output) against the
//! JEOD-logged t=0 row, then propagates one synthetic-cadence tick so
//! the pipeline runs end-to-end. The parity trait drives bit-identity
//! at the synthetic record. Initial-fixture sources live in
//! `crates/astrodyn_verif_jeod/test_data/body_init/{iss,sts_114}.json`,
//! parsed once per call via [`astrodyn_verif_jeod_fixtures::orbital_init`].

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    calendar_to_tjt, compute_t_parent_this_from_tjt, default_leap_second_table,
    init_from_mean_anomaly, ut1_to_gmst_seconds, CalendarDate, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, RotationModel,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use astrodyn_verif_jeod_fixtures::orbital_init::{load_orbital_init, load_trans_state};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Integrator step size shared by every recipe. Matches the value the
/// pre-recipe tier3 file used so the SyntheticTimes cadence drives
/// identical integration ticks across runner and bevy.
const DT_S: f64 = 10.0;

/// One propagation step exercises the integrator + frame-propagation
/// stages end-to-end. The orbinit Docker CSVs only log t=0, so the
/// tier3 sibling's substance is the initialization conversion rather
/// than long-horizon propagation; a single tick is enough to drive the
/// `Simulation` pipeline through.
const NUM_STEPS: usize = 1;

/// SIM_orbinit epoch components (from `Modified_data/earth.py`):
/// `set_date_and_time(2005, 7, 28, 10, 9, 59)` with `initializer = "UT1"`.
const ORBINIT_YEAR: i32 = 2005;
const ORBINIT_MONTH: i32 = 7;
const ORBINIT_DAY: i32 = 28;
const ORBINIT_HOUR: i32 = 10;
const ORBINIT_MINUTE: i32 = 9;
const ORBINIT_SECOND: f64 = 59.0;

fn load_mu_earth() -> f64 {
    // Cache the decoded `mu` for the lifetime of the test process —
    // every `build_*` call would otherwise re-parse the full GGM05C
    // coefficient table (~12 KiB) just to read a single scalar.
    static MU_EARTH: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MU_EARTH.get_or_init(|| astrodyn::gravity_fixtures::load_ggm05c().mu)
}

/// Compute the inertial-to-planet-fixed rotation matrix at the
/// SIM_orbinit epoch. SIM_orbinit uses `initializer = "UT1"` with
/// `set_date_and_time(2005, 7, 28, 10, 9, 59)` and
/// `earth.rnp.enable_polar = False`. Following JEOD
/// `rnp.update_rnp(tt, gmst, ut1)`, the rotation uses
/// precession+nutation (via TT) and GAST (via GMST). Cached so each
/// recipe pays the cost once across the whole test process.
fn t_inertial_pfix_at_epoch() -> DMat3 {
    static T: std::sync::OnceLock<DMat3> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let ut1_cal = CalendarDate::new(
            ORBINIT_YEAR,
            ORBINIT_MONTH,
            ORBINIT_DAY,
            ORBINIT_HOUR,
            ORBINIT_MINUTE,
            ORBINIT_SECOND,
        );
        let ut1_tjt = calendar_to_tjt(&ut1_cal);

        // SIM_orbinit doesn't override UT1-UTC, so UTC_tjt ≈ UT1_tjt
        // and TAI_tjt = UTC_tjt + (TAI-UTC)/86400.
        let leap = default_leap_second_table();
        let tai_utc_s = leap.tai_utc_at_utc_tjt(ut1_tjt);
        let tai_tjt = ut1_tjt + tai_utc_s / 86_400.0;

        // TT = TAI + 32.184 s.
        let tt_tjt = tai_tjt + 32.184 / 86_400.0;

        // GMST seconds since J2000 noon UT1, computed from UT1 directly
        // (matches `SimulationTime::recompute_derived()`).
        let du = ut1_tjt - 11_544.5;
        let gmst_seconds = ut1_to_gmst_seconds(du);

        // SIM_orbinit sets `enable_polar = False`; no polar motion.
        compute_t_parent_this_from_tjt(gmst_seconds, tt_tjt)
    })
}

/// Materialize a JEOD orbital-element fixture into an inertial-frame
/// translational state, reproducing the conversion the original
/// orbinit_docker tier3 file did inline.
///
/// For `time_periapsis`-parameterized orbits, mean anomaly is computed
/// as `M = t_peri * sqrt(mu/a^3)`
/// (JEOD `dyn_body_init_orbit.cc:295`). For pfix orbits, the orbital
/// elements are interpreted in the planet-fixed frame; the state is
/// built there and rotated to inertial via `T_pfix_to_inertial` (no
/// `ω × r` term — JEOD `dyn_body_init_orbit.cc:331-332` rotates
/// position and velocity as pure 3-vectors).
fn orbital_element_state(vehicle: &str, init_name: &str, mu_earth: f64) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let t_peri = init.time_periapsis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set01 expected time_periapsis in the fixture",)
    });
    let a = init.semi_major_axis;
    let n = (mu_earth / (a * a * a)).sqrt();
    let mean_anomaly = n * t_peri;

    let state_ref = init_from_mean_anomaly(
        init.semi_major_axis,
        init.eccentricity,
        init.inclination,
        init.ascending_node,
        init.arg_periapsis,
        mean_anomaly,
        mu_earth,
    );

    match init.reference_frame.as_str() {
        "Earth.inertial" => state_ref,
        "Earth.pfix" => {
            let t_inertial_pfix = t_inertial_pfix_at_epoch();
            let t_pfix_inertial = t_inertial_pfix.transpose();
            TranslationalState {
                position: t_pfix_inertial * state_ref.position,
                velocity: t_pfix_inertial * state_ref.velocity,
            }
        }
        other => panic!("{vehicle}/{init_name}: unsupported reference_frame '{other}'"),
    }
}

/// Materialize a JEOD direct-Cartesian fixture (RUN_0401 only) into an
/// inertial-frame translational state. The fixture is a pass-through:
/// `position`/`velocity` arrays are taken verbatim.
fn trans_state(vehicle: &str, init_name: &str) -> TranslationalState {
    let trans = load_trans_state(vehicle, init_name);
    TranslationalState {
        position: DVec3::from_array(trans.position),
        velocity: DVec3::from_array(trans.velocity),
    }
}

/// Shared scenario builder for every recipe. Parameterised by:
///   * `mu_earth` — Earth's gravitational parameter (point-mass);
///   * `body` — the vehicle's initial translational state in the
///     RootInertial frame.
fn build_orbinit_docker(mu_earth: f64, body: TranslationalState) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, DT_S);
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
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
        },
    );
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&body),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        ..Default::default()
    });
    sb
}

/// Recipes opt out of every runner-vs-JEOD tolerance group because they
/// pair with [`CsvReference::SyntheticTimes`]; the tier3 sibling asserts
/// against the JEOD-logged t=0 row directly, and the parity trait drives
/// bit-identity at every synthetic record.
fn synthetic_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

fn build_run_0001(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = orbital_element_state("ISS", "trans_Orbit_inertial_body_set01", mu);
    build_orbinit_docker(mu, state)
}

fn build_run_0101(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = orbital_element_state("STS_114", "trans_Orbit_inertial_body_set01", mu);
    build_orbinit_docker(mu, state)
}

fn build_run_0201(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = orbital_element_state("ISS", "trans_Orbit_pfix_body_set01", mu);
    build_orbinit_docker(mu, state)
}

fn build_run_0301(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = orbital_element_state("STS_114", "trans_Orbit_pfix_body_set01", mu);
    build_orbinit_docker(mu, state)
}

fn build_run_0401(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = trans_state("STS_114", "trans_TransState_inertial_body");
    build_orbinit_docker(mu, state)
}

/// RUN_0001: ISS orbital elements (set01) in `Earth.inertial`.
pub fn run_0001() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0001",
        scenario: build_run_0001,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: synthetic_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// RUN_0101: STS-114 orbital elements (set01) in `Earth.inertial`.
pub fn run_0101() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0101",
        scenario: build_run_0101,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: synthetic_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// RUN_0201: ISS orbital elements (set01) in `Earth.pfix`. The pfix
/// branch applies the inertial↔pfix rotation at the SIM epoch.
pub fn run_0201() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0201",
        scenario: build_run_0201,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: synthetic_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// RUN_0301: STS-114 orbital elements (set01) in `Earth.pfix`.
pub fn run_0301() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0301",
        scenario: build_run_0301,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: synthetic_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// RUN_0401: STS-114 direct Cartesian state (`DynBodyInitTransState`)
/// in `Earth.inertial`. The fixture is a pass-through: the body state
/// is the literal position/velocity arrays from
/// `trans_TransState_inertial_body`.
pub fn run_0401() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0401",
        scenario: build_run_0401,
        reference: CsvReference::SyntheticTimes {
            dt: DT_S,
            num_steps: NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: synthetic_tolerances(),
        extras: None,
        pre_step: None,
    }
}
