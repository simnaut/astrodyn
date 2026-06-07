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
//! The recipes correspond to JEOD's RUN list:
//!   * RUN_0001 — ISS orbital elements (set01) in `Earth.inertial`;
//!   * RUN_0002 — ISS orbital elements (set02, mean anomaly) in `Earth.inertial`;
//!   * RUN_0003 — ISS orbital elements (set03, semi-latus rectum + true anomaly) in `Earth.inertial`;
//!   * RUN_0004 — ISS orbital elements (set04, apo/peri altitudes + true anomaly) in `Earth.inertial`;
//!   * RUN_0005 — ISS orbital elements (set05, apo/peri altitudes + time periapsis) in `Earth.inertial`;
//!   * RUN_0006 — ISS orbital elements (set06, arg-latitude + radial-vel) in `Earth.inertial`;
//!   * RUN_0010 — ISS orbital elements (set10, sma/ecc + true anomaly) in `Earth.inertial`;
//!   * RUN_0011 — ISS orbital elements (set11, apo/peri altitudes + true anomaly) in `Earth.inertial`;
//!   * RUN_0106 — STS-114 orbital elements (set06, arg-latitude + radial-vel) in `Earth.inertial`;
//!   * RUN_0110 — STS-114 orbital elements (set10, sma/ecc + true anomaly) in `Earth.inertial`;
//!   * RUN_0111 — STS-114 orbital elements (set11, apo/peri altitudes + true anomaly) in `Earth.inertial`;
//!   * RUN_0101 — STS-114 orbital elements (set01) in `Earth.inertial`;
//!   * RUN_0102 — STS-114 orbital elements (set02, mean anomaly) in `Earth.inertial`;
//!   * RUN_0103 — STS-114 orbital elements (set03, semi-latus rectum + true anomaly) in `Earth.inertial`;
//!   * RUN_0104 — STS-114 orbital elements (set04, apo/peri altitudes + true anomaly) in `Earth.inertial`;
//!   * RUN_0105 — STS-114 orbital elements (set05, apo/peri altitudes + time periapsis) in `Earth.inertial`;
//!   * RUN_0201 — ISS orbital elements (set01) in `Earth.pfix`;
//!   * RUN_0301 — STS-114 orbital elements (set01) in `Earth.pfix`;
//!   * RUN_0202 — ISS orbital elements (set02, mean anomaly) in `Earth.pfix`;
//!   * RUN_0302 — STS-114 orbital elements (set02, mean anomaly) in `Earth.pfix`;
//!   * RUN_0203 — ISS orbital elements (set03, semi-latus rectum + true anomaly) in `Earth.pfix`;
//!   * RUN_0303 — STS-114 orbital elements (set03, semi-latus rectum + true anomaly) in `Earth.pfix`;
//!   * RUN_0204 — ISS orbital elements (set04, apo/peri altitudes + true anomaly) in `Earth.pfix`;
//!   * RUN_0304 — STS-114 orbital elements (set04, apo/peri altitudes + true anomaly) in `Earth.pfix`;
//!   * RUN_0205 — ISS orbital elements (set05, apo/peri altitudes + time periapsis) in `Earth.pfix`;
//!   * RUN_0305 — STS-114 orbital elements (set05, apo/peri altitudes + time periapsis) in `Earth.pfix`;
//!   * RUN_0206 — ISS orbital elements (set06, arg-latitude + radial-vel) in `Earth.pfix`;
//!   * RUN_0306 — STS-114 orbital elements (set06, arg-latitude + radial-vel) in `Earth.pfix`;
//!   * RUN_0210 — ISS orbital elements (set10, sma/ecc + true anomaly) in `Earth.pfix`;
//!   * RUN_0310 — STS-114 orbital elements (set10, sma/ecc + true anomaly) in `Earth.pfix`;
//!   * RUN_0211 — ISS orbital elements (set11, apo/peri altitudes + true anomaly) in `Earth.pfix`;
//!   * RUN_0311 — STS-114 orbital elements (set11, apo/peri altitudes + true anomaly) in `Earth.pfix`;
//!   * RUN_0401 — STS-114 direct Cartesian state in `Earth.inertial`;
//!   * RUN_0400 — ISS direct Cartesian state in `Earth.inertial`;
//!   * RUN_0410 — ISS direct Cartesian state in `Earth.pfix`;
//!   * RUN_0411 — STS-114 direct Cartesian state in `Earth.pfix`.
//!
//! set01 and set02 resolve to [`init_from_mean_anomaly`]; set01 derives
//! `M = t_peri·√(μ/a³)` from the fixture's `time_periapsis`, while set02
//! reads the fixture's `mean_anomaly` field directly. set03
//! (`SlrEccIncAscnodeArgperTanom`) resolves to
//! [`init_from_semi_latus_rectum_true_anomaly`], using the fixture's
//! `semi_latus_rectum` (as `semiparam`) and `true_anomaly` directly.
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

use super::fixtures::load_mu_earth;
use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    calendar_to_tjt, compute_body_lvlh_frame, compute_quaternion_from_euler_angles_typed,
    compute_t_parent_this_from_tjt, default_leap_second_table, init_from_altitudes_time_periapsis,
    init_from_altitudes_true_anomaly, init_from_arg_latitude_radial_vel, init_from_mean_anomaly,
    init_from_orbital_elements, init_from_semi_latus_rectum_true_anomaly, ut1_to_gmst_seconds,
    BodyAction, CalendarDate, EulerSequence, GravityControl, GravityControls, GravityGradient,
    GravityModel, GravitySource, GravitySourceEntry, JeodQuat, LvlhAngularVelocityFrame,
    RefFrameRot, RefFrameState, RefFrameTrans, RotationModel, RotationalState, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig, EARTH,
};
use astrodyn_verif_jeod_fixtures::orbital_init::{load_orbital_init, load_trans_state};
use glam::{DMat3, DVec3};
use uom::si::angle::degree;
use uom::si::f64::{Angle, Time};
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
    let a = init.semi_major_axis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set01 expected semi_major_axis in the fixture")
    });
    let n = (mu_earth / (a * a * a)).sqrt();
    let mean_anomaly = n * t_peri;

    let state_ref = init_from_mean_anomaly(
        a,
        require_eccentricity(vehicle, init_name, &init),
        init.inclination,
        init.ascending_node,
        require_arg_periapsis(vehicle, init_name, &init),
        mean_anomaly,
        mu_earth,
    );

    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Place an orbital state built in `init.reference_frame` into the
/// RootInertial frame. `Earth.inertial` passes through; `Earth.pfix`
/// rotates position and velocity by `T_pfix_to_inertial` at the SIM
/// epoch (no `ω × r` term — JEOD `dyn_body_init_orbit.cc:331-332`
/// rotates them as pure 3-vectors). Any other frame fails loudly.
fn resolve_reference_frame(
    vehicle: &str,
    init_name: &str,
    reference_frame: &str,
    state_ref: TranslationalState,
) -> TranslationalState {
    match reference_frame {
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

/// Materialize a JEOD set02 (`SmaEccIncAscnodeArgperMeanAnomaly`) fixture
/// into an inertial-frame translational state. Unlike set01, the mean
/// anomaly is supplied directly by the deck (fixture field `mean_anomaly`,
/// stored in radians by `extract_body_init`), so this is exactly
/// [`init_from_mean_anomaly`] with no time-periapsis derivation. The
/// orbital elements are interpreted in `init.reference_frame`:
/// `Earth.inertial` decks pass through, `Earth.pfix` decks are rotated
/// to inertial via [`resolve_reference_frame`].
fn mean_anomaly_element_state(vehicle: &str, init_name: &str, mu_earth: f64) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let mean_anomaly = init.mean_anomaly.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set02 expected mean_anomaly in the fixture")
    });
    let a = init.semi_major_axis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set02 expected semi_major_axis in the fixture")
    });
    let state_ref = init_from_mean_anomaly(
        a,
        require_eccentricity(vehicle, init_name, &init),
        init.inclination,
        init.ascending_node,
        require_arg_periapsis(vehicle, init_name, &init),
        mean_anomaly,
        mu_earth,
    );
    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Materialize a JEOD set03 (`SlrEccIncAscnodeArgperTanom`) fixture into an
/// inertial-frame translational state. set03 parameterizes the orbit by
/// **semi-latus rectum** (`semi_latus_rectum`) + **true anomaly**
/// (`true_anomaly`), both stored in SI by `extract_body_init` (m, rad). JEOD
/// uses the deck's semi-latus rectum verbatim as `elem.semiparam` (the
/// `semi_major_axis * (1 - e²)` derivation runs only for sma-parameterized
/// sets), so this is exactly [`init_from_semi_latus_rectum_true_anomaly`]
/// with no sma round-trip. The orbital elements are interpreted in
/// `init.reference_frame`: `Earth.inertial` decks pass through,
/// `Earth.pfix` decks are rotated to inertial via
/// [`resolve_reference_frame`].
fn true_anomaly_element_state(vehicle: &str, init_name: &str, mu_earth: f64) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let p = init.semi_latus_rectum.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set03 expected semi_latus_rectum in the fixture")
    });
    let true_anomaly = init.true_anomaly.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set03 expected true_anomaly in the fixture")
    });
    let state_ref = init_from_semi_latus_rectum_true_anomaly(
        p,
        require_eccentricity(vehicle, init_name, &init),
        init.inclination,
        init.ascending_node,
        require_arg_periapsis(vehicle, init_name, &init),
        true_anomaly,
        mu_earth,
    );
    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Unwrap the fixture's directly-supplied eccentricity (sma/slr sets carry it;
/// the altitude sets 04/05/11 and set06 derive it instead and must not call this).
fn require_eccentricity(
    vehicle: &str,
    init_name: &str,
    init: &astrodyn_verif_jeod_fixtures::orbital_init::OrbitalInitData,
) -> f64 {
    init.eccentricity.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: expected eccentricity in the fixture (sma/slr set)")
    })
}

/// Unwrap the fixture's directly-supplied argument of periapsis (every set
/// except set06 carries it; set06 derives it from `arg_latitude − true_anomaly`
/// inside [`arg_latitude_radial_vel_state`] and must not call this).
fn require_arg_periapsis(
    vehicle: &str,
    init_name: &str,
    init: &astrodyn_verif_jeod_fixtures::orbital_init::OrbitalInitData,
) -> f64 {
    init.arg_periapsis
        .unwrap_or_else(|| panic!("{vehicle}/{init_name}: expected arg_periapsis in the fixture"))
}

/// Materialize a JEOD set04 (`IncAscnodeAltperAltapoArgperTanom`) fixture into
/// an inertial-frame translational state. set04 parameterizes the orbit by
/// apo/peri **altitudes** above Earth's equatorial radius + **true anomaly**.
/// JEOD derives `a = r_eq + ½(alt_apo + alt_peri)` and
/// `e = (alt_apo − alt_peri)/(2a)` (`dyn_body_init_orbit.cc:277-283`), then
/// resolves the true anomaly. `r_eq` is JEOD's Earth equatorial radius
/// (`EARTH.shape.r_eq()` = `1000·6378.137` m), the same value
/// `environment/planet/data/src/earth.cc` assigns `Planet::r_eq`. The orbital
/// elements are interpreted in `init.reference_frame`: `Earth.inertial` decks
/// pass through, `Earth.pfix` decks are rotated to inertial via
/// [`resolve_reference_frame`]. set11 (`CaseEleven`) shares this converter.
fn altitudes_true_anomaly_state(
    vehicle: &str,
    init_name: &str,
    mu_earth: f64,
) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let alt_apo = init.alt_apoapsis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set04 expected alt_apoapsis in the fixture")
    });
    let alt_peri = init.alt_periapsis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set04 expected alt_periapsis in the fixture")
    });
    let true_anomaly = init.true_anomaly.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set04 expected true_anomaly in the fixture")
    });
    let state_ref = init_from_altitudes_true_anomaly(
        EARTH.shape.r_eq(),
        alt_apo,
        alt_peri,
        init.inclination,
        init.ascending_node,
        require_arg_periapsis(vehicle, init_name, &init),
        true_anomaly,
        mu_earth,
    );
    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Materialize a JEOD set05 (`IncAscnodeAltperAltapoArgperTimeperi`) fixture
/// into an inertial-frame translational state. set05 parameterizes the orbit
/// by apo/peri **altitudes** + **time since periapsis passage**. JEOD derives
/// `a`/`e` from the altitudes exactly as set04, then maps `time_periapsis` to
/// mean anomaly (`M = t_peri·√(μ/a)/a`, `dyn_body_init_orbit.cc:293-295`). The
/// set05 decks are `Earth.inertial` only.
fn altitudes_time_periapsis_state(
    vehicle: &str,
    init_name: &str,
    mu_earth: f64,
) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let alt_apo = init.alt_apoapsis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set05 expected alt_apoapsis in the fixture")
    });
    let alt_peri = init.alt_periapsis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set05 expected alt_periapsis in the fixture")
    });
    let t_peri = init.time_periapsis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set05 expected time_periapsis in the fixture")
    });
    let state_ref = init_from_altitudes_time_periapsis(
        EARTH.shape.r_eq(),
        alt_apo,
        alt_peri,
        init.inclination,
        init.ascending_node,
        require_arg_periapsis(vehicle, init_name, &init),
        t_peri,
        mu_earth,
    );
    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Materialize a JEOD set06 (`SmaIncAscnodeArglatRadRadvel`) fixture into an
/// inertial-frame translational state. set06 parameterizes the orbit by
/// **semi-major axis**, inclination, ascending node, **argument of latitude**,
/// **orbital radius**, and **radial velocity**. JEOD recovers `(e, ν, ω)` from
/// the radius / radial-velocity pair via the eccentric-anomaly identities and
/// then resolves the sma + true-anomaly shape
/// (`dyn_body_init_orbit.cc:221-261`). The orbital elements are interpreted in
/// `init.reference_frame`: `Earth.inertial` decks pass through, `Earth.pfix`
/// decks are rotated to inertial via [`resolve_reference_frame`].
fn arg_latitude_radial_vel_state(
    vehicle: &str,
    init_name: &str,
    mu_earth: f64,
) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let a = init.semi_major_axis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set06 expected semi_major_axis in the fixture")
    });
    let arg_latitude = init.arg_latitude.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set06 expected arg_latitude in the fixture")
    });
    let orb_radius = init.orb_radius.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set06 expected orb_radius in the fixture")
    });
    let radial_vel = init.radial_vel.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set06 expected radial_vel in the fixture")
    });
    let state_ref = init_from_arg_latitude_radial_vel(
        a,
        init.inclination,
        init.ascending_node,
        arg_latitude,
        orb_radius,
        radial_vel,
        mu_earth,
    );
    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Materialize a JEOD set10 (`SmaEccIncAscnodeArgperTanom`) fixture into an
/// inertial-frame translational state. set10 parameterizes the orbit by
/// **semi-major axis** + **eccentricity** + **true anomaly** — JEOD's
/// `ShapeSemiMajorAxis` + `LocationTrueAnom` path
/// (`dyn_body_init_orbit.cc:256-261`), which derives `semiparam = a·(1−e²)` and
/// resolves the true anomaly directly. This is exactly
/// [`init_from_orbital_elements`]. The orbital elements are interpreted in
/// `init.reference_frame`: `Earth.inertial` decks pass through, `Earth.pfix`
/// decks are rotated to inertial via [`resolve_reference_frame`].
fn true_anomaly_sma_state(vehicle: &str, init_name: &str, mu_earth: f64) -> TranslationalState {
    let init = load_orbital_init(vehicle, init_name);
    let a = init.semi_major_axis.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set10 expected semi_major_axis in the fixture")
    });
    let true_anomaly = init.true_anomaly.unwrap_or_else(|| {
        panic!("{vehicle}/{init_name}: set10 expected true_anomaly in the fixture")
    });
    let state_ref = init_from_orbital_elements(
        a,
        require_eccentricity(vehicle, init_name, &init),
        init.inclination,
        init.ascending_node,
        require_arg_periapsis(vehicle, init_name, &init),
        true_anomaly,
        mu_earth,
    );
    resolve_reference_frame(vehicle, init_name, &init.reference_frame, state_ref)
}

/// Materialize a JEOD direct-Cartesian (`DynBodyInitTransState`) fixture
/// into an inertial-frame translational state.
///
/// `Earth.inertial` fixtures are a pass-through: `position`/`velocity` are
/// taken verbatim. `Earth.pfix` fixtures are expressed in the rotating
/// planet-fixed frame and composed into inertial through the full
/// reference-frame relation JEOD applies for a direct trans-state init
/// (`DynBodyInit::apply_user_inputs` → `RefFrameState` composition).
///
/// Unlike the orbital-element pfix path — which rotates the elements'
/// Cartesian image as pure 3-vectors (JEOD `dyn_body_init_orbit.cc`
/// overrides the composition) — the direct trans-state path goes through
/// `compute_relative_state`, so the velocity carries the planet-rotation
/// `ω × r` term. With A = inertial, B = pfix, C = vehicle and the pfix
/// frame's `ang_vel_this = [0, 0, planet_omega]` (JEOD `planet_rnp.cc`),
/// JEOD's `RefFrameState::incr_left` reduces (zero parent-frame
/// translation) to:
///   r_inertial = T_pfix_inertial · r_pfix
///   v_inertial = T_pfix_inertial · (v_pfix + ω_pfix × r_pfix)
/// where the `ω × r` cross product is evaluated in the pfix frame before
/// the rotation, matching JEOD's order of operations. Any other frame
/// fails loudly.
fn trans_state(vehicle: &str, init_name: &str) -> TranslationalState {
    let trans = load_trans_state(vehicle, init_name);
    let position = DVec3::from_array(trans.position);
    let velocity = DVec3::from_array(trans.velocity);
    match trans.reference_frame.as_str() {
        "Earth.inertial" => TranslationalState { position, velocity },
        "Earth.pfix" => {
            let t_inertial_pfix = t_inertial_pfix_at_epoch();
            let t_pfix_inertial = t_inertial_pfix.transpose();
            let omega_pfix = DVec3::new(0.0, 0.0, EARTH.omega);
            let velocity_in_pfix = velocity + omega_pfix.cross(position);
            TranslationalState {
                position: t_pfix_inertial * position,
                velocity: t_pfix_inertial * velocity_in_pfix,
            }
        }
        other => panic!("{vehicle}/{init_name}: unsupported reference_frame '{other}'"),
    }
}

/// Shared scenario builder for every recipe. Parameterised by:
///   * `mu_earth` — Earth's gravitational parameter (point-mass);
///   * `body` — the vehicle's initial translational state in the
///     RootInertial frame.
fn build_orbinit_docker(mu_earth: f64, body: TranslationalState) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, DT_S);
    let _earth = sb.add_source(
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
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("sim-orbinit-docker-0")
    });
    sb
}

/// ISS mass properties used by SIM_orbinit's rotational-init RUNs.
///
/// Source-cited literals from `Modified_data/ISS/mass.py`
/// (`set_ISS_mass`): `mass = 100000.0` kg, CG `position =
/// [-10.201, 0.206, 2.558]` m, and a body-frame (CG-centred) diagonal
/// inertia `diag(7e12, 12e12, 10e12)` kg·m². The deck sets
/// `inertia_spec = Body` and an identity `pt_orientation`
/// (`StructToBody`), so the inertia is already expressed about the CG
/// in a body frame aligned with structure — a direct
/// [`MassProperties::with_inertia`](astrodyn::MassProperties) (`t_parent_this = identity`). These
/// are JEOD *source* initial conditions (permitted by the computational-
/// independence rule), not values read back from JEOD output.
fn iss_mass_properties() -> astrodyn::MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(7.0e12, 0.0, 0.0),
        DVec3::new(0.0, 12.0e12, 0.0),
        DVec3::new(0.0, 0.0, 10.0e12),
    );
    astrodyn::MassProperties::with_inertia(100_000.0, inertia, DVec3::new(-10.201, 0.206, 2.558))
}

/// Scenario builder for the rotational-init RUNs (RUN_1230 / RUN_2100).
///
/// Identical point-mass-Earth setup to [`build_orbinit_docker`], but the
/// vehicle is initialized 6-DOF: the supplied `rot` rotational state
/// (computed by a [`BodyAction`] rotational initializer) and the ISS
/// mass properties are attached alongside the translational state so the
/// integrator + frame propagation exercise the attitude/rate path
/// end-to-end. The mass properties match JEOD's `set_ISS_mass` deck so
/// the torque-free rate integration runs against the same inertia.
fn build_orbinit_docker_rot(
    mu_earth: f64,
    body: TranslationalState,
    rot: RotationalState,
) -> SimulationBuilder {
    let mut sb = build_orbinit_docker(mu_earth, body);
    sb.bodies[0].rot = Some(super::typed_helpers::rot_typed(&rot));
    sb.bodies[0].mass = Some(super::typed_helpers::mass_typed(&iss_mass_properties()));
    sb
}

/// RUN_2100 rotational state: direct inertial attitude + angular rate
/// (JEOD `DynBodyInitRotState`, `reference_ref_frame = Earth.inertial`).
///
/// The attitude is the Yaw-Pitch-Roll (JEOD `Yaw_Pitch_Roll` =
/// `EulerSequence::ZYX`, both discriminant 5) Euler triple
/// `[77.590713, -30.604895, -46.100115]` deg from
/// `Modified_data/ISS/att_RotState_inertial_body.py`. JEOD copies the
/// computed `T_parent_this` straight onto the user frame
/// (`dyn_body_init.cc:300-302`), and because the reference frame
/// `Earth.inertial` *is* ISS's integration frame the user frame's
/// rotational state is the body state with no further composition — so
/// the Euler-derived quaternion is the body's inertial attitude
/// directly.
///
/// The angular rate is the inertial body rate from
/// `Modified_data/ISS/rate_RotState_inertial_body.py`:
/// `[w_iss_lvlh[0], w_iss_lvlh[1] + w_lvlh, w_iss_lvlh[2]]` deg/s with
/// `w_iss_lvlh = [0.002, 0.006, -0.003]` (`iss_rate_def.py`) and
/// `w_lvlh = -0.06556131568278` (`lvlh_rate_def.py`). The deck leaves
/// `rate_in_parent` unset (JEOD default `false`,
/// `dyn_body_init.hh:176`), so `ang_velocity` is the body-frame rate
/// (`ang_vel_this`, `dyn_body_init.cc:310-312`) — used verbatim as
/// `ang_vel_body`.
fn attitude_rate_inertial_state() -> RotationalState {
    // Euler angles (deg) — Modified_data/ISS/att_RotState_inertial_body.py.
    let angles = [
        Angle::new::<degree>(77.590_713),
        Angle::new::<degree>(-30.604_895),
        Angle::new::<degree>(-46.100_115),
    ];
    let quaternion = compute_quaternion_from_euler_angles_typed(angles, EulerSequence::ZYX).inner();

    // Body rate wrt inertial (deg/s) — Modified_data/ISS rate decks.
    // w_iss_lvlh = [0.002, 0.006, -0.003]; w_lvlh = -0.06556131568278.
    let w_iss_lvlh = [0.002, 0.006, -0.003];
    let w_lvlh = -0.065_561_315_682_78;
    let ang_vel_deg = DVec3::new(w_iss_lvlh[0], w_iss_lvlh[1] + w_lvlh, w_iss_lvlh[2]);
    let ang_vel_body = DVec3::new(
        Angle::new::<degree>(ang_vel_deg.x).get::<uom::si::angle::radian>(),
        Angle::new::<degree>(ang_vel_deg.y).get::<uom::si::angle::radian>(),
        Angle::new::<degree>(ang_vel_deg.z).get::<uom::si::angle::radian>(),
    );

    BodyAction::InitRot {
        quaternion,
        ang_vel_body,
    }
    .apply_rotational()
    .expect("BodyAction::InitRot is a rotational action and must yield Some(RotationalState)")
}

/// RUN_1230 rotational state: LVLH-relative attitude + angular rate
/// (JEOD `DynBodyInitLvlhRotState`, planet Earth).
///
/// The body is aligned with the reference orbit's LVLH frame: the
/// Pitch-Yaw-Roll (JEOD `Pitch_Yaw_Roll` = `EulerSequence::YZX`, both
/// discriminant 2) Euler triple is `[0, 0, 0]` deg from
/// `Modified_data/ISS/rot_LvlhRotState_lvlh_body.py`, so the LVLH→body
/// quaternion is identity. The LVLH-relative angular velocity is
/// `w_iss_lvlh = [0.002, 0.006, -0.003]` deg/s (`iss_rate_def.py`); the
/// deck leaves `rate_in_parent` unset (JEOD default `false`), so the
/// rate is interpreted in the body frame
/// ([`LvlhAngularVelocityFrame::Body`]).
///
/// `init_rot_from_lvlh` composes the LVLH→body attitude / LVLH-relative
/// rate with the reference orbit's LVLH frame orientation and angular
/// velocity wrt inertial; the reference orbit is the inertial Cartesian
/// state from `trans_TransState_inertial_body`.
fn lvlh_rot_state(reference: TranslationalState) -> RotationalState {
    let angles = [
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
    ];
    let q_lvlh_body =
        compute_quaternion_from_euler_angles_typed(angles, EulerSequence::YZX).inner();

    // LVLH-relative body rate (deg/s) — iss_rate_def.py.
    let w_iss_lvlh_deg = DVec3::new(0.002, 0.006, -0.003);
    let ang_vel_lvlh_to_body = DVec3::new(
        Angle::new::<degree>(w_iss_lvlh_deg.x).get::<uom::si::angle::radian>(),
        Angle::new::<degree>(w_iss_lvlh_deg.y).get::<uom::si::angle::radian>(),
        Angle::new::<degree>(w_iss_lvlh_deg.z).get::<uom::si::angle::radian>(),
    );

    BodyAction::InitLvlhRot {
        q_lvlh_body,
        ang_vel_lvlh_to_body,
        ang_vel_frame: LvlhAngularVelocityFrame::Body,
        reference_position: reference.position,
        reference_velocity: reference.velocity,
    }
    .apply_rotational()
    .expect("BodyAction::InitLvlhRot is a rotational action and must yield Some(RotationalState)")
}

/// RUN_2100: ISS inertial Cartesian translation + direct inertial
/// attitude/rate init (`DynBodyInitRotState`, `Earth.inertial`).
fn build_run_2100(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let trans = trans_state("ISS", "trans_TransState_inertial_body");
    let rot = attitude_rate_inertial_state();
    build_orbinit_docker_rot(mu, trans, rot)
}

/// RUN_1230: ISS inertial Cartesian translation + LVLH-relative
/// attitude/rate init (`DynBodyInitLvlhRotState`, Earth).
fn build_run_1230(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let trans = trans_state("ISS", "trans_TransState_inertial_body");
    let rot = lvlh_rot_state(trans);
    build_orbinit_docker_rot(mu, trans, rot)
}

// ── Vehicle-relative initialization (double-vehicle RUNs) ───────────────────
//
// In RUN_0441/0571/0681/3771 the STS-114 chaser's state is specified relative
// to an ISS frame, and JEOD composes it with the already-initialized ISS
// inertial state through `RefFrameState::incr_left`. The ISS target falls
// into the default branch of `Modified_data/double_vehicle_run.py`:
//   set_ISS_trans_TransState_inertial_body  (inertial Cartesian, below)
//   set_ISS_rot_LvlhRotState_lvlh_body      (identity LVLH→body, LVLH rate)
// so the ISS inertial trans/rot used as the reference are exactly the
// RUN_1230 reference state. The chaser offset numbers come from the
// `Modified_data/STS_114/*` decks (JEOD source — permitted).

/// ISS target inertial translational state — `set_ISS_trans_TransState_inertial_body`
/// (`Modified_data/ISS/trans_TransState_inertial_body.py`), the default ISS
/// translation in `double_vehicle_run.py`.
fn iss_reference_trans() -> TranslationalState {
    trans_state("ISS", "trans_TransState_inertial_body")
}

/// ISS target inertial rotational state — `set_ISS_rot_LvlhRotState_lvlh_body`
/// (`Modified_data/ISS/rot_LvlhRotState_lvlh_body.py`): identity LVLH→body
/// (Pitch_Yaw_Roll Euler `[0,0,0]`) with LVLH-relative body rate
/// `w_iss_lvlh = [0.002, 0.006, -0.003]` deg/s (`iss_rate_def.py`, body
/// frame). `init_rot_from_lvlh` composes this with the ISS LVLH frame's own
/// orientation/rate wrt inertial — identical to RUN_1230's `lvlh_rot_state`.
fn iss_reference_rot(trans: TranslationalState) -> RotationalState {
    lvlh_rot_state(trans)
}

/// ISS composite-body frame `B` expressed wrt the inertial frame: origin at
/// the ISS inertial position/velocity, oriented by the ISS inertial→body
/// attitude `T_inertial_issbody`, rotating at the ISS body-frame inertial
/// rate `ω_inertial_issbody`. This is the reference frame for RUN_0441
/// (chaser offset in `ISS.composite_body`).
fn iss_body_frame_state() -> RefFrameState {
    let trans = iss_reference_trans();
    let rot = iss_reference_rot(trans);
    let t_parent_this = rot.quaternion.left_quat_to_transformation();
    RefFrameState {
        trans: RefFrameTrans {
            position: trans.position,
            velocity: trans.velocity,
        },
        rot: RefFrameRot {
            q_parent_this: rot.quaternion,
            t_parent_this,
            ang_vel_this: rot.ang_vel_body,
        },
    }
}

/// ISS LVLH frame `B` expressed wrt the inertial frame, from the ISS inertial
/// orbit state (JEOD `LvlhFrame::compute_lvlh_frame`): origin co-located /
/// co-moving with ISS, oriented by `T_inertial_lvlh`, rotating at the orbital
/// rate `ω_inertial_lvlh = [0, -|h|/|r|², 0]` in LVLH. Reference frame for
/// RUN_0571 / RUN_3771 (chaser offset / full state in the ISS LVLH frame).
fn iss_lvlh_frame_state() -> RefFrameState {
    let trans = iss_reference_trans();
    let lvlh = compute_body_lvlh_frame(trans.position, trans.velocity);
    RefFrameState {
        trans: RefFrameTrans {
            position: trans.position,
            velocity: trans.velocity,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&lvlh.t_parent_this),
            t_parent_this: lvlh.t_parent_this,
            ang_vel_this: lvlh.ang_vel_this,
        },
    }
}

/// STS-114 mass properties for the chaser body
/// (`Modified_data/STS_114/mass.py`, `set_STS_114_mass`): `mass = 10000` kg,
/// CG `position = [27.856, 0.003, 9.600]` m, body-frame (CG-centred) diagonal
/// inertia `diag(7e11, 12e11, 10e11)` kg·m². The deck's non-identity
/// `pt_orientation` (`StructToBody` = `diag(-1, 1, -1)`) rotates the structure
/// frame relative to the body frame, but the relative-init targets and the
/// CSV log the **composite_body** frame, so the structure↔body rotation does
/// not enter this translation/attitude cross-validation; the body-frame
/// inertia is used directly (`with_inertia`, identity `t_parent_this`).
fn sts_mass_properties() -> astrodyn::MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(7.0e11, 0.0, 0.0),
        DVec3::new(0.0, 12.0e11, 0.0),
        DVec3::new(0.0, 0.0, 10.0e11),
    );
    astrodyn::MassProperties::with_inertia(10_000.0, inertia, DVec3::new(27.856, 0.003, 9.600))
}

/// STS-114 chaser ISS-relative position offset, common to RUN_0441 / 0571 /
/// 3771 (the body / LVLH translation decks share the same component sum from
/// `Modified_data/STS_114/trans_*`):
///   x = 10.201 + 9.844 + 5 + 9.600 − 9.600
///   y = −0.206 + 0 + 0 + 0.003 − 0.003
///   z = −2.558 + 5.252 + 100 − 3.937 + 27.856
fn sts_offset_position() -> DVec3 {
    DVec3::new(
        10.201 + 9.844 + 5.0 + 9.600 - 9.600,
        -0.206 + 0.000 + 0.0 + 0.003 - 0.003,
        -2.558 + 5.252 + 100.0 - 3.937 + 27.856,
    )
}

/// STS-114 chaser ISS-relative velocity offset for RUN_0441 / 0571 / 3771:
/// `[0, 0, -1]` m/s (`Modified_data/STS_114/trans_*` velocity).
fn sts_offset_velocity() -> DVec3 {
    DVec3::new(0.0, 0.0, -1.0)
}

/// Build a 6-DOF chaser scenario for the double-vehicle relative-init RUNs.
/// The ISS reference state is computed and used to construct the reference
/// frame; the chaser is the single body the CSV logs and cross-validates.
/// `rot` carries the chaser's composite-body rotational state (identity for
/// the translation-only RUNs where `rotational_dynamics = False`, the
/// composed LVLH attitude for RUN_3771).
fn build_orbinit_relative(
    mu_earth: f64,
    chaser_trans: TranslationalState,
    chaser_rot: RotationalState,
) -> SimulationBuilder {
    let mut sb = build_orbinit_docker(mu_earth, chaser_trans);
    sb.bodies[0].rot = Some(super::typed_helpers::rot_typed(&chaser_rot));
    sb.bodies[0].mass = Some(super::typed_helpers::mass_typed(&sts_mass_properties()));
    sb
}

/// RUN_0441: STS-114 chaser translation given in the ISS composite-body frame
/// (`trans_TransState_tbody_body`, `reference_ref_frame_name =
/// "ISS.composite_body"`). The offset is composed with the ISS body frame's
/// inertial state via `incr_left` (translation only; chaser attitude/rate
/// stay at identity/zero, matching the deck's `rotational_dynamics = False`).
fn build_run_0441(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let frame = iss_body_frame_state();
    let chaser_trans = BodyAction::InitTransRelativeFrame {
        reference_frame: frame,
        offset_position: sts_offset_position(),
        offset_velocity: sts_offset_velocity(),
    }
    .apply_translational()
    .expect("InitTransRelativeFrame must yield Some(TranslationalState)");
    build_orbinit_relative(mu, chaser_trans, RotationalState::default())
}

/// RUN_0571: STS-114 chaser translation given in the ISS LVLH frame
/// (`trans_LvlhTransState_tlvlh_body`, `ref_body_name = "ISS"`, planet Earth).
/// The offset is composed with the ISS LVLH frame's inertial state via
/// `incr_left`, which carries the LVLH frame-rate `ω × r` term (the LVLH frame
/// rotates at the orbital rate). Translation only.
fn build_run_0571(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let frame = iss_lvlh_frame_state();
    let chaser_trans = BodyAction::InitTransRelativeFrame {
        reference_frame: frame,
        offset_position: sts_offset_position(),
        offset_velocity: sts_offset_velocity(),
    }
    .apply_translational()
    .expect("InitTransRelativeFrame must yield Some(TranslationalState)");
    build_orbinit_relative(mu, chaser_trans, RotationalState::default())
}

/// ISS NED frame `B` expressed wrt the inertial frame, for RUN_0681
/// (chaser offset in the NED frame relative to ISS, spherical lat/lon).
///
/// JEOD `DynBodyInitNedState::apply` (`ref_body = ISS`) builds the NED frame
/// as a child of the planet-fixed (pfix) frame: its origin is co-located /
/// co-moving with ISS (ISS state expressed wrt pfix), its orientation is the
/// NED-axes-from-spherical-lat/lon matrix, and it has *zero* angular velocity
/// wrt pfix (`NorthEastDown::build_ned_orientation` zeroes `ang_vel_this`).
/// Because pfix itself rotates at the planet rate wrt inertial, the NED
/// frame's inertial state is obtained by composing pfix→inertial:
///   S_inertial:ned = incr_left(S_inertial:pfix, S_pfix:ned)
/// where the pfix frame carries `ω_planet` (in pfix coordinates, +z). The
/// spherical latitude/longitude (`lat = asin(z/r)`, `lon = atan2(y, x)` from
/// the pfix position) match JEOD `PlanetFixedPosition::cart_to_spher`, and the
/// NED-axes matrix matches `build_ned_orientation`.
fn iss_ned_frame_state() -> RefFrameState {
    let trans = iss_reference_trans();
    let t_inertial_pfix = t_inertial_pfix_at_epoch();
    let omega_pfix = DVec3::new(0.0, 0.0, EARTH.omega);

    // ISS state wrt pfix, expressed in pfix coordinates (JEOD
    // `compute_relative_state` ISS.composite_body wrt pfix):
    //   r_pfix = T_inertial_pfix · r_inertial
    //   v_pfix = T_inertial_pfix · v_inertial − ω_planet × r_pfix
    let r_pfix = t_inertial_pfix * trans.position;
    let v_pfix = t_inertial_pfix * trans.velocity - omega_pfix.cross(r_pfix);

    // Spherical latitude / longitude from the pfix position
    // (JEOD `cart_to_spher`).
    let r_local = r_pfix.length();
    let lat = (r_pfix.z / r_local).asin();
    let lon = r_pfix.y.atan2(r_pfix.x);

    // NED-axes matrix T_pfix_ned (JEOD `build_ned_orientation`): rows are
    // North, East, Down. glam `from_cols` takes columns, so each column j
    // gathers the jth component of (North, East, Down).
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let t_pfix_ned = DMat3::from_cols(
        DVec3::new(-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon),
        DVec3::new(-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon),
        DVec3::new(cos_lat, 0.0, -sin_lat),
    );

    // S_inertial:pfix — pfix frame wrt inertial (origin coincident, rotating
    // at ω_planet expressed in pfix coordinates, +z).
    let s_inertial_pfix = RefFrameState {
        trans: RefFrameTrans {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_inertial_pfix),
            t_parent_this: t_inertial_pfix,
            ang_vel_this: omega_pfix,
        },
    };

    // S_pfix:ned — NED frame wrt pfix (origin at ISS pfix state, NED axes,
    // zero rate wrt pfix). Composed up to inertial in place.
    let mut s_inertial_ned = RefFrameState {
        trans: RefFrameTrans {
            position: r_pfix,
            velocity: v_pfix,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_pfix_ned),
            t_parent_this: t_pfix_ned,
            ang_vel_this: DVec3::ZERO,
        },
    };
    s_inertial_ned.incr_left(&s_inertial_pfix);
    s_inertial_ned
}

/// RUN_0681: STS-114 chaser translation in the NED frame relative to ISS
/// (`trans_NedTransState_tned_body`, `ref_body_name = "ISS"`,
/// `altlatlong_type = spherical`). The offset
/// (`Modified_data/STS_114/trans_NedTransState_tned_body.py`) is position
/// `[17.504, 17.914, 126.613]` m and planet-fixed NED velocity
/// `[-0.101060, -0.095858, -0.972466]` m/s, composed with the ISS NED frame
/// via `incr_left`. The planet-rotation velocity contribution enters through
/// the pfix→inertial step in [`iss_ned_frame_state`]. Translation only.
fn build_run_0681(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let frame = iss_ned_frame_state();
    let chaser_trans = BodyAction::InitTransRelativeFrame {
        reference_frame: frame,
        offset_position: DVec3::new(17.504, 17.914, 126.613),
        offset_velocity: DVec3::new(-0.101060, -0.095858, -0.972466),
    }
    .apply_translational()
    .expect("InitTransRelativeFrame must yield Some(TranslationalState)");
    build_orbinit_relative(mu, chaser_trans, RotationalState::default())
}

/// RUN_3771: STS-114 chaser full state (Pos_Vel_Att_Rate) in the ISS LVLH
/// frame (`full_LvlhState_tlvlh_body`). Position/velocity are the same LVLH
/// offset as RUN_0571; the attitude is the Pitch_Roll_Yaw (JEOD
/// `Pitch_Roll_Yaw` = `EulerSequence::YXZ`) Euler triple `[90, 0, 0]` deg
/// (LVLH→body), and the LVLH-relative body rate is
/// `[−w_sts_lvlh[2], w_sts_lvlh[1], w_sts_lvlh[0]]` deg/s with
/// `w_sts_lvlh = [0.06, 0.03, 0.02]` (`chaser_rate_def.py`). Both substates
/// are composed with the ISS LVLH frame via `incr_left`.
fn build_run_3771(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let frame = iss_lvlh_frame_state();

    // LVLH→body attitude: Pitch_Roll_Yaw (YXZ) Euler [90, 0, 0] deg.
    let angles = [
        Angle::new::<degree>(90.0),
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
    ];
    let q_frame_subject =
        compute_quaternion_from_euler_angles_typed(angles, EulerSequence::YXZ).inner();

    // LVLH-relative body rate (deg/s), body frame (rate_in_parent unset →
    // false). chaser_rate_def.py: w_sts_lvlh = [0.06, 0.03, 0.02];
    // ang_velocity = [-w_sts_lvlh[2], w_sts_lvlh[1], w_sts_lvlh[0]].
    let w_sts_lvlh = [0.06, 0.03, 0.02];
    let ang_vel_deg = DVec3::new(-w_sts_lvlh[2], w_sts_lvlh[1], w_sts_lvlh[0]);
    let ang_vel_frame_to_subject = DVec3::new(
        Angle::new::<degree>(ang_vel_deg.x).get::<uom::si::angle::radian>(),
        Angle::new::<degree>(ang_vel_deg.y).get::<uom::si::angle::radian>(),
        Angle::new::<degree>(ang_vel_deg.z).get::<uom::si::angle::radian>(),
    );

    let action = BodyAction::InitFullRelativeFrame {
        reference_frame: frame,
        offset_position: sts_offset_position(),
        offset_velocity: sts_offset_velocity(),
        q_frame_subject,
        ang_vel_frame_to_subject,
    };
    let chaser_trans = action
        .apply_translational()
        .expect("InitFullRelativeFrame must yield Some(TranslationalState)");
    let chaser_rot = action
        .apply_rotational()
        .expect("InitFullRelativeFrame must yield Some(RotationalState)");
    build_orbinit_relative(mu, chaser_trans, chaser_rot)
}

// ── Single-vehicle NED full-state initialization (RUN_3822) ─────────────────
//
// RUN_3822 (`full_NedState_ned_struct`, `TARGET_ELLIPTICAL_NED`) initializes a
// single PAD_39A vehicle's full state (Pos_Vel_Att_Rate) in the local NED frame
// at a fixed ground point — no reference body. The point is geodetic
// (elliptical/ellipsoid) at lat 28.6082°, lon −80.6040°, alt 3.0 m
// (`Modified_data/PAD_39A/full_NedState_ned_struct.py`), the body sits
// `[0, 0, 10]` m below it (NED z = Down) at rest in NED, and the attitude is the
// Pitch_Yaw_Roll Euler triple `[0, 0, 0]` deg (body aligned with local NED).
// Because the NED frame rotates with the Earth, the body's *inertial* velocity
// is `ω_earth × r` and its *inertial* angular velocity recovers `ω_earth`, even
// though the NED-frame-relative velocity and rate are zero.

/// PAD_39A mass properties (`Modified_data/PAD_39A/mass.py`, `set_PAD_39A_mass`):
/// `mass = 1.0` kg, CG at the structure origin (`position = [0, 0, 0]`),
/// identity inertia (`diag(1, 1, 1)` kg·m²), `inertia_spec = Body`, and an
/// identity `pt_orientation` (`StructToBody`). The structure frame therefore
/// coincides with the body / composite_body frame (no offset, no rotation), so
/// `body_frame_id = "structure"` maps to the composite-body frame the CSV logs
/// without any struct↔body transform. JEOD *source* initial conditions
/// (permitted by the computational-independence rule).
fn pad_39a_mass_properties() -> astrodyn::MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    astrodyn::MassProperties::with_inertia(1.0, inertia, DVec3::ZERO)
}

/// PAD_39A geodetic ground point (deg→rad, ellipsoidal lat/lon, alt 3 m), from
/// `Modified_data/PAD_39A/full_NedState_ned_struct.py` (and the matching
/// `earth.pad_39a.loc` in `double_vehicle_run.py`).
fn pad_39a_geodetic() -> astrodyn::GeodeticState {
    astrodyn::GeodeticState {
        latitude: Angle::new::<degree>(28.6082).get::<uom::si::angle::radian>(),
        longitude: Angle::new::<degree>(-80.6040).get::<uom::si::angle::radian>(),
        altitude: 3.0,
    }
}

/// PAD_39A full-NED initialization action (`DynBodyInitNedState`,
/// `set_items = Pos_Vel_Att_Rate`, elliptical), shared by RUN_3822 and the
/// RUN_4681 NED-frame construction. The body sits `[0, 0, 10]` m Down from the
/// geodetic point, aligned with the local NED frame (Pitch_Yaw_Roll `[0,0,0]`),
/// at rest in NED (`Modified_data/PAD_39A/full_NedState_ned_struct.py`).
fn pad_39a_full_ned_action() -> BodyAction {
    let angles = [
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
    ];
    let q_ned_body = compute_quaternion_from_euler_angles_typed(angles, EulerSequence::YZX).inner();
    BodyAction::InitFullNed {
        geodetic: pad_39a_geodetic(),
        ned_position: DVec3::new(0.0, 0.0, 10.0),
        ned_velocity: DVec3::ZERO,
        q_ned_body,
        ang_vel_ned_to_body: DVec3::ZERO,
        r_equatorial: EARTH.shape.r_eq(),
        r_polar: EARTH.shape.r_pol(),
        t_eci_pcpf: t_inertial_pfix_at_epoch(),
        // pfix-frame angular velocity (JEOD `planet_rnp.cc` stores
        // `ang_vel_this = [0, 0, planet_omega]` about the pfix z-axis).
        omega_planet: DVec3::new(0.0, 0.0, EARTH.omega),
    }
}

/// RUN_3822: PAD_39A single-vehicle full state (Pos_Vel_Att_Rate) in the local
/// NED frame at a geodetic ground point (`DynBodyInitNedState`,
/// `altlatlong_type = elliptical`, no reference body).
///
/// The geodetic reference point (lat 28.6082°, lon −80.6040°, alt 3.0 m) and the
/// `[0, 0, 10]` m NED offset / identity attitude / zero NED rate come from
/// `Modified_data/PAD_39A/full_NedState_ned_struct.py`. `BodyAction::InitFullNed`
/// builds the inertial→NED frame for the ground point (NED axes from the
/// geodetic lat/lon, the frame stationary wrt pfix but carrying Earth's
/// `ω_planet` rate wrt inertial through the pfix→inertial composition), then
/// composes the body's NED-frame offset / attitude / rate up to inertial via
/// `RefFrameState::incr_left`. The inertial→pfix rotation is the SIM_orbinit
/// epoch matrix [`t_inertial_pfix_at_epoch`]; Earth's polar radius and rotation
/// rate are `EARTH.r_pol()` / `EARTH.omega`.
///
/// The attitude is the Pitch_Yaw_Roll (JEOD `Pitch_Yaw_Roll` =
/// `EulerSequence::YZX`) Euler triple `[0, 0, 0]` deg, so the NED→body
/// quaternion is identity — but the body's *inertial* attitude is the
/// non-trivial inertial→NED rotation at 28.6°N / −80.6°E, and its *inertial*
/// angular velocity recovers `ω_earth`.
fn build_run_3822(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let action = pad_39a_full_ned_action();
    let trans = action
        .apply_translational()
        .expect("InitFullNed must yield Some(TranslationalState)");
    let rot = action
        .apply_rotational()
        .expect("InitFullNed must yield Some(RotationalState)");

    let mut sb = build_orbinit_docker(mu, trans);
    sb.bodies[0].rot = Some(super::typed_helpers::rot_typed(&rot));
    sb.bodies[0].mass = Some(super::typed_helpers::mass_typed(&pad_39a_mass_properties()));
    sb
}

// ── Structure-frame / named-mass-point vehicle-relative initialization ──────
//
// RUN_4451 / RUN_5461 / RUN_4681 initialize the STS-114 chaser relative to a
// *non-composite-body* frame of the target — the target's `structure` frame
// (RUN_4451) or a named mass point `attach_point` (RUN_5461) — and the chaser's
// own `body_frame_id` is `structure` (RUN_4451/4681) or `attach_point`
// (RUN_5461), not `composite_body`. JEOD's `DynBodyInit::apply_user_inputs`
// composes the user offset/attitude onto the reference frame, sets the result
// on the subject's `body_frame_id` frame, then `propagate_state()` derives the
// chaser's `composite_body` frame (what the CSV logs) from it.
//
// Mass tree (each vehicle): the `structure` point is the root; the
// `composite_body` (CG) point is its `StructToBody` child (position = CG in
// structure coords, `T_parent_this` = T_struct_body); a named mass point is a
// `StructToPoint` child (position = point in structure coords, `T_parent_this`
// = T_struct_point). JEOD `mass_point_init.cc` stores StructTo* matrices as
// `T_parent_this` (parent = structure) verbatim.
//
// All composition below is the canonical `RefFrameState` math
// (`incr_right` / `negate`) — the same operators JEOD's
// `compute_derived_state_forward` / `compute_derived_state_reverse`
// (`dyn_body_propagate_state.cc`) and `compute_relative_state`
// (`ref_frame_state.cc`) reduce to for rigid mass-point offsets. No new physics
// kernel is required: a structure / named-point frame state is a deterministic
// rigid compose of the composite-body inertial state with the mass tree.

/// A rigid mass-point relative state `S_parent:point` (parent → point): the
/// point's origin at `position` in parent coordinates, axes rotated by
/// `t_struct_point` (parent → point), and *zero* angular velocity wrt the
/// parent (the point is rigidly fixed in the structure). This mirrors a JEOD
/// `MassPoint`'s stored `{position, T_parent_this}` (`mass_point.hh`), promoted
/// to a [`RefFrameState`] so it composes through [`RefFrameState::incr_right`].
fn rigid_mass_point_state(position: DVec3, t_struct_point: DMat3) -> RefFrameState {
    RefFrameState {
        trans: RefFrameTrans {
            position,
            velocity: DVec3::ZERO,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_struct_point),
            t_parent_this: t_struct_point,
            ang_vel_this: DVec3::ZERO,
        },
    }
}

/// Build a target vehicle's `structure`-frame state wrt the inertial frame from
/// its `composite_body` inertial state and its `StructToBody` mass properties
/// (`cg_struct` = CG location in structure coordinates, `t_struct_body` =
/// structure → body rotation).
///
/// JEOD computes this in `propagate_state_from_composite` via
/// `compute_derived_state_reverse(composite_body, composite_properties,
/// structure)` (`dyn_body_propagate_state.cc`). Expressed as a frame compose,
/// `S_inertial:struct = S_inertial:composite ∘ S_composite:struct`, where
/// `S_composite:struct = negate(S_struct:composite)` and `S_struct:composite`
/// is the rigid `{cg_struct, t_struct_body}` point. The reverse step carries
/// the composite-body inertial rate into the structure-frame `ω × r` velocity.
// JEOD_INV: BA.17 — structure frame from composite via the rigid mass-tree
// reverse relation (compute_derived_state_reverse), a RefFrameState compose.
fn vehicle_structure_frame_state(
    composite_inertial: &RefFrameState,
    cg_struct: DVec3,
    t_struct_body: DMat3,
) -> RefFrameState {
    let s_struct_composite = rigid_mass_point_state(cg_struct, t_struct_body);
    let s_composite_struct = RefFrameState::negate(&s_struct_composite);
    composite_inertial.incr_right(&s_composite_struct)
}

/// Build a target vehicle's named-mass-point frame state wrt the inertial frame
/// from its `composite_body` inertial state, its `StructToBody` mass properties,
/// and the named point's `StructToPoint` data (`pt_struct` = point location in
/// structure coordinates, `t_struct_point` = structure → point rotation).
///
/// JEOD computes this in `compute_vehicle_point_states` via
/// `compute_derived_state_forward(structure, point->mass_point, point)`
/// (`dyn_body_propagate_state.cc`) — i.e. forward from the *structure* frame
/// (built first by the reverse step above) through the rigid point. Expressed as
/// a frame compose, `S_inertial:point = S_inertial:struct ∘ S_struct:point`,
/// where `S_struct:point` is the rigid `{pt_struct, t_struct_point}` point.
fn vehicle_mass_point_frame_state(
    composite_inertial: &RefFrameState,
    cg_struct: DVec3,
    t_struct_body: DMat3,
    pt_struct: DVec3,
    t_struct_point: DMat3,
) -> RefFrameState {
    let s_inertial_struct =
        vehicle_structure_frame_state(composite_inertial, cg_struct, t_struct_body);
    let s_struct_point = rigid_mass_point_state(pt_struct, t_struct_point);
    s_inertial_struct.incr_right(&s_struct_point)
}

/// Rigid `S_subject_bframe:composite` for converting a chaser state initialized
/// on its `body_frame_id` frame into the `composite_body` frame the CSV logs.
///
/// JEOD's `DynBodyInit::apply` sets the computed state on the subject's
/// `body_frame_id` frame, then `propagate_state()` derives `composite_body`.
/// When `body_frame_id = structure`, `S_struct:composite` is the rigid
/// `StructToBody` point `{cg_struct, t_struct_body}`. When `body_frame_id` is a
/// named point, `S_point:composite = negate(S_struct:point) ∘ S_struct:composite`
/// walks point → structure → composite through the mass tree
/// (`compute_relative_state`, `mass_point.cc`).
fn chaser_bframe_to_composite_structure(cg_struct: DVec3, t_struct_body: DMat3) -> RefFrameState {
    rigid_mass_point_state(cg_struct, t_struct_body)
}

/// Rigid `S_point:composite` for a chaser initialized on a named mass point:
/// `S_point:composite = negate(S_struct:point) ∘ S_struct:composite`.
fn chaser_bframe_to_composite_point(
    cg_struct: DVec3,
    t_struct_body: DMat3,
    pt_struct: DVec3,
    t_struct_point: DMat3,
) -> RefFrameState {
    let s_struct_point = rigid_mass_point_state(pt_struct, t_struct_point);
    let s_point_struct = RefFrameState::negate(&s_struct_point);
    let s_struct_composite = rigid_mass_point_state(cg_struct, t_struct_body);
    s_point_struct.incr_right(&s_struct_composite)
}

/// STS-114 chaser `StructToBody` CG location in structure coordinates (m) and
/// the structure → body rotation matrix `T_struct_body`, from
/// `Modified_data/STS_114/mass.py` (`set_STS_114_mass`): `position =
/// [27.856, 0.003, 9.600]`, `pt_orientation` (InputMatrix, StructToBody) rows
/// `[-1,0,0] / [0,1,0] / [0,0,-1]` = `diag(-1, 1, -1)`.
fn sts_cg_struct() -> DVec3 {
    DVec3::new(27.856, 0.003, 9.600)
}
fn sts_t_struct_body() -> DMat3 {
    // Row-major rows [-1,0,0]/[0,1,0]/[0,0,-1]; glam from_cols takes columns.
    DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, -1.0),
    )
}

/// STS-114 chaser `attach_point` mass point: structure-coordinate location
/// `[3.937, 0.003, 9.600]` m and `T_struct_point` rows `[-1,0,0] / [0,1,0] /
/// [0,0,-1]` = `diag(-1, 1, -1)` (`Modified_data/STS_114/mass.py`, StructToPoint).
fn sts_attach_pt_struct() -> DVec3 {
    DVec3::new(3.937, 0.003, 9.600)
}
fn sts_attach_t_struct_point() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, -1.0),
    )
}

/// ISS target `StructToBody` CG location `[-10.201, 0.206, 2.558]` m and
/// identity `T_struct_body` (`Modified_data/ISS/mass.py`, StructToBody
/// `pt_orientation` = identity).
fn iss_cg_struct() -> DVec3 {
    DVec3::new(-10.201, 0.206, 2.558)
}
fn iss_t_struct_body() -> DMat3 {
    DMat3::IDENTITY
}

/// ISS target `attach_point` mass point: structure-coordinate location
/// `[9.844, 0.000, 5.282]` m and `T_struct_point` rows `[0,0,1] / [0,-1,0] /
/// [1,0,0]` (`Modified_data/ISS/mass.py`, StructToPoint, InputMatrix).
fn iss_attach_pt_struct() -> DVec3 {
    DVec3::new(9.844, 0.000, 5.282)
}
fn iss_attach_t_struct_point() -> DMat3 {
    // Row-major rows [0,0,1]/[0,-1,0]/[1,0,0]; glam from_cols takes columns.
    DMat3::from_cols(
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
    )
}

/// Convert a deg/s 3-vector to a rad/s `DVec3` (uom-checked angular-velocity
/// conversion, no lossy literal).
fn deg_per_s_to_rad(v: DVec3) -> DVec3 {
    use uom::si::angular_velocity::{degree_per_second, radian_per_second};
    use uom::si::f64::AngularVelocity;
    DVec3::new(
        AngularVelocity::new::<degree_per_second>(v.x).get::<radian_per_second>(),
        AngularVelocity::new::<degree_per_second>(v.y).get::<radian_per_second>(),
        AngularVelocity::new::<degree_per_second>(v.z).get::<radian_per_second>(),
    )
}

/// RUN_4451: STS-114 chaser full state (trans + rot) given in the **ISS
/// structure** frame (`trans_TransState_tstruct_struct` +
/// `rot_RotState_tstruct_struct`, `reference_ref_frame_name = "ISS.structure"`,
/// `body_frame_id = "structure"`, `state_items = Both`).
///
/// The reference is the ISS structure frame, derived from the ISS composite-body
/// inertial state and its `StructToBody` mass properties. The chaser's user
/// offset / attitude / rate are composed onto it; the result is the chaser's
/// *structure*-frame inertial state, then converted to the chaser
/// `composite_body` frame the CSV logs. Offsets are
/// `Modified_data/STS_114/trans_TransState_tstruct_struct.py` /
/// `rot_RotState_tstruct_struct.py`:
///   * position `[9.844+5+9.600, 0.003, 5.252+100−3.937]` m, velocity `[0,0,−1]`,
///   * attitude Pitch_Yaw_Roll (YZX) Euler `[−90, 0, 0]` deg (structure→structure),
///   * rate `[−w_sts_issb[0], w_sts_issb[1], −w_sts_issb[2]]` deg/s, with
///     `w_sts_issb = w_sts_lvlh − w_iss_lvlh`, `w_sts_lvlh = [0.06, 0.03, 0.02]`
///     (`chaser_rate_def.py`), `w_iss_lvlh = [0.002, 0.006, −0.003]`
///     (`iss_rate_def.py`). The deck leaves `rate_in_parent` unset (JEOD default
///     false), so the rate is in the chaser structure frame.
fn build_run_4451(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();

    // ISS structure frame wrt inertial, from the ISS composite-body state.
    let iss_composite = iss_body_frame_state();
    let ref_frame =
        vehicle_structure_frame_state(&iss_composite, iss_cg_struct(), iss_t_struct_body());

    // Chaser offsets, attitude (YZX [-90,0,0]), and structure-frame rate.
    let offset_position = DVec3::new(9.844 + 5.0 + 9.600, 0.000 + 0.003, 5.252 + 100.0 - 3.937);
    let offset_velocity = DVec3::new(0.0, 0.0, -1.0);

    let angles = [
        Angle::new::<degree>(-90.0),
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
    ];
    let q_frame_subject =
        compute_quaternion_from_euler_angles_typed(angles, EulerSequence::YZX).inner();

    // w_sts_issb = w_sts_lvlh − w_iss_lvlh; ang_velocity = [-issb0, issb1, -issb2].
    let w_sts_lvlh = DVec3::new(0.06, 0.03, 0.02);
    let w_iss_lvlh = DVec3::new(0.002, 0.006, -0.003);
    let w_sts_issb = w_sts_lvlh - w_iss_lvlh;
    let ang_vel_frame_to_subject =
        deg_per_s_to_rad(DVec3::new(-w_sts_issb.x, w_sts_issb.y, -w_sts_issb.z));

    // Compose the user offset/attitude/rate onto the ISS structure frame: this
    // is the chaser's *structure*-frame inertial state.
    let action = BodyAction::InitFullRelativeFrame {
        reference_frame: ref_frame,
        offset_position,
        offset_velocity,
        q_frame_subject,
        ang_vel_frame_to_subject,
    };
    let chaser_struct_trans = action
        .apply_translational()
        .expect("InitFullRelativeFrame must yield Some(TranslationalState)");
    let chaser_struct_rot = action
        .apply_rotational()
        .expect("InitFullRelativeFrame must yield Some(RotationalState)");

    // Convert chaser structure-frame state → chaser composite_body (CSV subject).
    let (chaser_trans, chaser_rot) = chaser_struct_to_composite(
        &chaser_struct_trans,
        &chaser_struct_rot,
        chaser_bframe_to_composite_structure(sts_cg_struct(), sts_t_struct_body()),
    );
    build_orbinit_relative(mu, chaser_trans, chaser_rot)
}

/// Compose a chaser `body_frame_id`-frame inertial state with the rigid
/// `S_bframe:composite` mass-point relation to obtain the chaser
/// `composite_body` inertial state (the JEOD `propagate_state` derivation,
/// expressed as `S_inertial:composite = S_inertial:bframe ∘ S_bframe:composite`).
fn chaser_struct_to_composite(
    bframe_trans: &TranslationalState,
    bframe_rot: &RotationalState,
    s_bframe_composite: RefFrameState,
) -> (TranslationalState, RotationalState) {
    let t_parent_this = bframe_rot.quaternion.left_quat_to_transformation();
    let s_inertial_bframe = RefFrameState {
        trans: RefFrameTrans {
            position: bframe_trans.position,
            velocity: bframe_trans.velocity,
        },
        rot: RefFrameRot {
            q_parent_this: bframe_rot.quaternion,
            t_parent_this,
            ang_vel_this: bframe_rot.ang_vel_body,
        },
    };
    let s_inertial_composite = s_inertial_bframe.incr_right(&s_bframe_composite);
    (
        TranslationalState {
            position: s_inertial_composite.trans.position,
            velocity: s_inertial_composite.trans.velocity,
        },
        RotationalState {
            quaternion: s_inertial_composite.rot.q_parent_this,
            ang_vel_body: s_inertial_composite.rot.ang_vel_this,
        },
    )
}

/// RUN_5461: STS-114 chaser with **mixed references** — position/attitude given
/// relative to the **ISS `attach_point`** named mass point
/// (`trans_TransState_tpoint_point` + `att_RotState_tpoint_point`,
/// `body_frame_id = "attach_point"`), and the angular rate given separately in
/// the **ISS LVLH** frame (`rate_LvlhRotState_tlvlh_body`,
/// `body_frame_id = "composite_body"`, `ref_body = ISS`, planet Earth).
///
/// JEOD applies the two rotational body-actions independently: the attitude
/// action sets the chaser's `attach_point`-frame attitude (no rate), and the
/// LVLH rate action sets the chaser's `composite_body`-frame rate. We compute
/// each path separately and assemble the chaser `composite_body` state.
///
/// Offsets (`Modified_data/STS_114/*`):
///   * position `[100, 0, 5]` m, velocity `[−1, 0, 0]` relative to attach_point,
///   * attitude Yaw_Pitch_Roll (ZYX) Euler `[180, 0, 0]` deg
///     (attach_point → attach_point),
///   * LVLH rate `[−w_sts_lvlh[2], w_sts_lvlh[1], w_sts_lvlh[0]]` deg/s with
///     `w_sts_lvlh = [0.06, 0.03, 0.02]` (`chaser_rate_def.py`), interpreted in
///     the chaser composite body frame (`rate_in_parent` unset → false).
fn build_run_5461(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let iss_composite = iss_body_frame_state();

    // ── Position + attitude path: relative to the ISS attach_point ──────────
    let ref_point = vehicle_mass_point_frame_state(
        &iss_composite,
        iss_cg_struct(),
        iss_t_struct_body(),
        iss_attach_pt_struct(),
        iss_attach_t_struct_point(),
    );
    let offset_position = DVec3::new(100.0, 0.0, 5.0);
    let offset_velocity = DVec3::new(-1.0, 0.0, 0.0);

    // Attitude Yaw_Pitch_Roll (ZYX) [180,0,0] deg, attach_point → attach_point.
    let angles = [
        Angle::new::<degree>(180.0),
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
    ];
    let q_point_subject =
        compute_quaternion_from_euler_angles_typed(angles, EulerSequence::ZYX).inner();

    // Attitude action carries zero attach-point-relative rate (the rate is set
    // by the separate LVLH action). Compose position + attitude onto the
    // attach_point frame: this is the chaser's *attach_point*-frame inertial
    // pos/vel/att.
    let att_action = BodyAction::InitFullRelativeFrame {
        reference_frame: ref_point,
        offset_position,
        offset_velocity,
        q_frame_subject: q_point_subject,
        ang_vel_frame_to_subject: DVec3::ZERO,
    };
    let chaser_pt_trans = att_action
        .apply_translational()
        .expect("InitFullRelativeFrame must yield Some(TranslationalState)");
    let chaser_pt_rot = att_action
        .apply_rotational()
        .expect("InitFullRelativeFrame must yield Some(RotationalState)");

    // ── Rate path: chaser composite-body rate given in the ISS LVLH frame ───
    // The chaser composite_body rate wrt inertial composes the LVLH-relative
    // body rate with the ISS LVLH frame's inertial rate — exactly RUN_1230's
    // `init_rot_from_lvlh` composition (the LVLH frame is built from the ISS
    // inertial orbit, the same reference RUN_3771 uses).
    let iss_trans = iss_reference_trans();
    let w_sts_lvlh = DVec3::new(0.06, 0.03, 0.02);
    let ang_vel_lvlh_to_body =
        deg_per_s_to_rad(DVec3::new(-w_sts_lvlh.z, w_sts_lvlh.y, w_sts_lvlh.x));
    let composite_ang_vel = BodyAction::InitLvlhRot {
        q_lvlh_body: JeodQuat::identity(),
        ang_vel_lvlh_to_body,
        ang_vel_frame: LvlhAngularVelocityFrame::Body,
        reference_position: iss_trans.position,
        reference_velocity: iss_trans.velocity,
    }
    .apply_rotational()
    .expect("InitLvlhRot must yield Some(RotationalState)")
    .ang_vel_body;

    // JEOD `update_integrated_state` (integration frame = composite_body)
    // reconciles the independently-sourced states onto composite_body:
    //   * attitude / position from the attach_point source frame,
    //   * rate from the composite_body source (the LVLH action), so the
    //     velocity ω×r term uses the *composite* rate, not the attach path's.
    // `S_composite:attach` = rigid attach_point→composite relation through the
    // chaser mass tree; the conversion mirrors lines 388-516 of
    // `dyn_body_propagate_state.cc`.
    let s_attach_composite = chaser_bframe_to_composite_point(
        sts_cg_struct(),
        sts_t_struct_body(),
        sts_attach_pt_struct(),
        sts_attach_t_struct_point(),
    );
    let s_composite_attach = RefFrameState::negate(&s_attach_composite);

    // Composite attitude: T_inertial_composite = T_composite_attach^T · T_inertial_attach
    // i.e. Q_inertial_composite = conj(Q_composite_attach) · Q_inertial_attach.
    let q_inertial_composite = s_composite_attach
        .rot
        .q_parent_this
        .conjugate()
        .multiply(&chaser_pt_rot.quaternion);
    let t_inertial_composite = q_inertial_composite.left_quat_to_transformation();

    // r_composite->attach:composite from the rigid relation.
    let r_composite_attach = s_composite_attach.trans.position;

    // Position: r_composite = r_attach − T_inertial_composite^T · r_composite->attach:composite.
    let chaser_position =
        chaser_pt_trans.position - t_inertial_composite.transpose() * r_composite_attach;

    // Velocity: v_composite = v_attach − T_inertial_composite^T · (ω_composite × r_composite->attach:composite),
    // with ω_composite the LVLH-sourced composite rate.
    let chaser_velocity = chaser_pt_trans.velocity
        - t_inertial_composite.transpose() * composite_ang_vel.cross(r_composite_attach);

    let chaser_trans = TranslationalState {
        position: chaser_position,
        velocity: chaser_velocity,
    };
    let chaser_rot = RotationalState {
        quaternion: q_inertial_composite,
        ang_vel_body: composite_ang_vel,
    };
    build_orbinit_relative(mu, chaser_trans, chaser_rot)
}

/// RUN_4681: target PAD_39A initialized full-NED at its geodetic ground point
/// (the RUN_3822 path), and STS-114 chaser initialized in the **NED frame
/// relative to PAD_39A** (`trans_NedTransState_tned_struct_pad_39a` +
/// `rot_NedRotState_tned_struct_pad_39a`, `reference = Earth.inertial`,
/// `ref_body = PAD_39A`, `body_frame_id = "structure"`, elliptical NED).
///
/// The reference is the NED frame at PAD_39A's geodetic location (the PAD sits
/// at its ground point with identity structure↔body, so the NED frame origin is
/// the PAD position). The chaser NED offset / attitude / rate compose onto it;
/// because the chaser `body_frame_id = structure` and STS-114 has a non-identity
/// `StructToBody`, the result is converted to the chaser `composite_body` frame.
///
/// Offsets (`Modified_data/STS_114/*_pad_39a.py`):
///   * NED position `[0, 0, −40]` m, NED velocity `[0, 0, −100]` m/s,
///   * attitude Pitch_Yaw_Roll (YZX) Euler `[0, 0, 0]` deg (structure aligned
///     with NED), rate `[1, 0, 0]` deg/s (chaser structure frame).
fn build_run_4681(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();

    // NED frame at PAD_39A's actual position (10 m below the ground point).
    let ned_frame = ned_reference_frame_state_at_pad();

    // Chaser NED offset / attitude (YZX [0,0,0] = identity) / structure-frame rate.
    let offset_position = DVec3::new(0.0, 0.0, -40.0);
    let offset_velocity = DVec3::new(0.0, 0.0, -100.0);
    let angles = [
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
        Angle::new::<degree>(0.0),
    ];
    let q_frame_subject =
        compute_quaternion_from_euler_angles_typed(angles, EulerSequence::YZX).inner();
    let ang_vel_frame_to_subject = deg_per_s_to_rad(DVec3::new(1.0, 0.0, 0.0));

    let action = BodyAction::InitFullRelativeFrame {
        reference_frame: ned_frame,
        offset_position,
        offset_velocity,
        q_frame_subject,
        ang_vel_frame_to_subject,
    };
    let chaser_struct_trans = action
        .apply_translational()
        .expect("InitFullRelativeFrame must yield Some(TranslationalState)");
    let chaser_struct_rot = action
        .apply_rotational()
        .expect("InitFullRelativeFrame must yield Some(RotationalState)");

    // Chaser structure-frame state → chaser composite_body (CSV subject).
    let (chaser_trans, chaser_rot) = chaser_struct_to_composite(
        &chaser_struct_trans,
        &chaser_struct_rot,
        chaser_bframe_to_composite_structure(sts_cg_struct(), sts_t_struct_body()),
    );
    build_orbinit_relative(mu, chaser_trans, chaser_rot)
}

/// Build the NED frame state wrt inertial at PAD_39A for the RUN_4681 chaser
/// reference (`DynBodyInitNedState` with `ref_body = PAD_39A`).
///
/// JEOD `dyn_body_init_ned_state.cc:148-162` builds the NED frame at the *ref
/// body's* location, not the bare geodetic point: it takes
/// `PAD_39A.composite_body` expressed wrt pfix, sets the NED origin to that pfix
/// position/velocity (`set_ned_trans_states`), recovers the geodetic lat/lon
/// from that pfix position (`update_from_cart`), and builds the elliptical
/// NED-axes matrix from it. The PAD sits 10 m Down from the geodetic ground
/// point (its own `full_NedState` offset), so the NED frame origin is the PAD's
/// actual position, 10 m below the bare ground point — using the bare geodetic
/// point would mis-place the reference by exactly that 10 m. The PAD inertial
/// state is the RUN_3822 result.
///
/// Construction mirrors [`iss_ned_frame_state`] (the RUN_0681 NED frame), but
/// the lat/lon comes from the geodetic (elliptical) inversion of the PAD pfix
/// position rather than a spherical lat/lon, and the origin is the PAD pfix
/// state rather than the ISS pfix state.
fn ned_reference_frame_state_at_pad() -> RefFrameState {
    // PAD_39A composite_body inertial state (RUN_3822).
    let pad = pad_39a_full_ned_action();
    let pad_trans = pad
        .apply_translational()
        .expect("InitFullNed must yield Some(TranslationalState)");

    let t_inertial_pfix = t_inertial_pfix_at_epoch();
    let omega_pfix = DVec3::new(0.0, 0.0, EARTH.omega);

    // PAD state wrt pfix, in pfix coordinates (JEOD `compute_relative_state`).
    let r_pfix = t_inertial_pfix * pad_trans.position;
    let v_pfix = t_inertial_pfix * pad_trans.velocity - omega_pfix.cross(r_pfix);

    // Geodetic (elliptical) lat/lon from the PAD pfix position (JEOD
    // `update_from_cart` / `update_from_ellip`).
    let geo =
        astrodyn::GeodeticState::from_planet_fixed(r_pfix, EARTH.shape.r_eq(), EARTH.shape.r_pol());
    let (sin_lat, cos_lat) = geo.latitude.sin_cos();
    let (sin_lon, cos_lon) = geo.longitude.sin_cos();

    // NED-axes matrix T_pfix_ned (JEOD `build_ned_orientation`): rows North,
    // East, Down. glam `from_cols` takes columns.
    let t_pfix_ned = DMat3::from_cols(
        DVec3::new(-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon),
        DVec3::new(-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon),
        DVec3::new(cos_lat, 0.0, -sin_lat),
    );

    // S_inertial:pfix — pfix frame wrt inertial (origin coincident, rotating at
    // ω_planet expressed in pfix coordinates, +z).
    let s_inertial_pfix = RefFrameState {
        trans: RefFrameTrans {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_inertial_pfix),
            t_parent_this: t_inertial_pfix,
            ang_vel_this: omega_pfix,
        },
    };

    // S_pfix:ned — NED frame wrt pfix (origin at PAD pfix state, NED axes, zero
    // rate wrt pfix). Composed up to inertial in place.
    let mut s_inertial_ned = RefFrameState {
        trans: RefFrameTrans {
            position: r_pfix,
            velocity: v_pfix,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_pfix_ned),
            t_parent_this: t_pfix_ned,
            ang_vel_this: DVec3::ZERO,
        },
    };
    s_inertial_ned.incr_left(&s_inertial_pfix);
    s_inertial_ned
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

/// RUN_0400: ISS direct Cartesian state in `Earth.inertial` (pass-through).
fn build_run_0400(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = trans_state("ISS", "trans_TransState_inertial_body");
    build_orbinit_docker(mu, state)
}

/// RUN_0410: ISS direct Cartesian state in `Earth.pfix`. The pfix branch
/// composes the planet-fixed state into inertial including the planet
/// rotation `ω × r` velocity term.
fn build_run_0410(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = trans_state("ISS", "trans_TransState_pfix_body");
    build_orbinit_docker(mu, state)
}

/// RUN_0411: STS-114 direct Cartesian state in `Earth.pfix`.
fn build_run_0411(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = trans_state("STS_114", "trans_TransState_pfix_body");
    build_orbinit_docker(mu, state)
}

/// RUN_0202: ISS set02 (mean-anomaly) elements in `Earth.pfix`. The
/// pfix branch rotates the planet-fixed state to inertial at the SIM
/// epoch.
fn build_run_0202(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = mean_anomaly_element_state("ISS", "trans_Orbit_pfix_body_set02", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0302: STS-114 set02 (mean-anomaly) elements in `Earth.pfix`.
fn build_run_0302(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = mean_anomaly_element_state("STS_114", "trans_Orbit_pfix_body_set02", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0203: ISS set03 (semi-latus rectum + true-anomaly) elements in
/// `Earth.pfix`.
fn build_run_0203(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_element_state("ISS", "trans_Orbit_pfix_body_set03", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0303: STS-114 set03 (semi-latus rectum + true-anomaly) elements in
/// `Earth.pfix`.
fn build_run_0303(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_element_state("STS_114", "trans_Orbit_pfix_body_set03", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0204: ISS set04 (altitudes + true-anomaly) elements in `Earth.pfix`.
fn build_run_0204(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("ISS", "trans_Orbit_pfix_body_set04", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0304: STS-114 set04 (altitudes + true-anomaly) elements in `Earth.pfix`.
fn build_run_0304(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("STS_114", "trans_Orbit_pfix_body_set04", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0205: ISS set05 (altitudes + time-periapsis) elements in `Earth.pfix`.
fn build_run_0205(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_time_periapsis_state("ISS", "trans_Orbit_pfix_body_set05", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0305: STS-114 set05 (altitudes + time-periapsis) elements in `Earth.pfix`.
fn build_run_0305(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_time_periapsis_state("STS_114", "trans_Orbit_pfix_body_set05", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0206: ISS set06 (arg-latitude + radial-vel) elements in `Earth.pfix`.
fn build_run_0206(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = arg_latitude_radial_vel_state("ISS", "trans_Orbit_pfix_body_set06", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0306: STS-114 set06 (arg-latitude + radial-vel) elements in `Earth.pfix`.
fn build_run_0306(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = arg_latitude_radial_vel_state("STS_114", "trans_Orbit_pfix_body_set06", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0210: ISS set10 (sma/ecc + true-anomaly) elements in `Earth.pfix`.
fn build_run_0210(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_sma_state("ISS", "trans_Orbit_pfix_body_set10", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0310: STS-114 set10 (sma/ecc + true-anomaly) elements in `Earth.pfix`.
fn build_run_0310(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_sma_state("STS_114", "trans_Orbit_pfix_body_set10", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0211: ISS set11 (altitudes + true-anomaly) elements in `Earth.pfix`.
/// Same JEOD option as set04 (`CaseEleven`).
fn build_run_0211(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("ISS", "trans_Orbit_pfix_body_set11", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0311: STS-114 set11 (altitudes + true-anomaly) elements in `Earth.pfix`.
/// Same JEOD option as set04 (`CaseEleven`).
fn build_run_0311(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("STS_114", "trans_Orbit_pfix_body_set11", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0002: ISS set02 (mean-anomaly) elements from the committed
/// `iss.json` fixture (`trans_Orbit_inertial_body_set02`), in
/// `Earth.inertial`.
fn build_run_0002(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = mean_anomaly_element_state("ISS", "trans_Orbit_inertial_body_set02", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0102: STS-114 set02 (mean-anomaly) elements from the committed
/// `sts_114.json` fixture (`trans_Orbit_inertial_body_set02`), in
/// `Earth.inertial`.
fn build_run_0102(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = mean_anomaly_element_state("STS_114", "trans_Orbit_inertial_body_set02", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0003: ISS set03 (semi-latus rectum + true-anomaly) elements from the
/// committed `iss.json` fixture (`trans_Orbit_inertial_body_set03`), in
/// `Earth.inertial`.
fn build_run_0003(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_element_state("ISS", "trans_Orbit_inertial_body_set03", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0103: STS-114 set03 (semi-latus rectum + true-anomaly) elements from
/// the committed `sts_114.json` fixture (`trans_Orbit_inertial_body_set03`),
/// in `Earth.inertial`.
fn build_run_0103(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_element_state("STS_114", "trans_Orbit_inertial_body_set03", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0004: ISS set04 (altitudes + true-anomaly) elements from the committed
/// `iss.json` fixture (`trans_Orbit_inertial_body_set04`), in `Earth.inertial`.
fn build_run_0004(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("ISS", "trans_Orbit_inertial_body_set04", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0104: STS-114 set04 (altitudes + true-anomaly) elements from the
/// committed `sts_114.json` fixture (`trans_Orbit_inertial_body_set04`), in
/// `Earth.inertial`.
fn build_run_0104(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("STS_114", "trans_Orbit_inertial_body_set04", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0005: ISS set05 (altitudes + time-periapsis) elements from the committed
/// `iss.json` fixture (`trans_Orbit_inertial_body_set05`), in `Earth.inertial`.
fn build_run_0005(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_time_periapsis_state("ISS", "trans_Orbit_inertial_body_set05", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0105: STS-114 set05 (altitudes + time-periapsis) elements from the
/// committed `sts_114.json` fixture (`trans_Orbit_inertial_body_set05`), in
/// `Earth.inertial`.
fn build_run_0105(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_time_periapsis_state("STS_114", "trans_Orbit_inertial_body_set05", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0006: ISS set06 (arg-latitude + radial-vel) elements from the committed
/// `iss.json` fixture (`trans_Orbit_inertial_body_set06`), in `Earth.inertial`.
fn build_run_0006(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = arg_latitude_radial_vel_state("ISS", "trans_Orbit_inertial_body_set06", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0106: STS-114 set06 (arg-latitude + radial-vel) elements from the
/// committed `sts_114.json` fixture (`trans_Orbit_inertial_body_set06`), in
/// `Earth.inertial`.
fn build_run_0106(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = arg_latitude_radial_vel_state("STS_114", "trans_Orbit_inertial_body_set06", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0010: ISS set10 (sma/ecc + true-anomaly) elements from the committed
/// `iss.json` fixture (`trans_Orbit_inertial_body_set10`), in `Earth.inertial`.
fn build_run_0010(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_sma_state("ISS", "trans_Orbit_inertial_body_set10", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0110: STS-114 set10 (sma/ecc + true-anomaly) elements from the committed
/// `sts_114.json` fixture (`trans_Orbit_inertial_body_set10`), in `Earth.inertial`.
fn build_run_0110(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = true_anomaly_sma_state("STS_114", "trans_Orbit_inertial_body_set10", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0011: ISS set11 (altitudes + true-anomaly) elements from the committed
/// `iss.json` fixture (`trans_Orbit_inertial_body_set11`), in `Earth.inertial`.
/// JEOD's `CaseEleven` is the same option as set04 (`IncAscnodeAltperAltapoArgperTanom`),
/// so this reuses the set04 converter.
fn build_run_0011(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("ISS", "trans_Orbit_inertial_body_set11", mu);
    build_orbinit_docker(mu, state)
}

/// RUN_0111: STS-114 set11 (altitudes + true-anomaly) elements from the
/// committed `sts_114.json` fixture (`trans_Orbit_inertial_body_set11`), in
/// `Earth.inertial`. Same JEOD option as set04 (`CaseEleven`).
fn build_run_0111(_init: &InitialConditions) -> SimulationBuilder {
    let mu = load_mu_earth();
    let state = altitudes_true_anomaly_state("STS_114", "trans_Orbit_inertial_body_set11", mu);
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

/// RUN_0002: ISS orbital elements (set02, mean-anomaly) in `Earth.inertial`.
pub fn run_0002() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0002",
        scenario: build_run_0002,
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

/// RUN_0102: STS-114 orbital elements (set02, mean-anomaly) in `Earth.inertial`.
pub fn run_0102() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0102",
        scenario: build_run_0102,
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

/// RUN_0003: ISS orbital elements (set03, semi-latus rectum + true anomaly)
/// in `Earth.inertial`.
pub fn run_0003() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0003",
        scenario: build_run_0003,
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

/// RUN_0103: STS-114 orbital elements (set03, semi-latus rectum + true
/// anomaly) in `Earth.inertial`.
pub fn run_0103() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0103",
        scenario: build_run_0103,
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

/// RUN_0004: ISS orbital elements (set04, altitudes + true anomaly) in
/// `Earth.inertial`.
pub fn run_0004() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0004",
        scenario: build_run_0004,
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

/// RUN_0104: STS-114 orbital elements (set04, altitudes + true anomaly) in
/// `Earth.inertial`.
pub fn run_0104() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0104",
        scenario: build_run_0104,
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

/// RUN_0005: ISS orbital elements (set05, altitudes + time periapsis) in
/// `Earth.inertial`.
pub fn run_0005() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0005",
        scenario: build_run_0005,
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

/// RUN_0105: STS-114 orbital elements (set05, altitudes + time periapsis) in
/// `Earth.inertial`.
pub fn run_0105() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0105",
        scenario: build_run_0105,
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

/// RUN_0006: ISS orbital elements (set06, arg-latitude + radial-vel) in
/// `Earth.inertial`.
pub fn run_0006() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0006",
        scenario: build_run_0006,
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

/// RUN_0106: STS-114 orbital elements (set06, arg-latitude + radial-vel) in
/// `Earth.inertial`.
pub fn run_0106() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0106",
        scenario: build_run_0106,
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

/// RUN_0010: ISS orbital elements (set10, sma/ecc + true anomaly) in
/// `Earth.inertial`.
pub fn run_0010() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0010",
        scenario: build_run_0010,
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

/// RUN_0110: STS-114 orbital elements (set10, sma/ecc + true anomaly) in
/// `Earth.inertial`.
pub fn run_0110() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0110",
        scenario: build_run_0110,
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

/// RUN_0011: ISS orbital elements (set11, altitudes + true anomaly) in
/// `Earth.inertial`. Same JEOD option as set04 (`CaseEleven`).
pub fn run_0011() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0011",
        scenario: build_run_0011,
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

/// RUN_0111: STS-114 orbital elements (set11, altitudes + true anomaly) in
/// `Earth.inertial`. Same JEOD option as set04 (`CaseEleven`).
pub fn run_0111() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0111",
        scenario: build_run_0111,
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

/// RUN_0202: ISS orbital elements (set02, mean-anomaly) in `Earth.pfix`.
pub fn run_0202() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0202",
        scenario: build_run_0202,
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

/// RUN_0302: STS-114 orbital elements (set02, mean-anomaly) in `Earth.pfix`.
pub fn run_0302() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0302",
        scenario: build_run_0302,
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

/// RUN_0203: ISS orbital elements (set03, semi-latus rectum + true anomaly)
/// in `Earth.pfix`.
pub fn run_0203() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0203",
        scenario: build_run_0203,
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

/// RUN_0303: STS-114 orbital elements (set03, semi-latus rectum + true
/// anomaly) in `Earth.pfix`.
pub fn run_0303() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0303",
        scenario: build_run_0303,
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

/// RUN_0204: ISS orbital elements (set04, altitudes + true anomaly) in
/// `Earth.pfix`.
pub fn run_0204() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0204",
        scenario: build_run_0204,
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

/// RUN_0304: STS-114 orbital elements (set04, altitudes + true anomaly) in
/// `Earth.pfix`.
pub fn run_0304() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0304",
        scenario: build_run_0304,
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

/// RUN_0205: ISS orbital elements (set05, altitudes + time periapsis) in
/// `Earth.pfix`.
pub fn run_0205() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0205",
        scenario: build_run_0205,
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

/// RUN_0305: STS-114 orbital elements (set05, altitudes + time periapsis) in
/// `Earth.pfix`.
pub fn run_0305() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0305",
        scenario: build_run_0305,
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

/// RUN_0206: ISS orbital elements (set06, arg-latitude + radial-vel) in
/// `Earth.pfix`. The pfix branch applies the inertial↔pfix rotation at the
/// SIM epoch.
pub fn run_0206() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0206",
        scenario: build_run_0206,
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

/// RUN_0306: STS-114 orbital elements (set06, arg-latitude + radial-vel) in
/// `Earth.pfix`.
pub fn run_0306() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0306",
        scenario: build_run_0306,
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

/// RUN_0210: ISS orbital elements (set10, sma/ecc + true anomaly) in
/// `Earth.pfix`.
pub fn run_0210() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0210",
        scenario: build_run_0210,
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

/// RUN_0310: STS-114 orbital elements (set10, sma/ecc + true anomaly) in
/// `Earth.pfix`.
pub fn run_0310() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0310",
        scenario: build_run_0310,
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

/// RUN_0211: ISS orbital elements (set11, altitudes + true anomaly) in
/// `Earth.pfix`. Same JEOD option as set04 (`CaseEleven`).
pub fn run_0211() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0211",
        scenario: build_run_0211,
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

/// RUN_0311: STS-114 orbital elements (set11, altitudes + true anomaly) in
/// `Earth.pfix`. Same JEOD option as set04 (`CaseEleven`).
pub fn run_0311() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0311",
        scenario: build_run_0311,
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

/// RUN_0400: ISS direct Cartesian state (`DynBodyInitTransState`) in
/// `Earth.inertial`. Pass-through of the literal position/velocity from
/// `trans_TransState_inertial_body`.
pub fn run_0400() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0400",
        scenario: build_run_0400,
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

/// RUN_0410: ISS direct Cartesian state (`DynBodyInitTransState`) in
/// `Earth.pfix`. The pfix branch composes the planet-fixed state into
/// inertial with the planet-rotation velocity term.
pub fn run_0410() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0410",
        scenario: build_run_0410,
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

/// RUN_0411: STS-114 direct Cartesian state (`DynBodyInitTransState`) in
/// `Earth.pfix`.
pub fn run_0411() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0411",
        scenario: build_run_0411,
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

/// RUN_2100: ISS inertial Cartesian translation + direct inertial
/// attitude/rate initialization (`DynBodyInitRotState`,
/// `Earth.inertial`). The first *rotational* RUN in SIM_orbinit: the
/// recipe attaches a 6-DOF body whose initial attitude is the
/// Yaw-Pitch-Roll Euler triple `[77.59, -30.60, -46.10]` deg and whose
/// body-frame rate is the inertial rate from the ISS / LVLH rate decks.
pub fn run_2100() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_2100",
        scenario: build_run_2100,
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

/// RUN_1230: ISS inertial Cartesian translation + LVLH-relative
/// attitude/rate initialization (`DynBodyInitLvlhRotState`, Earth). The
/// body is aligned with the reference orbit's LVLH frame (identity
/// LVLH→body) with an LVLH-relative body rate; `init_rot_from_lvlh`
/// composes that with the LVLH frame's own orientation / angular
/// velocity wrt inertial.
pub fn run_1230() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_1230",
        scenario: build_run_1230,
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

/// RUN_0441: STS-114 chaser, translation in the ISS composite-body frame.
pub fn run_0441() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0441",
        scenario: build_run_0441,
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

/// RUN_0571: STS-114 chaser, translation in the ISS LVLH frame.
pub fn run_0571() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0571",
        scenario: build_run_0571,
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

/// RUN_0681: STS-114 chaser, translation in the NED frame relative to ISS.
pub fn run_0681() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_0681",
        scenario: build_run_0681,
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

/// RUN_3822: PAD_39A single vehicle, full state in the local NED frame at a
/// geodetic ground point.
pub fn run_3822() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_3822",
        scenario: build_run_3822,
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

/// RUN_3771: STS-114 chaser, full state (pos/vel/att/rate) in the ISS LVLH frame.
pub fn run_3771() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_3771",
        scenario: build_run_3771,
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

/// RUN_4451: STS-114 chaser, full state in the ISS structure frame.
pub fn run_4451() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_4451",
        scenario: build_run_4451,
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

/// RUN_5461: STS-114 chaser, mixed-reference init (pos/att rel ISS attach_point,
/// rate rel ISS LVLH).
pub fn run_5461() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_5461",
        scenario: build_run_5461,
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

/// RUN_4681: STS-114 chaser, full state in the NED frame relative to PAD_39A.
pub fn run_4681() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_docker_run_4681",
        scenario: build_run_4681,
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
