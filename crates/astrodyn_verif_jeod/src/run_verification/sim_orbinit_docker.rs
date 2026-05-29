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
    calendar_to_tjt, compute_t_parent_this_from_tjt, default_leap_second_table,
    init_from_altitudes_time_periapsis, init_from_altitudes_true_anomaly,
    init_from_arg_latitude_radial_vel, init_from_mean_anomaly, init_from_orbital_elements,
    init_from_semi_latus_rectum_true_anomaly, ut1_to_gmst_seconds, CalendarDate, GravityControl,
    GravityControls, GravityGradient, GravityModel, GravitySource, GravitySourceEntry,
    RotationModel, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig, EARTH,
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
