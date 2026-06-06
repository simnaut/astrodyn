//! `VerificationCase` constructors for the orbinit-round-trip analytical
//! family (`tier3_sim_orbinit_roundtrip`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "verif step counts bounded by Tier 3 propagation span (<< usize / f64 mantissa)"
)]
//!
//! These cases have no JEOD reference CSV — they exercise the closed
//! identity that Cartesian↔Keplerian conversion plus a full pipeline
//! propagation under point-mass gravity returns to the same Keplerian
//! shape/orientation it started from. The recipes share a single
//! point-mass Earth source + `from_cartesian`-initialised body; only
//! the per-case initial state and propagation length differ, so the
//! per-case factories all delegate to a shared scenario constructor
//! and pair the resulting [`SimulationBuilder`] with
//! [`CsvReference::SyntheticTimes`] for the parity trait's lockstep
//! `runner ↔ bevy` bit-identity assertion.
//!
//! The matching analytical assertions live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_orbinit_roundtrip.rs`;
//! each tier3 test pulls one recipe's scenario factory, builds the
//! `Simulation`, propagates, and asserts the closed-form round-trip
//! property (shape/orientation elements, or specific energy for the
//! near-circular case). Splitting the scenario into a recipe is what
//! makes the parity wrapper possible — the bridge needs an
//! adapter-neutral `SimulationBuilder` to materialise, and a
//! hand-rolled tier3 test that constructs a `Simulation` directly has
//! no bridge entry point.

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::recipes::helpers::state_helpers::state_from_elements;
use astrodyn::{
    default_leap_second_table, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, RotationModel, SimulationBuilder, SimulationTime,
    TranslationalState, VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

/// Earth gravitational parameter (m³/s²) — JEOD `earth_GGM05C.cc`. The
/// `astrodyn::EARTH.shape.mu` source is a compile-time literal, so
/// every recipe and every tier3 closed-form assertion that references
/// `MU_EARTH` resolves to the same bit-pattern without any fixture
/// decode.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth equatorial radius (m) — JEOD `earth.cc`. Same const-folding
/// rationale as [`MU_EARTH`].
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq();

/// Integrator step size shared by every full-period recipe. The
/// hyperbolic case keeps its own short-propagation `dt` because its
/// trajectory is unbounded, not period-keyed.
const DT_S: f64 = 10.0;

/// Hyperbolic recipe step size — shorter than [`DT_S`] because the
/// hyperbolic flyby propagates only 100 s and the round-trip is
/// asserted at the recovered-element level rather than the full-period
/// level.
const DT_HYP_S: f64 = 1.0;

/// Hyperbolic recipe step count — fixed 100-step short flyby. The
/// round-trip identity is independent of propagation length once the
/// body has been integrated a non-trivial distance, so the choice is
/// only a function of how many steps are needed to exercise the
/// pipeline.
const HYP_NUM_STEPS: usize = 100;

/// Closed-form circular-orbit period for semi-major axis `a` and
/// gravitational parameter `mu`. Used to size SyntheticTimes cadences
/// for the full-period scans.
fn period_s(mu_earth: f64, a: f64) -> f64 {
    2.0 * std::f64::consts::PI * (a * a * a / mu_earth).sqrt()
}

/// Cadence for a full-period propagation at [`DT_S`]. `.ceil()` so the
/// scan actually covers a full orbital period — bare truncation would
/// stop a few seconds short of the period boundary.
fn full_period_num_steps(a: f64) -> usize {
    (period_s(MU_EARTH, a) / DT_S).ceil() as usize
}

/// Shared scenario builder for every orbinit round-trip recipe.
/// Parameterised by:
///   * `dt` — integrator timestep (`DT_S` for bound orbits, `DT_HYP_S`
///     for the hyperbolic flyby);
///   * `body` — the vehicle's initial translational state in
///     `RootInertial`.
///
/// Each recipe wraps this with its case-specific values and pairs the
/// returned builder with the matching `SyntheticTimes` cadence so the
/// parity trait can drive `runner ↔ bevy` bit-identity at every step.
fn build_orbinit_roundtrip(dt: f64, body: TranslationalState) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
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
        ..VehicleConfig::named("sim-orbinit-roundtrip-0")
    });
    sb
}

/// Analytical recipes opt out of every runner-vs-JEOD tolerance group
/// because they pair with [`CsvReference::SyntheticTimes`] and assert
/// in-test against closed-form values rather than logged JEOD columns.
/// The parity trait still asserts `runner ↔ bevy` bit-identity at every
/// synthetic record.
fn analytical_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Build a `VerificationCase` from a case name, scenario constructor,
/// step size, and step count. All orbinit-round-trip cases share the
/// same Earth-point-mass shape, analytical-tolerance opt-out, and lack
/// of pre-step hooks, so the per-case factory bodies all collapse to
/// this single helper.
fn make_case(
    name: &'static str,
    scenario: fn(&InitialConditions) -> SimulationBuilder,
    dt: f64,
    num_steps: usize,
) -> VerificationCase {
    VerificationCase {
        name,
        scenario,
        reference: CsvReference::SyntheticTimes { dt, num_steps },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Circular LEO ─────────────────────────────────────────────────────

/// Initial orbital elements for the circular recipe. Held as a
/// function rather than a const so the tier3 closed-form assertion and
/// the recipe builder share the single source of truth without an
/// extra public `pub const` widening this module's surface.
fn circular_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 400_000.0;
    let e = 0.0;
    let i = 51.6_f64.to_radians();
    let raan = 30.0_f64.to_radians();
    let argp = 0.0;
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_circular(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = circular_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_S, trans)
}

/// Circular LEO (a = R_E + 400 km, e = 0, i = 51.6°) propagated for one
/// full orbital period. The matching tier3 test asserts the
/// branch-independent specific energy `E = v²/2 − μ/r` returned to its
/// initial value, plus inclination and RAAN preservation. Eccentricity
/// recovery is also asserted but kept loose because `from_cartesian`
/// switches branches near `e ≈ 1e-13`.
pub fn circular() -> VerificationCase {
    let (a, ..) = circular_elements();
    make_case(
        "tier3_orbinit_roundtrip_circular",
        build_circular,
        DT_S,
        full_period_num_steps(a),
    )
}

// ── Eccentric (e = 0.3) ──────────────────────────────────────────────

fn eccentric_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 2_000_000.0;
    let e = 0.3;
    let i = 28.5_f64.to_radians();
    let raan = 45.0_f64.to_radians();
    let argp = 90.0_f64.to_radians();
    let nu = 0.0; // periapsis start — cleanest closed-form round-trip
    (a, e, i, raan, argp, nu)
}

fn build_eccentric(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = eccentric_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_S, trans)
}

/// Eccentric orbit (e = 0.3, i = 28.5°) propagated for one full
/// orbital period. The tier3 test asserts the shape/orientation
/// elements (a, e, i, RAAN, argp) recover to their initial values.
pub fn eccentric() -> VerificationCase {
    let (a, ..) = eccentric_elements();
    make_case(
        "tier3_orbinit_roundtrip_eccentric",
        build_eccentric,
        DT_S,
        full_period_num_steps(a),
    )
}

// ── Retrograde ───────────────────────────────────────────────────────

fn retrograde_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 800_000.0;
    let e = 0.05;
    let i = 150.0_f64.to_radians();
    let raan = 200.0_f64.to_radians();
    let argp = 30.0_f64.to_radians();
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_retrograde(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = retrograde_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_S, trans)
}

/// Retrograde orbit (i = 150°). Exercises the inclination > 90° branch
/// of the Cartesian→Keplerian recovery — the angular-momentum vector
/// flips sign relative to a prograde orbit but the recovered
/// inclination still lies in `[0, π]`.
pub fn retrograde() -> VerificationCase {
    let (a, ..) = retrograde_elements();
    make_case(
        "tier3_orbinit_roundtrip_retrograde",
        build_retrograde,
        DT_S,
        full_period_num_steps(a),
    )
}

// ── Equatorial ───────────────────────────────────────────────────────

fn equatorial_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 600_000.0;
    let e = 0.1;
    let i = 0.0;
    let raan = 0.0;
    let argp = 45.0_f64.to_radians();
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_equatorial(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = equatorial_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_S, trans)
}

/// Equatorial orbit (i = 0). The tier3 test skips RAAN because it is
/// geometrically undefined when the orbit plane coincides with the
/// equator (the line of nodes vanishes); shape elements still recover
/// cleanly.
pub fn equatorial() -> VerificationCase {
    let (a, ..) = equatorial_elements();
    make_case(
        "tier3_orbinit_roundtrip_equatorial",
        build_equatorial,
        DT_S,
        full_period_num_steps(a),
    )
}

// ── Polar ────────────────────────────────────────────────────────────

fn polar_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 500_000.0;
    let e = 0.02;
    let i = 90.0_f64.to_radians();
    let raan = 60.0_f64.to_radians();
    let argp = 0.0;
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_polar(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = polar_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_S, trans)
}

/// Polar orbit (i = 90°). Exercises the boundary where the angular-
/// momentum z-component is zero; shape and orientation elements still
/// recover to their initial values.
pub fn polar() -> VerificationCase {
    let (a, ..) = polar_elements();
    make_case(
        "tier3_orbinit_roundtrip_polar",
        build_polar,
        DT_S,
        full_period_num_steps(a),
    )
}

// ── Molniya (high-eccentricity) ──────────────────────────────────────

fn molniya_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 10_000_000.0;
    let e = 0.7;
    let i = 63.4_f64.to_radians();
    let raan = 120.0_f64.to_radians();
    let argp = 270.0_f64.to_radians();
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_molniya(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = molniya_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_S, trans)
}

/// Highly eccentric Molniya-like orbit (e = 0.7, i = 63.4°). The
/// tier3 test relaxes the per-element tolerances slightly to cover the
/// integrator error budget over a single long-period orbit.
pub fn molniya() -> VerificationCase {
    let (a, ..) = molniya_elements();
    make_case(
        "tier3_orbinit_roundtrip_molniya",
        build_molniya,
        DT_S,
        full_period_num_steps(a),
    )
}

// ── Hyperbolic (short flyby) ─────────────────────────────────────────

fn hyperbolic_elements() -> (f64, f64, f64, f64, f64, f64) {
    let e = 1.5;
    let r_peri = R_EARTH + 300_000.0;
    let a = -(r_peri / (e - 1.0));
    let i = 30.0_f64.to_radians();
    let raan = 0.0;
    let argp = 0.0;
    let nu = 0.1;
    (a, e, i, raan, argp, nu)
}

fn build_hyperbolic(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = hyperbolic_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_roundtrip(DT_HYP_S, trans)
}

/// Hyperbolic flyby (e = 1.5) over a 100 s short propagation. With
/// `e > 1` there is no bounded period, so the recipe uses a fixed
/// 100-step horizon at the hyperbolic step size. The tier3 test
/// asserts the recovered elements still match the initial Keplerian
/// shape — for a hyperbola `a < 0` and the parabolic / elliptic
/// branch boundaries don't apply.
pub fn hyperbolic() -> VerificationCase {
    make_case(
        "tier3_orbinit_roundtrip_hyperbolic",
        build_hyperbolic,
        DT_HYP_S,
        HYP_NUM_STEPS,
    )
}

/// Initial orbital-element tuples exposed for the tier3 closed-form
/// assertions. Returning the same `(a, e, i, raan, argp, nu)` shape
/// that drives [`state_from_elements`] inside each builder keeps the
/// tier3 file's expected-value vector and the recipe's built initial
/// state on the same single source of truth.
pub mod elements {
    /// Initial orbital elements for the [`super::circular`] recipe.
    pub fn circular() -> (f64, f64, f64, f64, f64, f64) {
        super::circular_elements()
    }
    /// Initial orbital elements for the [`super::eccentric`] recipe.
    pub fn eccentric() -> (f64, f64, f64, f64, f64, f64) {
        super::eccentric_elements()
    }
    /// Initial orbital elements for the [`super::retrograde`] recipe.
    pub fn retrograde() -> (f64, f64, f64, f64, f64, f64) {
        super::retrograde_elements()
    }
    /// Initial orbital elements for the [`super::equatorial`] recipe.
    pub fn equatorial() -> (f64, f64, f64, f64, f64, f64) {
        super::equatorial_elements()
    }
    /// Initial orbital elements for the [`super::polar`] recipe.
    pub fn polar() -> (f64, f64, f64, f64, f64, f64) {
        super::polar_elements()
    }
    /// Initial orbital elements for the [`super::molniya`] recipe.
    pub fn molniya() -> (f64, f64, f64, f64, f64, f64) {
        super::molniya_elements()
    }
    /// Initial orbital elements for the [`super::hyperbolic`] recipe.
    pub fn hyperbolic() -> (f64, f64, f64, f64, f64, f64) {
        super::hyperbolic_elements()
    }
}
