//! `VerificationCase` constructors for the orbinit-families
//! conservation scans (`tier3_sim_orbinit_families`).
//!
//! Like the round-trip recipes in
//! [`super::sim_orbinit_roundtrip`], these cases have no JEOD reference
//! CSV — they exercise analytical invariants (specific orbital energy
//! and angular momentum conservation under point-mass gravity, plus
//! per-family geometric checks). Splitting the [`SimulationBuilder`]
//! construction into a recipe is what makes the parity wrapper
//! possible: the bridge needs an adapter-neutral builder to
//! materialise, and a hand-rolled tier3 test that constructs a
//! `Simulation` directly has no entry point.
//!
//! The per-case factories all delegate to a shared scenario
//! constructor and pair the resulting [`SimulationBuilder`] with
//! [`CsvReference::SyntheticTimes`] so the parity trait can drive
//! `runner ↔ bevy` bit-identity at every step. The matching analytical
//! assertions live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_orbinit_families.rs`;
//! each tier3 test pulls one recipe's scenario factory, builds the
//! `Simulation`, propagates, and asserts energy + angular momentum
//! conservation (plus the case-specific geometric invariant: radius
//! constancy for circular, periapsis/apoapsis bounds for elliptic,
//! equatorial-plane confinement, polar passage, hyperbolic escape, …).

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
/// every recipe and every tier3 conservation assertion that references
/// `MU_EARTH` resolves to the same bit-pattern without any fixture
/// decode.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth equatorial radius (m) — JEOD `earth.cc`. Same const-folding
/// rationale as [`MU_EARTH`].
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq;

/// Integrator step size shared by every bound-orbit recipe. The
/// hyperbolic and near-parabolic flybys keep a shorter step size
/// because their trajectories are unbounded / near-singular and the
/// scans propagate a fixed wall-clock horizon rather than a closed
/// period.
const DT_S: f64 = 10.0;

/// Short step size for the hyperbolic and near-parabolic recipes. The
/// short flyby horizon means closed-period sizing doesn't apply — the
/// step count is fixed in absolute terms instead.
const DT_SHORT_S: f64 = 1.0;

/// Hyperbolic recipe step count — 600-step (10-minute) horizon at
/// [`DT_SHORT_S`]. Long enough to exercise the integrator past
/// periapsis but short enough to keep the body within the integration
/// region.
const HYP_NUM_STEPS: usize = 600;

/// Near-parabolic recipe step count — 300-step (5-minute) horizon at
/// [`DT_SHORT_S`]. Shorter than the hyperbolic case because the
/// near-parabolic orbit lingers near periapsis where the energy
/// conservation metric is most sensitive to integrator step size.
const NEAR_PARABOLIC_NUM_STEPS: usize = 300;

/// Closed-form circular-orbit period for semi-major axis `a` and
/// gravitational parameter `mu`. Used to size SyntheticTimes cadences
/// for the bound-orbit scans.
fn period_s(mu_earth: f64, a: f64) -> f64 {
    2.0 * std::f64::consts::PI * (a * a * a / mu_earth).sqrt()
}

/// Cadence for an `n_orbits`-period propagation at [`DT_S`]. `.ceil()`
/// so the scan actually covers the requested period count — bare
/// truncation would stop a few seconds short of the final period
/// boundary.
fn num_steps_for_orbits(a: f64, n_orbits: f64) -> usize {
    (n_orbits * period_s(MU_EARTH, a) / DT_S).ceil() as usize
}

/// Shared scenario builder for every orbinit-families recipe.
/// Parameterised by:
///   * `dt` — integrator timestep ([`DT_S`] for bound orbits,
///     [`DT_SHORT_S`] for the hyperbolic and near-parabolic flybys);
///   * `body` — the vehicle's initial translational state in
///     `RootInertial`.
///
/// Each recipe wraps this with its case-specific values and pairs the
/// returned builder with the matching [`CsvReference::SyntheticTimes`]
/// cadence so the parity trait can drive `runner ↔ bevy` bit-identity
/// at every step.
fn build_orbinit_families(dt: f64, body: TranslationalState) -> SimulationBuilder {
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
        ..Default::default()
    });
    sb
}

/// Analytical recipes opt out of every runner-vs-JEOD tolerance group
/// because they pair with [`CsvReference::SyntheticTimes`] and assert
/// in-test against closed-form invariants rather than logged JEOD
/// columns. The parity trait still asserts `runner ↔ bevy`
/// bit-identity at every synthetic record.
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
/// step size, and step count. All orbinit-families cases share the
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
/// function rather than a const so the tier3 conservation assertion and
/// the recipe builder share the single source of truth without an
/// extra public `pub const` widening this module's surface.
fn circular_leo_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 400_000.0;
    let e = 0.0;
    let i = 51.6_f64.to_radians();
    let raan = 30.0_f64.to_radians();
    let argp = 0.0;
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_circular_leo(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = circular_leo_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_S, trans)
}

/// Circular LEO (a = R_E + 400 km, e = 0, i = 51.6°) propagated for
/// two full orbital periods. The matching tier3 test asserts specific
/// energy and angular momentum conservation plus radius constancy
/// (max relative deviation from the initial radius).
pub fn circular_leo() -> VerificationCase {
    let (a, ..) = circular_leo_elements();
    make_case(
        "tier3_orbinit_circular_leo",
        build_circular_leo,
        DT_S,
        num_steps_for_orbits(a, 2.0),
    )
}

// ── Eccentric (e = 0.3) ──────────────────────────────────────────────

fn eccentric_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 2_000_000.0;
    let e = 0.3;
    let i = 28.5_f64.to_radians();
    let raan = 45.0_f64.to_radians();
    let argp = 90.0_f64.to_radians();
    let nu = 60.0_f64.to_radians();
    (a, e, i, raan, argp, nu)
}

fn build_eccentric(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = eccentric_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_S, trans)
}

/// Eccentric orbit (e = 0.3, i = 28.5°) propagated for two orbital
/// periods. The tier3 test asserts energy + angular momentum
/// conservation plus periapsis/apoapsis radius bounds.
pub fn eccentric() -> VerificationCase {
    let (a, ..) = eccentric_elements();
    make_case(
        "tier3_orbinit_eccentric",
        build_eccentric,
        DT_S,
        num_steps_for_orbits(a, 2.0),
    )
}

// ── Highly eccentric (e = 0.7, Molniya-like) ─────────────────────────

fn highly_eccentric_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 10_000_000.0;
    let e = 0.7;
    let i = 63.4_f64.to_radians();
    let raan = 120.0_f64.to_radians();
    let argp = 270.0_f64.to_radians();
    let nu = 0.0;
    (a, e, i, raan, argp, nu)
}

fn build_highly_eccentric(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = highly_eccentric_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_S, trans)
}

/// Highly eccentric orbit (e = 0.7, i = 63.4°) propagated for a
/// single long-period orbit. The tier3 test asserts conservation with
/// a relaxed energy tolerance to accommodate the integrator error
/// budget over a long-period orbit.
pub fn highly_eccentric() -> VerificationCase {
    let (a, ..) = highly_eccentric_elements();
    make_case(
        "tier3_orbinit_highly_eccentric",
        build_highly_eccentric,
        DT_S,
        num_steps_for_orbits(a, 1.0),
    )
}

// ── Retrograde ───────────────────────────────────────────────────────

fn retrograde_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 800_000.0;
    let e = 0.05;
    let i = 150.0_f64.to_radians();
    let raan = 200.0_f64.to_radians();
    let argp = 30.0_f64.to_radians();
    let nu = 180.0_f64.to_radians();
    (a, e, i, raan, argp, nu)
}

fn build_retrograde(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = retrograde_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_S, trans)
}

/// Retrograde orbit (i = 150°) propagated for two orbital periods. In
/// addition to the shared conservation invariants, the tier3 test
/// asserts the angular-momentum z-component stays negative — the
/// definitional geometric property of a retrograde orbit.
pub fn retrograde() -> VerificationCase {
    let (a, ..) = retrograde_elements();
    make_case(
        "tier3_orbinit_retrograde",
        build_retrograde,
        DT_S,
        num_steps_for_orbits(a, 2.0),
    )
}

// ── Equatorial ───────────────────────────────────────────────────────

fn equatorial_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 600_000.0;
    let e = 0.1;
    let i = 0.0;
    let raan = 0.0;
    let argp = 45.0_f64.to_radians();
    let nu = 90.0_f64.to_radians();
    (a, e, i, raan, argp, nu)
}

fn build_equatorial(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = equatorial_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_S, trans)
}

/// Equatorial orbit (i = 0). The tier3 test asserts conservation and
/// that the orbit stays in the equatorial plane (`max |z|/r` near
/// zero across all steps).
pub fn equatorial() -> VerificationCase {
    let (a, ..) = equatorial_elements();
    make_case(
        "tier3_orbinit_equatorial",
        build_equatorial,
        DT_S,
        num_steps_for_orbits(a, 2.0),
    )
}

// ── Polar ────────────────────────────────────────────────────────────

fn polar_elements() -> (f64, f64, f64, f64, f64, f64) {
    let a = R_EARTH + 500_000.0;
    let e = 0.02;
    let i = 90.0_f64.to_radians();
    let raan = 60.0_f64.to_radians();
    let argp = 0.0;
    let nu = 45.0_f64.to_radians();
    (a, e, i, raan, argp, nu)
}

fn build_polar(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = polar_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_S, trans)
}

/// Polar orbit (i = 90°). The tier3 test asserts conservation, that
/// the orbit passes over the poles (`max |z|/r > 0.5`), and that the
/// angular-momentum z-component stays near zero (consistent with a
/// polar orbit plane).
pub fn polar() -> VerificationCase {
    let (a, ..) = polar_elements();
    make_case(
        "tier3_orbinit_polar",
        build_polar,
        DT_S,
        num_steps_for_orbits(a, 2.0),
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
    build_orbinit_families(DT_SHORT_S, trans)
}

/// Hyperbolic flyby (e = 1.5, periapsis at 300 km altitude) over a
/// 10-minute short propagation. The tier3 test asserts positive
/// specific energy (the hyperbolic-branch invariant), conservation of
/// energy + angular momentum, and a final radius well past periapsis
/// (the body is escaping).
pub fn hyperbolic() -> VerificationCase {
    make_case(
        "tier3_orbinit_hyperbolic",
        build_hyperbolic,
        DT_SHORT_S,
        HYP_NUM_STEPS,
    )
}

// ── Near-parabolic (e ≈ 1) ───────────────────────────────────────────

fn near_parabolic_elements() -> (f64, f64, f64, f64, f64, f64) {
    // `e` stays within `ORBIT_SWITCH_TOL` (1e-2) so this case remains
    // in JEOD's near-parabolic branch.
    let e = 1.005;
    let r_peri = R_EARTH + 500_000.0;
    let a = -(r_peri / (e - 1.0));
    let i = 10.0_f64.to_radians();
    let raan = 0.0;
    let argp = 0.0;
    let nu = 0.05;
    (a, e, i, raan, argp, nu)
}

fn build_near_parabolic(_init: &InitialConditions) -> SimulationBuilder {
    let (a, e, i, raan, argp, nu) = near_parabolic_elements();
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);
    build_orbinit_families(DT_SHORT_S, trans)
}

/// Near-parabolic orbit (e ≈ 1.005, periapsis at 500 km altitude)
/// over a 5-minute short propagation. The energy-conservation metric
/// switches from relative to `μ/r`-normalised because `|E₀|` is near
/// zero and the standard relative-error formulation is
/// ill-conditioned.
pub fn near_parabolic() -> VerificationCase {
    make_case(
        "tier3_orbinit_near_parabolic",
        build_near_parabolic,
        DT_SHORT_S,
        NEAR_PARABOLIC_NUM_STEPS,
    )
}

/// Initial orbital-element tuples exposed for the tier3 conservation
/// assertions. Returning the same `(a, e, i, raan, argp, nu)` shape
/// that drives [`state_from_elements`] inside each builder keeps the
/// tier3 file's expected-value bounds and the recipe's built initial
/// state on the same single source of truth.
pub mod elements {
    /// Initial orbital elements for the [`super::circular_leo`] recipe.
    pub fn circular_leo() -> (f64, f64, f64, f64, f64, f64) {
        super::circular_leo_elements()
    }
    /// Initial orbital elements for the [`super::eccentric`] recipe.
    pub fn eccentric() -> (f64, f64, f64, f64, f64, f64) {
        super::eccentric_elements()
    }
    /// Initial orbital elements for the [`super::highly_eccentric`]
    /// recipe.
    pub fn highly_eccentric() -> (f64, f64, f64, f64, f64, f64) {
        super::highly_eccentric_elements()
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
    /// Initial orbital elements for the [`super::hyperbolic`] recipe.
    pub fn hyperbolic() -> (f64, f64, f64, f64, f64, f64) {
        super::hyperbolic_elements()
    }
    /// Initial orbital elements for the [`super::near_parabolic`]
    /// recipe.
    pub fn near_parabolic() -> (f64, f64, f64, f64, f64, f64) {
        super::near_parabolic_elements()
    }
}
