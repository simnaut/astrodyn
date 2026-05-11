//! `VerificationCase` constructors for the SIM_OrbElem comprehensive
//! sweep (`tier3_sim_orbelem_comprehensive`).
//!
//! These cases exercise the `Simulation` pipeline from the t=0 row of
//! each `orbelem_verif_tXX_orbelem.csv` (JEOD's verification output for
//! the orbital-elements derived state across seven orbit families:
//! circular, eccentric, hyperbolic, near-parabolic, retrograde,
//! equatorial, polar). The per-RUN initial inertial position/velocity
//! is fed to a point-mass-Earth `Simulation` configured with the
//! `OrbitalElementsConfigC`-equivalent `DerivedStateConfig`, propagated
//! a single step at a tiny `dt`, and the resulting orbital elements
//! are compared against the JEOD-logged columns by the tier3 sibling.
//!
//! The integrator step size is intentionally tiny (`DT_S = 1e-9`):
//! one step barely moves the state, but it does drive the full pipeline
//! through to the stage-9 derived-state computation. With a normal
//! step the orbital elements drift away from the JEOD-logged t=0 row
//! by enough that the tight per-element tolerances would fail.
//!
//! Each recipe pairs with [`CsvReference::SyntheticTimes`] so the
//! parity wrapper drives a lockstep `runner ↔ bevy` bit-identity
//! assertion at the synthetic checkpoint without consulting the
//! reference CSV at all. (The orbelem fixture CSVs *do* contain
//! multiple rows — most are just t=0..2, but t50 spans t=0..5000 —
//! but the per-case tier3 assertions only need the JEOD-logged
//! initial state, which is baked into each recipe directly rather
//! than parsed at runtime.)
//!
//! Initial state values originate from the JEOD output rows of
//! `crates/astrodyn_verif_jeod/test_data/orbelem_verif_tXX_orbelem.csv`,
//! committed alongside the rest of the verif fixtures.

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    default_leap_second_table, DerivedStateConfig, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, RotationModel,
    SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::DVec3;
use uom::si::f64::Time;
use uom::si::time::second;

/// Integrator step size. Matches the value the pre-recipe tier3 file
/// used (`Simulation::new(time, 1e-9)`) — one tiny step exercises the
/// pipeline through stage-9 derived-state computation while keeping
/// the state drift below ≈1e-5 m so the JEOD-logged orbital elements
/// stay within the per-element tolerances.
const DT_S: f64 = 1e-9;

/// One propagation step is enough to trigger derived-state computation
/// — the original tier3 file called `sim.step()` exactly once after
/// construction, so a single synthetic-cadence checkpoint preserves
/// the test's coverage shape.
const NUM_STEPS: usize = 1;

fn load_mu_earth() -> f64 {
    // Cache the decoded `mu` for the lifetime of the test process —
    // every `build_*` call would otherwise re-parse the full GGM05C
    // coefficient table (≈12 KiB) just to read a single scalar.
    static MU_EARTH: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MU_EARTH.get_or_init(|| astrodyn::gravity_fixtures::load_ggm05c().mu)
}

/// Shared scenario builder for every recipe. Parameterised by:
///   * `mu_earth` — Earth's gravitational parameter (point-mass);
///   * `body` — the vehicle's initial translational state in the
///     RootInertial frame.
///
/// The body is configured with `orbital_elements_source` so stage 9
/// produces the per-step `OrbitalElements` the tier3 sibling reads.
fn build_orbelem_comprehensive(mu_earth: f64, body: TranslationalState) -> SimulationBuilder {
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
        derived: DerivedStateConfig {
            orbital_elements_source: Some(earth),
            ..Default::default()
        },
        ..Default::default()
    });
    sb
}

/// Recipes opt out of every runner-vs-JEOD tolerance group because they
/// pair with [`CsvReference::SyntheticTimes`]; the tier3 sibling asserts
/// per-orbital-element bounds against the JEOD-logged CSV columns
/// directly, and the parity trait drives bit-identity at every
/// synthetic record.
fn synthetic_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

// ── Per-RUN initial states ───────────────────────────────────────────
//
// Values are copied verbatim from the t=0 row of each
// `orbelem_verif_tXX_orbelem.csv` (the JEOD verification output).
// The case IDs (T01, T10, ...) are JEOD's RUN labels; the
// per-case descriptors below summarise the actual orbital elements
// JEOD wrote for that row (rather than the family the RUN was
// originally designed for, which sometimes diverges from the
// committed fixture — see #389 thread).
//
//   * T01 — equatorial circular (e=0, i=0, sma≈7378 km);
//   * T10 — inclined near-circular (i≈30°, e≈0, sma≈7378 km);
//   * T20 — eccentric inclined (e=0.2, i≈45°, sma≈9223 km);
//   * T30 — orb_elem edge case (sma=0 in JEOD output despite a
//           valid 7.2 Mm Cartesian state; exercises a degenerate
//           branch of the orbital-element extraction);
//   * T40 — orb_elem edge case with eccentricity (sma=0 in
//           JEOD output, e=0.2, i≈45°);
//   * T50 — equatorial circular with extended log (matches T01
//           initial state, but the fixture spans t=0..5000s);
//   * T55 — ISS-like LEO (i≈51.7°, e≈0.0025, sma≈6733 km).

fn t01_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(7_378_145.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7_350.141_635_341_643, 0.0),
    }
}

fn t10_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(-4_518_172.624_566_75, 4_518_172.624_566_75, 3_689_072.5),
        velocity: DVec3::new(-5_197.334_993_031_66, -5_197.334_993_031_66, 0.0),
    }
}

fn t20_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5_217_136.362_077_619, -5_217_136.362_077_62, 0.0),
        velocity: DVec3::new(
            4_025.838_374_534_529,
            4_025.838_374_534_528,
            -5_693.395_229_188_787,
        ),
    }
}

fn t30_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(
            -4_518_172.624_566_758,
            4_518_172.624_566_755,
            3_689_072.500_000_006,
        ),
        velocity: DVec3::new(
            -5_197.334_993_031_658,
            -5_197.334_993_031_644,
            4.964_758_611_634_907e-12,
        ),
    }
}

fn t40_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(
            5_217_136.362_077_619,
            -5_217_136.362_077_619,
            -3.102_474_751_043_898e-11,
        ),
        velocity: DVec3::new(
            4_025.838_374_534_53,
            4_025.838_374_534_528,
            -5_693.395_229_188_788,
        ),
    }
}

fn t50_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(7_378_145.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7_350.141_635_341_643, 0.0),
    }
}

fn t55_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5_657_077.34, 3_080_301.15, 1_918_782.98),
        velocity: DVec3::new(-3_868.050_682, 3_564.535_653, 5_633.830_366),
    }
}

fn build_t01(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t01_state())
}

fn build_t10(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t10_state())
}

fn build_t20(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t20_state())
}

fn build_t30(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t30_state())
}

fn build_t40(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t40_state())
}

fn build_t50(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t50_state())
}

fn build_t55(_init: &InitialConditions) -> SimulationBuilder {
    build_orbelem_comprehensive(load_mu_earth(), t55_state())
}

/// T01 — equatorial circular (e=0, i=0, sma≈7378 km).
pub fn t01() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t01",
        scenario: build_t01,
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

/// T10 — inclined near-circular (i≈30°, e≈0, sma≈7378 km).
pub fn t10() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t10",
        scenario: build_t10,
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

/// T20 — eccentric inclined (e=0.2, i≈45°, sma≈9223 km).
pub fn t20() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t20",
        scenario: build_t20,
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

/// T30 — orb_elem edge case (JEOD logs sma=0 with a valid 7.2 Mm
/// Cartesian state; exercises a degenerate branch of the
/// orbital-element extraction).
pub fn t30() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t30",
        scenario: build_t30,
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

/// T40 — orb_elem edge case with eccentricity (JEOD logs sma=0,
/// e=0.2, i≈45°).
pub fn t40() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t40",
        scenario: build_t40,
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

/// T50 — equatorial circular, extended log (same initial state as
/// T01; fixture spans t=0..5000s).
pub fn t50() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t50",
        scenario: build_t50,
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

/// T55 — ISS-like LEO (i≈51.7°, e≈0.0025, sma≈6733 km).
pub fn t55() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbelem_comprehensive_t55",
        scenario: build_t55,
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
