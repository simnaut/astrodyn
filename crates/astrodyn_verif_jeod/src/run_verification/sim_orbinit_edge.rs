//! `VerificationCase` constructors for the SIM_orbinit edge family
//! (`tier3_sim_orbinit_edge`).
//!
//! These cases exercise the `Simulation` pipeline starting from JEOD's
//! SIM_orbinit post-initialization output: each RUN's `composite_body`
//! inertial state (logged at t=0 in `orbinit_0XXX_orbinit.csv`) is fed
//! to a point-mass-Earth `Simulation` and propagated a single step so
//! the integrator and frame-propagation stages run end-to-end. The
//! per-RUN states come from JEOD output (the t=0 row of a JEOD
//! reference CSV is "JEOD source data" under the workspace's
//! computational-independence rule).
//!
//! Cross-consistency between the STS-114 RUNs (0101, 0301, 0401) and
//! the cross-vehicle ISS-vs-STS-114 sanity bound stay in the tier3
//! file — those are properties of the *collection* of recipes, not of
//! any single one, so they can't live inside a per-recipe factory.
//!
//! Each recipe pairs with [`CsvReference::SyntheticTimes`] so the
//! parity wrapper drives a lockstep `runner ↔ bevy` bit-identity
//! assertion at the synthetic checkpoint without needing a
//! multi-record reference CSV (the orbinit CSVs are
//! initialization-only — they log only the t=0 row, which leaves
//! `ref_states.iter().skip(1)` empty for `CsvReference::OrbInit`).
//!
//! Initial state values originate from the JEOD output rows of
//! `crates/astrodyn_verif_jeod/test_data/orbinit_0XXX_orbinit.csv`,
//! committed alongside the rest of the verif fixtures.

use super::fixtures::load_mu_earth;
use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    default_leap_second_table, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, RotationModel, SimulationBuilder, SimulationTime,
    TranslationalState, VehicleConfig,
};
use glam::DVec3;
use uom::si::f64::Time;
use uom::si::time::second;

/// Step size shared by every recipe. Matches the value the pre-recipe
/// tier3 file used so the SyntheticTimes cadence drives identical
/// integration ticks.
const DT_S: f64 = 10.0;

/// One propagation step is enough to exercise the integrator and
/// frame-propagation stages — the original tier3 file called
/// `sim.step()` exactly once after construction, so a single
/// synthetic-cadence checkpoint preserves the test's coverage shape.
const NUM_STEPS: usize = 1;

/// Shared scenario builder for every recipe. Parameterised by:
///   * `mu_earth` — Earth's gravitational parameter (point-mass);
///   * `body` — the vehicle's initial translational state in the
///     RootInertial frame.
fn build_orbinit_edge(mu_earth: f64, body: TranslationalState) -> SimulationBuilder {
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
        ..VehicleConfig::named("sim-orbinit-edge-0")
    });
    sb
}

/// Recipes opt out of every runner-vs-JEOD tolerance group because they
/// pair with [`CsvReference::SyntheticTimes`]; the tier3 sibling
/// asserts state-range sanity bounds and cross-RUN consistency
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
// `orbinit_0XXX_orbinit.csv` (the JEOD post-initialization output).
// Distinct RUN labels exercise different SIM_orbinit code paths:
//
//   * RUN_0101 — STS-114, orbital elements in the inertial frame;
//   * RUN_0201 — ISS, orbital elements in the planet-fixed frame;
//   * RUN_0301 — STS-114, orbital elements in the planet-fixed frame;
//   * RUN_0401 — STS-114, direct Cartesian state in the inertial
//                frame (3 decimals less precision than the OE runs —
//                the CSV truncation is JEOD's, not ours).
//
// The pair sharing the same vehicle/configuration trio (0101, 0301,
// 0401 for STS-114) should land within ≈1 m of each other; that
// cross-consistency assertion stays in the tier3 sibling.

fn run_0101_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(1_244_471.751921491, 5_655_811.75396678, 3_425_519.023834599),
        velocity: DVec3::new(-6003.553560448576, -1469.322206986292, 4590.577121215543),
    }
}

fn run_0201_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(
            1_244_540.336462717,
            5_655_938.802744529,
            3_425_643.360843232,
        ),
        velocity: DVec3::new(-6003.83315183158, -1469.496289202212, 4590.511665610799),
    }
}

fn run_0301_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(
            1_244_471.742508102,
            5_655_811.758825388,
            3_425_519.021394584,
        ),
        velocity: DVec3::new(-6003.553568336125, -1469.322209110831, 4590.577121539032),
    }
}

fn run_0401_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(1_244_471.94, 5_655_811.8, 3_425_518.88),
        velocity: DVec3::new(-6003.553468, -1469.321965, 4590.57723),
    }
}

fn build_run_0101(_init: &InitialConditions) -> SimulationBuilder {
    build_orbinit_edge(load_mu_earth(), run_0101_state())
}

fn build_run_0201(_init: &InitialConditions) -> SimulationBuilder {
    build_orbinit_edge(load_mu_earth(), run_0201_state())
}

fn build_run_0301(_init: &InitialConditions) -> SimulationBuilder {
    build_orbinit_edge(load_mu_earth(), run_0301_state())
}

fn build_run_0401(_init: &InitialConditions) -> SimulationBuilder {
    build_orbinit_edge(load_mu_earth(), run_0401_state())
}

/// RUN_0101: STS-114 orbital elements in the inertial frame.
pub fn run_0101() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_edge_run_0101",
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

/// RUN_0201: ISS orbital elements in the planet-fixed frame.
pub fn run_0201() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_edge_run_0201",
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

/// RUN_0301: STS-114 orbital elements in the planet-fixed frame.
pub fn run_0301() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_edge_run_0301",
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

/// RUN_0401: STS-114 direct Cartesian state in the inertial frame.
pub fn run_0401() -> VerificationCase {
    VerificationCase {
        name: "tier3_orbinit_edge_run_0401",
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
