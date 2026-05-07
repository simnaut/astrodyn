// The `recipes::verification` module is hidden from rendered rustdoc
// (declared in `recipes/mod.rs`) — the entire submodule is
// workspace-internal Tier 3 scaffolding that downstream mission code
// should not consume. Intra-doc links inside this file therefore
// aren't surfaced anywhere; allow the broken-link lint so we don't
// have to chase resolution that rustdoc suppresses for hidden
// modules.
#![allow(rustdoc::broken_intra_doc_links)]

//! Verification-case scaffolding.
//!
//! [`VerificationCase`] bundles a scenario constructor, a reference
//! CSV path, propagation duration, and per-component tolerances into a
//! single declarative unit. Tier 3 tests in Phase 7/8 collapse to:
//!
//! // reason: `run_and_assert` is defined by `astrodyn_runner::VerificationCaseExt`, which astrodyn cannot depend on without a circular workspace dependency.
//! ```ignore
//! #[test]
//! fn tier3_dyncomp_run2_3dof() {
//!     verification::dyncomp_run2_3dof().run_and_assert();
//! }
//! ```
//!
//! Phase 6 shipped only the scaffold: the [`VerificationCase`] /
//! [`Tolerances`] / [`CsvReference`] types and the
//! [`reference_data`] submodule for JEOD-source-dependent loaders
//! (gravity coefficient files, etc.). Phase 7 expands [`CsvReference`]
//! into a tagged enum that names the per-CSV layout and provides one
//! constructor per Tier 3 case in `sim_dyncomp` (and follow-on
//! family modules).
//!
//! `run_and_assert` itself is implemented by `astrodyn_runner` via an
//! extension trait, since materializing a [`SimulationBuilder`] into a
//! runtime `astrodyn_runner::Simulation` is runner-specific. The runner-side
//! trait also dispatches on the [`CsvReference`] variant, calling the
//! matching loader from `astrodyn_test_data::tier3_csv`.

#[cfg(feature = "jeod-source")]
#[path = "reference_data.rs"]
pub mod reference_data;

/// Stub `reference_data` module used when JEOD-source-backed loaders are
/// unavailable.
///
/// Keeps the public `verification::reference_data` path present for
/// rustdoc and intra-doc links in builds that do not enable the
/// `jeod-source` feature.
#[cfg(not(feature = "jeod-source"))]
pub mod reference_data {}

use glam::{DQuat, DVec3};
use uom::si::f64::Time;

use crate::SimulationBuilder;

/// Adapter-neutral interface for the operations a `pre_step` hook needs.
///
/// Implemented by `astrodyn_runner::Simulation` (and any future ECS adapter
/// that materializes a `VerificationCase`). Lets a `pre_step` closure
/// inject state into the running simulation between reference-CSV time
/// steps without depending on the `astrodyn_runner` crate.
pub trait SimContext {
    /// Set the inertial position of source `source_idx`.
    fn set_source_position(&mut self, source_idx: usize, position: DVec3);
    /// Set the inertial position and velocity of source `source_idx`.
    fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3);
    /// Update the inertial position of one tidal body inside source
    /// `source_idx`'s tidal configuration. Used by tide-validation
    /// hooks that drive Sun/Moon positions for the tidal ΔC20 each
    /// step. Panics if `source_idx` lacks a tidal config or
    /// `tidal_body_idx` is out of range — these are programmer errors,
    /// not runtime conditions, since the recipe wires the tidal
    /// config at construction time.
    ///
    /// The default implementation panics with an explicit
    /// "tidal bodies not supported" message so existing `SimContext`
    /// implementors stay source-compatible. Adapters that wire
    /// tidal-body state into a `Simulation`-equivalent should
    /// override this.
    fn set_tidal_body_position(
        &mut self,
        source_idx: usize,
        tidal_body_idx: usize,
        position: DVec3,
    ) {
        let _ = (source_idx, tidal_body_idx, position);
        panic!("tidal bodies not supported by this SimContext implementation");
    }
}

/// Closure type produced by a [`PreStepBuilder`]. Invoked once per
/// reference-CSV time step, before the simulation propagates.
///
/// The `time` argument is the reference record's time in seconds since
/// the simulation epoch. Closures that need a TDB Julian date should
/// derive it as `j2000_jd + time / 86_400.0` (assuming a J2000 epoch),
/// or capture the epoch's JD when they're constructed by their
/// [`PreStepBuilder`].
pub type PreStepClosure = Box<dyn FnMut(&mut dyn SimContext, f64) + Send>;

/// Factory for a [`PreStepClosure`]. Invoked once at the start of
/// `run_and_assert` with the t=0 [`InitialConditions`], so the closure
/// it returns can capture state (a loaded ephemeris, J2000 JD, source
/// indices, …) that the per-step body would otherwise re-derive on every
/// call.
pub type PreStepBuilder = fn(&InitialConditions) -> PreStepClosure;

/// Initial conditions extracted from the t=0 row of a reference CSV and
/// passed to a scenario constructor by `run_and_assert`. This lets the
/// runner parse each reference CSV exactly once: it loads the full
/// trajectory, hands the t=0 record here to build the scenario, and
/// reuses the rest of the trajectory for the per-step comparison.
///
/// All variants use raw `glam` types so this struct stays adapter-
/// neutral (no dependency on `astrodyn_test_data` from `astrodyn` outside
/// of dev-deps).
///
/// **Quaternion convention.** `glam::DQuat` is laid out as `(x, y, z, w)`
/// where `w` is the scalar component. JEOD's convention is scalar-first
/// `[q0, q1, q2, q3]` where `q0` is the scalar. A JEOD quaternion
/// `[q0, q1, q2, q3]` therefore maps to
/// `DQuat { x: q1, y: q2, z: q3, w: q0 }`. Scenarios that need a
/// [`crate::JeodQuat`] convert via `JeodQuat::from_glam`.
#[derive(Clone, Debug, Default)]
pub struct InitialConditions {
    /// Reference time (seconds since the sim epoch). Always populated.
    pub time: f64,
    /// RootInertial position. Always populated for the variants used by
    /// migrated Tier 3 cases.
    pub position: DVec3,
    /// RootInertial velocity. Always populated for the variants used by
    /// migrated Tier 3 cases.
    pub velocity: DVec3,
    /// Body-frame attitude quaternion in `glam::DQuat` layout
    /// `(x, y, z, w)` where `w` is the scalar. JEOD's scalar-first
    /// convention `[q0, q1, q2, q3]` (with `q0` scalar) maps to
    /// `DQuat { x: q1, y: q2, z: q3, w: q0 }`. `Some` for 6-DOF cases,
    /// `None` for 3-DOF (point-mass translational-only) scenarios.
    pub quaternion: Option<DQuat>,
    /// Body-frame angular velocity. `Some` for 6-DOF cases, `None` for
    /// 3-DOF.
    pub ang_vel: Option<DVec3>,
}

/// A reference-CSV file used by a Tier 3 verification case.
///
/// Each variant tags a distinct column layout produced by one of JEOD's
/// `log_state_ASCII` configurations. The wrapped `&'static str` is the
/// file name relative to the workspace `test_data/` directory. The
/// runner-side `run_and_assert` machinery dispatches on the variant to
/// pick the right loader.
#[derive(Clone, Debug)]
pub enum CsvReference {
    /// 80-column SIM_dyncomp state CSV consumed as a 3-DOF reference:
    /// position/velocity only — quaternion and ang_vel columns are
    /// dropped at the `astrodyn_test_data::crossval::StateLog` layer. Use
    /// this for scenarios that build a `astrodyn::VehicleConfig` without `rot`,
    /// so per-step compares don't synthesize spurious rotational
    /// reference values from CSV columns the simulation never produces.
    Dyncomp3Dof(&'static str),
    /// 80-column SIM_dyncomp state CSV consumed as a 6-DOF reference:
    /// position/velocity *plus* `composite_body.quaternion` and
    /// `composite_body.ang_vel` are populated on the reference
    /// `astrodyn_test_data::crossval::StateLog`.
    Dyncomp6Dof(&'static str),
    /// 21+-column SIM_OrbElem CSV (classical elements + state).
    Orbelem(&'static str),
    /// 17+-column SIM_LVLH CSV (T_parent_this + ang_vel_mag + state).
    Lvlh(&'static str),
    /// 16+-column SIM_NED CSV (geodetic + spherical altitudes/lat/lon
    /// + state).
    Ned(&'static str),
    /// 7-column SIM_3_ORBIT SRP CSV (time + pos + vel).
    Srp(&'static str),
    /// 9-column SIM_1_BASIC SRP CSV (force, torque, flux, temperature).
    SrpBasic(&'static str),
    /// 11-column SIM_VER_DRAG CSV (aero force/torque + inertial vel +
    /// accel mag).
    Drag(&'static str),
    /// 56-column SIM_Euler CSV (36 angles + state + T + quat).
    Euler(&'static str),
    /// 8-column SIM_SolarBeta CSV (time + beta + interleaved pos/vel).
    SolarBeta(&'static str),
    /// 11-column SIM_2A_SHADOW_CALC CSV.
    Shadow(&'static str),
    /// 26-column SIM_torque_compare_simple CSV.
    TorqueSimple(&'static str),
    /// 9-column atmosphere-trajectory CSV (state + density + temp).
    AtmosTraj(&'static str),
    /// 14-column aero-trajectory CSV (state + aero force/torque + density).
    AeroTraj(&'static str),
    /// 7-column trajectory CSV with schema `time + pos[3] + vel[3]`.
    /// Used by any sim whose `log_state_ASCII` config emits exactly the
    /// composite-body inertial state (no rotation matrix, quaternion, or
    /// angular velocity columns). Originating sims include `SIM_orbinit`,
    /// `SIM_GJ_test`, and `SIM_Planetary` — the variant is generic over
    /// the schema, not specific to any one of them.
    OrbInit(&'static str),
    /// 8-column SIM_tide_verif CSV (time + pos + vel + dC20).
    Tide(&'static str),
}

impl CsvReference {
    /// Returns the underlying file name (relative to `test_data/`).
    pub fn file_name(&self) -> &'static str {
        match self {
            CsvReference::Dyncomp3Dof(s)
            | CsvReference::Dyncomp6Dof(s)
            | CsvReference::Orbelem(s)
            | CsvReference::Lvlh(s)
            | CsvReference::Ned(s)
            | CsvReference::Srp(s)
            | CsvReference::SrpBasic(s)
            | CsvReference::Drag(s)
            | CsvReference::Euler(s)
            | CsvReference::SolarBeta(s)
            | CsvReference::Shadow(s)
            | CsvReference::TorqueSimple(s)
            | CsvReference::AtmosTraj(s)
            | CsvReference::AeroTraj(s)
            | CsvReference::OrbInit(s)
            | CsvReference::Tide(s) => s,
        }
    }
}

/// Per-component tolerances for trajectory cross-validation.
///
/// Each field corresponds to a `CrossvalReport::assert_*` method —
/// `position_m` per axis, `velocity_m_s` per axis, scalar
/// `quat_angle_rad`, `ang_vel_rad_s` per axis. `extras` lets a Tier 3
/// case attach scenario-specific tolerances (e.g., the GR perihelion-
/// advance arc-second-per-century check on the Mercury case).
///
/// **Skip semantics.** A whole metric group is skipped only when *all*
/// of its component tolerances are zero (`position_m: [0.0; 3]`,
/// `velocity_m_s: [0.0; 3]`, scalar `quat_angle_rad == 0.0`,
/// `ang_vel_rad_s: [0.0; 3]`). This is the pattern used by 3-DOF cases
/// to opt out of rotational assertions. A non-zero entry alongside a
/// zero entry in the same array does *not* skip the zero axis — the
/// runner still asserts `error_axis < 0.0` on it, which always fails.
/// Mixing zero and non-zero entries within a single array is therefore
/// almost always a configuration mistake.
#[derive(Clone, Debug)]
pub struct Tolerances {
    /// Per-axis position tolerance (m). All-zero opts out of the
    /// position assertion entirely.
    pub position_m: [f64; 3],
    /// Per-axis velocity tolerance (m/s). All-zero opts out of the
    /// velocity assertion entirely.
    pub velocity_m_s: [f64; 3],
    /// Scalar quaternion-angle tolerance (rad). Zero opts out of the
    /// attitude assertion entirely.
    pub quat_angle_rad: f64,
    /// Per-axis angular-velocity tolerance (rad/s). All-zero opts out of
    /// the angular-velocity assertion entirely.
    pub ang_vel_rad_s: [f64; 3],
    /// Family-specific extras: `(name, abs-tolerance)` pairs that the
    /// runner asserts against `report.add_extra(name, ...)` outputs.
    pub extras: &'static [(&'static str, f64)],
}

/// Per-family extras comparator dispatched by `run_and_assert`.
///
/// Each variant tags a family-specific extractor that pairs a
/// [`crate::recipes::verification::CsvReference`]'s typed record at
/// step *k* with the runner-side `astrodyn_runner::VehicleOutput` at the
/// same step, yielding `(name, abs_error)` pairs the runner
/// accumulates as max errors and asserts against
/// [`Tolerances::extras`].
///
/// The runner-side dispatch lives in `astrodyn_runner::run_verification`
/// (it has access to typed records and `VehicleOutput`); this enum is
/// adapter-neutral so `VerificationCase` itself stays in `astrodyn`.
#[derive(Clone, Debug)]
pub enum ExtrasComparator {
    /// Classical orbital elements: 7 extras (sma, eccentricity, inclination,
    /// arg_periapsis, long_asc_node, true_anom, mean_anom).
    Orbelem,
    /// LVLH frame: 2 extras (`t_parent_this` matrix-element max error,
    /// `ang_vel` magnitude error).
    Lvlh,
    /// Geodetic state: 3 extras (`altitude`, `latitude`, `longitude`).
    /// `spherical=true` compares against the spherical-Earth columns;
    /// `false` (default) compares against ellipsoidal columns.
    Ned {
        /// `true` compares against the spherical-Earth NED columns;
        /// `false` (default) compares against the ellipsoidal columns.
        spherical: bool,
    },
    /// Euler angles: 3 extras (`euler_roll`, `euler_pitch`, `euler_yaw`)
    /// computed against JEOD's logged quaternion via our own port of the
    /// Euler-from-matrix conversion (self-consistency check of our Euler
    /// extractor against the JEOD-quaternion reference).
    Euler,
    /// Same Euler self-consistency check as [`Self::Euler`] but reading
    /// the reference quaternion from a [`CsvReference::Dyncomp6Dof`]
    /// `composite_body.quaternion` row rather than a SIM_Euler CSV. Used
    /// by SIM_Euler runs that drive themselves from the SIM_dyncomp
    /// RUN_2 trajectory.
    DyncompEuler,
    /// Solar beta angle: 1 extra (`beta`) comparing `body.solar_beta`
    /// against the matching column in JEOD's SIM_SolarBeta reference
    /// CSV. Pairs with [`CsvReference::SolarBeta`]. Solar beta in this
    /// codebase is constrained to `[-π/2, π/2]` per
    /// `astrodyn_math::solar_beta_angle_*`, so the metric is a plain
    /// absolute difference (no angular wrap-around to handle).
    SolarBeta,
    /// Solid-body tidal ΔC20: 1 extra (`dc20`) comparing the
    /// simulation's per-step ΔC20 (sourced from
    /// `Simulation::source_delta_c20(earth_source_idx)`) against the
    /// `dC20` column logged by JEOD's SIM_tide_verif. Pairs with
    /// [`CsvReference::Tide`]. The recipe carries the Earth source
    /// index because dC20 is per-source, not per-body.
    TideDc20 {
        /// Index (in the simulation's source table) of the Earth source
        /// whose ΔC20 series the comparator will sample.
        earth_source_idx: usize,
    },
}

/// A single Tier 3 verification case.
///
/// Phase 6 shipped the type; Phase 7+ populates `verification::*`
/// constructors that produce one of these per existing Tier 3 test.
/// `run_and_assert` is provided by `astrodyn_runner::run_verification`
/// because materializing the scenario into a runtime
/// `astrodyn_runner::Simulation` is runner-specific.
#[derive(Clone, Debug)]
pub struct VerificationCase {
    /// Unique name used for `target/tier3_crossval/{name}.json` reports.
    pub name: &'static str,
    /// Scenario constructor. Receives the t=0 [`InitialConditions`]
    /// extracted from `reference` so the scenario does not need to
    /// re-parse the reference CSV. The fn pointer stays adapter-neutral
    /// so the runner and (Phase 9) Bevy adapter consume the same
    /// scenario.
    pub scenario: fn(&InitialConditions) -> SimulationBuilder,
    /// Reference CSV produced by the corresponding JEOD verification
    /// simulation.
    pub reference: CsvReference,
    /// Total propagation duration. The runner truncates iteration over
    /// the reference CSV to records with `record.time <= duration`.
    /// `Time::new::<second>(0.0)` (or any value `<= 0.0`) means *use the
    /// full CSV*. If `duration` exceeds the last record's time the loop
    /// simply runs to the end (no extrapolation).
    pub duration: Time,
    /// Per-component tolerances for the cross-validation report.
    pub tolerances: Tolerances,
    /// Optional per-family extras comparator. When `Some`, the runner
    /// computes the family's `(name, error)` pairs alongside the state
    /// log and asserts each against the matching entry in
    /// [`Tolerances::extras`].
    pub extras: Option<ExtrasComparator>,
    /// Optional pre-step hook factory. When `Some`, the runner calls the
    /// factory once at the start of `run_and_assert` (with the t=0
    /// [`InitialConditions`]) to obtain a [`PreStepClosure`], then
    /// invokes that closure before each `sim.step_until(record.time)`
    /// call. Use this to inject per-step state — most commonly source
    /// ephemeris updates — into the running simulation.
    ///
    /// The factory pattern lets the closure capture run-once state (a
    /// loaded DE421 ephemeris, J2000 JD, source indices) that the
    /// per-step body would otherwise re-derive on every call.
    pub pre_step: Option<PreStepBuilder>,
}
