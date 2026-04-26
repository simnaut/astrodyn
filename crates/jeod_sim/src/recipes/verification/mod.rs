//! Verification-case scaffolding.
//!
//! [`VerificationCase`] bundles a scenario constructor, a reference
//! CSV path, propagation duration, and per-component tolerances into a
//! single declarative unit. Tier 3 tests in Phase 7/8 collapse to:
//!
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
//! constructor per Tier 3 case in [`sim_dyncomp`] (and follow-on
//! family modules).
//!
//! `run_and_assert` itself is implemented by `jeod_runner` via an
//! extension trait, since materializing a [`SimulationBuilder`] into a
//! runtime [`Simulation`] is runner-specific. The runner-side trait
//! also dispatches on the [`CsvReference`] variant, calling the
//! matching loader from `jeod_test_data::tier3_csv`.

pub mod reference_data;

use uom::si::f64::Time;

use crate::SimulationBuilder;

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
    /// dropped at the [`StateLog`](jeod_test_data::crossval::StateLog)
    /// layer. Use this for scenarios that build a [`VehicleConfig`]
    /// without `rot`, so per-step compares don't synthesize spurious
    /// rotational reference values from CSV columns the simulation
    /// never produces.
    Dyncomp3Dof(&'static str),
    /// 80-column SIM_dyncomp state CSV consumed as a 6-DOF reference:
    /// position/velocity *plus* `composite_body.quaternion` and
    /// `composite_body.ang_vel` are populated on the reference
    /// [`StateLog`](jeod_test_data::crossval::StateLog).
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
    /// 7-column SIM_orbinit / SIM_GJ_test CSV (time + pos + vel).
    OrbInit(&'static str),
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
            | CsvReference::OrbInit(s) => s,
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
/// A tolerance of `0.0` means *skip the assertion for that component*
/// — used for 3-DOF cases that have no rotational state.
#[derive(Clone, Debug)]
pub struct Tolerances {
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    pub quat_angle_rad: f64,
    pub ang_vel_rad_s: [f64; 3],
    pub extras: &'static [(&'static str, f64)],
}

/// Per-family extras comparator dispatched by `run_and_assert`.
///
/// Each variant tags a family-specific extractor that pairs a
/// [`crate::recipes::verification::CsvReference`]'s typed record at
/// step *k* with the [`VehicleOutput`](jeod_runner::VehicleOutput) at
/// the same step, yielding `(name, abs_error)` pairs the runner
/// accumulates as max errors and asserts against
/// [`Tolerances::extras`].
///
/// The runner-side dispatch lives in `jeod_runner::run_verification`
/// (it has access to typed records and `VehicleOutput`); this enum is
/// adapter-neutral so `VerificationCase` itself stays in `jeod_sim`.
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
    Ned { spherical: bool },
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
}

impl Default for Tolerances {
    /// Default tolerances broad enough to spot a regression while not
    /// rejecting bit-stable runs. Concrete cases tighten these.
    fn default() -> Self {
        Self {
            position_m: [1.0, 1.0, 1.0],
            velocity_m_s: [1.0e-3, 1.0e-3, 1.0e-3],
            quat_angle_rad: 1.0e-6,
            ang_vel_rad_s: [1.0e-9, 1.0e-9, 1.0e-9],
            extras: &[],
        }
    }
}

/// A single Tier 3 verification case.
///
/// Phase 6 shipped the type; Phase 7+ populates `verification::*`
/// constructors that produce one of these per existing Tier 3 test.
/// `run_and_assert` is provided by `jeod_runner::run_verification`
/// because materializing the scenario into a runtime [`Simulation`]
/// is runner-specific.
#[derive(Clone, Debug)]
pub struct VerificationCase {
    /// Unique name used for `target/tier3_crossval/{name}.json` reports.
    pub name: &'static str,
    /// Scenario constructor. The fn pointer stays adapter-neutral so
    /// the runner and (Phase 9) Bevy adapter consume the same scenario.
    pub scenario: fn() -> SimulationBuilder,
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
}
