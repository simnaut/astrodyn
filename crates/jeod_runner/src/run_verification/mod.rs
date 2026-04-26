//! Phase 7 of #101 — `run_and_assert` machinery for Tier 3 verification.
//!
//! [`VerificationCase`](jeod_sim::recipes::verification::VerificationCase)
//! lives in `jeod_sim` as adapter-neutral data. Materializing the scenario
//! into a runtime [`Simulation`](crate::Simulation), loading the reference
//! CSV, propagating, and asserting tolerances is runner-specific — it
//! lives here as the [`VerificationCaseExt`] trait.
//!
//! The trait dispatches on the [`CsvReference`] variant to pick the right
//! loader from `jeod_test_data::tier3_csv`, then delegates assertion to
//! [`CrossvalReport`]. Tolerances of `0.0` mean *skip the assertion for
//! that component* — used for 3-DOF cases that have no rotational state.
//!
//! Per-family scenario constructors live in this module's submodules
//! ([`sim_dyncomp`], etc.). Each test in `jeod_runner/tests/tier3_*.rs`
//! collapses to a one-liner of the form:
//!
//! ```ignore
//! use jeod_runner::prelude::*;
//! use jeod_runner::run_verification::sim_dyncomp;
//!
//! #[test]
//! fn tier3_sim_dyncomp_run2_3dof() {
//!     sim_dyncomp::run2_3dof().run_and_assert();
//! }
//! ```

pub mod sim_dyncomp;
pub mod sim_planetary;
pub mod sim_polar_motion;

use jeod_sim::recipes::verification::{CsvReference, VerificationCase};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::tier3_csv;

use crate::builder::SimulationBuilderExt;
use crate::{Simulation, VehicleOutput};

/// Extension trait that runs a [`VerificationCase`] end-to-end and
/// asserts its tolerances.
pub trait VerificationCaseExt {
    /// Build the scenario, load the reference CSV, propagate via
    /// [`Simulation::step_until`] up to the case's `duration`, and
    /// assert tolerances on the resulting [`CrossvalReport`]. Panics on
    /// any tolerance breach.
    fn run_and_assert(&self);
}

impl VerificationCaseExt for VerificationCase {
    fn run_and_assert(&self) {
        // 1. Build the scenario into a runtime Simulation.
        let mut sim = (self.scenario)()
            .build()
            .unwrap_or_else(|e| panic!("scenario `{}` failed validation: {e:?}", self.name));

        // 2. Load the reference CSV into time-aligned StateLogs.
        let ref_path = tier3_csv::test_data_path(self.reference.file_name());
        assert!(
            ref_path.exists(),
            "JEOD reference CSV not found at {}.\n\
             Generate with: docker run --rm -v $(pwd)/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            ref_path.display()
        );
        let ref_states = load_reference_states(&self.reference, &ref_path);
        assert!(
            !ref_states.is_empty(),
            "{}: reference CSV {} produced 0 records",
            self.name,
            ref_path.display()
        );

        // 3. Propagate, sampling at each non-initial reference time.
        let mut our_states = Vec::with_capacity(ref_states.len() - 1);
        let mut sampled_refs = Vec::with_capacity(ref_states.len() - 1);
        for record in ref_states.iter().skip(1) {
            sim.step_until(record.time);
            our_states.push(snapshot(&sim, 0, record));
            sampled_refs.push(record.clone());
        }

        // 4. Compute the cross-validation report.
        let report = CrossvalReport::compute(self.name, &our_states, &sampled_refs);
        report.write();

        // 5. Assert tolerances. A 0.0 component means "skip this axis"
        //    (e.g., 3-DOF cases have no rotational state).
        let tols = &self.tolerances;
        if tols.position_m.iter().any(|t| *t > 0.0) {
            report.assert_position(tols.position_m);
        }
        if tols.velocity_m_s.iter().any(|t| *t > 0.0) {
            report.assert_velocity(tols.velocity_m_s);
        }
        if tols.quat_angle_rad > 0.0 {
            report.assert_quat_angle(tols.quat_angle_rad);
        }
        if tols.ang_vel_rad_s.iter().any(|t| *t > 0.0) {
            report.assert_ang_vel(tols.ang_vel_rad_s);
        }
        // `extras` are not handled here — Phase 7 covers archetype A
        // (translational + optional rotational). Cases with derived-state
        // extras (orbelem, lvlh, ned, euler, …) will be migrated when the
        // matching loader-and-extras dispatch lands.
        assert!(
            tols.extras.is_empty(),
            "{}: extras are not yet supported by run_and_assert ({:?})",
            self.name,
            tols.extras
        );
    }
}

/// Build a [`StateLog`] from the simulation's current body output, with
/// the time copied from the matching reference record so
/// [`CrossvalReport::compute`]'s alignment check passes.
fn snapshot(sim: &Simulation, body_idx: usize, ref_record: &StateLog) -> StateLog {
    let body: VehicleOutput = sim.body(body_idx);
    StateLog {
        time: ref_record.time,
        position: Some(body.trans.position),
        velocity: Some(body.trans.velocity),
        acceleration: None,
        quaternion: body.rot.as_ref().map(|r| r.quaternion.to_glam()),
        ang_vel: body.rot.as_ref().map(|r| r.ang_vel_body),
        ang_accel: None,
    }
}

/// Dispatch on the [`CsvReference`] variant and produce a `Vec<StateLog>`
/// suitable for [`CrossvalReport::compute`]. Variants whose CSV has more
/// than position/velocity (LVLH transform, geodetic coords, Euler
/// angles, …) drop the extra columns at this layer; comparing those
/// fields is the responsibility of family-specific runners that Phase
/// 7+ wires up incrementally.
fn load_reference_states(csv: &CsvReference, path: &std::path::Path) -> Vec<StateLog> {
    use tier3_csv::*;
    match csv {
        CsvReference::Dyncomp(_) => {
            let records = load_dyncomp_csv(path);
            // 6-DOF tests want quaternion + ang_vel; 3-DOF tests only
            // compare position/velocity. We always populate the 6-DOF
            // shape — `assert_quat_angle` / `assert_ang_vel` are gated
            // by tolerances above.
            records.iter().map(dyncomp_to_state_log_6dof).collect()
        }
        CsvReference::Orbelem(_) => load_orbelem_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::Lvlh(_) => load_lvlh_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::Ned(_) => load_ned_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::Srp(_) => load_srp_trajectory(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::SrpBasic(_) => load_srp_basic_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                ..Default::default()
            })
            .collect(),
        CsvReference::Drag(_) => load_drag_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                ..Default::default()
            })
            .collect(),
        CsvReference::Euler(_) => load_euler_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::SolarBeta(_) => load_solar_beta_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::Shadow(_) => load_shadow_calc_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                ..Default::default()
            })
            .collect(),
        CsvReference::TorqueSimple(_) => load_torque_simple_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ang_vel: Some(r.ang_vel),
                ..Default::default()
            })
            .collect(),
        CsvReference::AtmosTraj(_) => load_atmos_traj_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::AeroTraj(_) => load_aero_traj_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
        CsvReference::OrbInit(_) => load_orbinit_csv(path)
            .iter()
            .map(|r| StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                ..Default::default()
            })
            .collect(),
    }
}
