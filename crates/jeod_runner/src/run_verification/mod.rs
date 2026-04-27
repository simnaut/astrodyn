//! Phase 7 of #101 — `run_and_assert` machinery for Tier 3 verification.
//!
//! [`jeod_sim::recipes::verification::VerificationCase`]
//! lives in `jeod_sim` as adapter-neutral data. Materializing the scenario
//! into a runtime [`crate::Simulation`], loading the reference
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

pub mod sim_derived_state;
pub mod sim_dyncomp;
pub mod sim_planetary;
pub mod sim_polar_motion;
pub mod sim_solar_beta;
pub mod sim_srp;
pub mod sim_torque_simple;

use glam::DVec3;
use jeod_sim::recipes::verification::{
    CsvReference, ExtrasComparator, InitialConditions, SimContext, VerificationCase,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::tier3_csv;
use uom::si::time::second;

use crate::builder::SimulationBuilderExt;
use crate::{Simulation, VehicleOutput};

/// Forward `SimContext` to the runtime simulation so `pre_step` closures
/// stored on a `VerificationCase` can drive source ephemeris updates
/// without depending on the `jeod_runner` crate directly.
impl SimContext for Simulation {
    fn set_source_position(&mut self, source_idx: usize, position: DVec3) {
        Simulation::set_source_position(self, source_idx, position);
    }
    fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3) {
        Simulation::set_source_state(self, source_idx, position, velocity);
    }
}

/// Per-family typed records held alongside the [`StateLog`] vec so
/// extras comparators can read columns the generic state log drops
/// (orbital elements, LVLH transforms, geodetic coordinates, …).
enum CsvRecords {
    Dyncomp(Vec<tier3_csv::DyncompRecord>),
    Orbelem(Vec<tier3_csv::OrbElemRecord>),
    Lvlh(Vec<tier3_csv::LvlhRecord>),
    Ned(Vec<tier3_csv::NedRecord>),
    Euler(Vec<tier3_csv::EulerRecord>),
    SolarBeta(Vec<tier3_csv::SolarBetaRecord>),
    /// Variants without family-specific extras keep only the per-step
    /// time so [`StateLog`]-based assertions still align.
    Times(Vec<f64>),
}

impl CsvRecords {
    fn len(&self) -> usize {
        match self {
            Self::Dyncomp(v) => v.len(),
            Self::Orbelem(v) => v.len(),
            Self::Lvlh(v) => v.len(),
            Self::Ned(v) => v.len(),
            Self::Euler(v) => v.len(),
            Self::SolarBeta(v) => v.len(),
            Self::Times(v) => v.len(),
        }
    }
}

/// Extension trait that runs a [`VerificationCase`] end-to-end and
/// asserts its tolerances.
pub trait VerificationCaseExt {
    /// Build the scenario, load the reference CSV, propagate via
    /// [`crate::Simulation::step_until`] up to the case's `duration`, and
    /// assert tolerances on the resulting [`CrossvalReport`]. Panics on
    /// any tolerance breach.
    fn run_and_assert(&self);
}

impl VerificationCaseExt for VerificationCase {
    fn run_and_assert(&self) {
        // 1. Load the reference CSV exactly once — the t=0 row supplies
        //    the scenario's initial conditions, the rest drives the
        //    per-step comparison loop.
        let ref_path = tier3_csv::test_data_path(self.reference.file_name());
        assert!(
            ref_path.exists(),
            "JEOD reference CSV not found at {}.\n\
             Generate with: docker run --rm -v $(pwd)/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            ref_path.display()
        );
        let (ref_states, typed_records) =
            load_reference(&self.reference, &ref_path, self.extras.as_ref());
        assert!(
            !ref_states.is_empty(),
            "{}: reference CSV {} produced 0 records",
            self.name,
            ref_path.display()
        );
        assert_eq!(
            ref_states.len(),
            typed_records.len(),
            "{}: state/typed-record length mismatch ({} vs {})",
            self.name,
            ref_states.len(),
            typed_records.len()
        );

        // 2. Build the scenario from the t=0 record into a runtime
        //    Simulation. The scenario constructor consumes the
        //    `InitialConditions` derived from the StateLog, so it does
        //    not need to re-parse the reference CSV.
        let init = initial_conditions_from(&ref_states[0]);
        let mut sim = (self.scenario)(&init)
            .build()
            .unwrap_or_else(|e| panic!("scenario `{}` failed validation: {e:?}", self.name));

        // 2b. If the case carries a pre-step factory, invoke it now so
        //     the resulting closure can capture run-once state (a
        //     loaded DE421 ephemeris, J2000 JD, source indices) the
        //     per-step body would otherwise re-derive on every call.
        let mut pre_step = self.pre_step.map(|builder| builder(&init));

        // 3. Propagate, sampling at each non-initial reference time up
        //    to the case's `duration` (a value of 0.0 or negative means
        //    "use full CSV"; a value greater than the last record's
        //    time runs to the end without extrapolation).
        let duration_s = self.duration.get::<second>();
        let mut our_states = Vec::with_capacity(ref_states.len() - 1);
        let mut sampled_refs = Vec::with_capacity(ref_states.len() - 1);
        let mut extras_acc = self.extras.as_ref().map(ExtrasAccumulator::new);
        for (idx, record) in ref_states.iter().enumerate().skip(1) {
            if duration_s > 0.0 && record.time > duration_s {
                break;
            }
            // Run the pre-step hook (e.g. ephemeris source-position
            // update) before propagation, so the simulation sees
            // up-to-date inputs for this step.
            if let Some(hook) = pre_step.as_mut() {
                hook(&mut sim, record.time);
            }
            sim.step_until(record.time);
            let body = sim.body(0);
            if let Some(acc) = extras_acc.as_mut() {
                acc.observe(&body, &typed_records, idx, self.name);
            }
            our_states.push(snapshot_from(&body, record));
            sampled_refs.push(record.clone());
        }

        // 4. Compute the cross-validation report.
        let mut report = CrossvalReport::compute(self.name, &our_states, &sampled_refs);
        if let Some(acc) = extras_acc.as_ref() {
            for (name, val, unit) in acc.entries() {
                report.add_extra(name, *val, unit);
            }
        }
        report.write();

        // 5. Assert tolerances. A whole metric group is skipped only
        //    when every component tolerance in the group is zero — used
        //    by 3-DOF cases to opt out of rotational assertions. A
        //    non-zero entry alongside zeros in the same array still
        //    asserts every axis (the zeros require exact match).
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
        if let Some(acc) = extras_acc.as_ref() {
            acc.assert_against(tols.extras, self.name);
        } else {
            assert!(
                tols.extras.is_empty(),
                "{}: tolerances declare extras {:?} but case has no ExtrasComparator",
                self.name,
                tols.extras
            );
        }
    }
}

/// Project the t=0 [`StateLog`] from the reference CSV into the
/// adapter-neutral [`InitialConditions`] passed to scenario builders.
/// Scenarios consume this rather than re-parsing the CSV themselves.
fn initial_conditions_from(t0: &StateLog) -> InitialConditions {
    InitialConditions {
        time: t0.time,
        position: t0.position.unwrap_or_default(),
        velocity: t0.velocity.unwrap_or_default(),
        quaternion: t0.quaternion,
        ang_vel: t0.ang_vel,
    }
}

/// Build a [`StateLog`] from a body's snapshot, with the time copied
/// from the reference record so [`CrossvalReport::compute`]'s alignment
/// check passes.
fn snapshot_from(body: &VehicleOutput, ref_record: &StateLog) -> StateLog {
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

/// Dispatch on the [`CsvReference`] variant and produce both a
/// `Vec<StateLog>` for the cross-validation report and (when extras
/// are configured) a [`CsvRecords`] holding typed records the extras
/// comparator can read.
///
/// Variants whose CSV has columns the generic [`StateLog`] doesn't
/// model (LVLH transform, geodetic coords, Euler angles, …) drop the
/// extra columns at the [`StateLog`] layer; family-specific extras
/// comparators read them from `typed_records` instead.
fn load_reference(
    csv: &CsvReference,
    path: &std::path::Path,
    extras: Option<&ExtrasComparator>,
) -> (Vec<StateLog>, CsvRecords) {
    use tier3_csv::*;
    // The extras dispatch determines which family-specific record vec
    // we hand back. For variants without an extras comparator we keep
    // a parallel `Times` vec so length-checks line up cheaply.
    match csv {
        CsvReference::Dyncomp3Dof(_) => {
            // 3-DOF: scenario builds a body without `rot`, so our
            // snapshot's quaternion/ang_vel are `None`. Build the
            // reference StateLog with `None` too — feeding the CSV's
            // logged quaternion/ang_vel here would compare apples to
            // oranges (and `CrossvalReport` per-step compare would mix
            // `Some` against `None`).
            let records = load_dyncomp_csv(path);
            let states: Vec<StateLog> = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.composite_body.position),
                    velocity: Some(r.composite_body.velocity),
                    ..Default::default()
                })
                .collect();
            (states, CsvRecords::Dyncomp(records))
        }
        CsvReference::Dyncomp6Dof(_) => {
            // 6-DOF: scenario builds a body with `rot`, so populate
            // quaternion + ang_vel on the reference StateLog as well.
            let records = load_dyncomp_csv(path);
            let states: Vec<StateLog> = records.iter().map(dyncomp_to_state_log_6dof).collect();
            (states, CsvRecords::Dyncomp(records))
        }
        CsvReference::Orbelem(_) => {
            let records = load_orbelem_csv(path);
            let states: Vec<StateLog> = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect();
            let typed = if matches!(extras, Some(ExtrasComparator::Orbelem)) {
                CsvRecords::Orbelem(records)
            } else {
                CsvRecords::Times(states.iter().map(|s| s.time).collect())
            };
            (states, typed)
        }
        CsvReference::Lvlh(_) => {
            let records = load_lvlh_csv(path);
            let states: Vec<StateLog> = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect();
            let typed = if matches!(extras, Some(ExtrasComparator::Lvlh)) {
                CsvRecords::Lvlh(records)
            } else {
                CsvRecords::Times(states.iter().map(|s| s.time).collect())
            };
            (states, typed)
        }
        CsvReference::Ned(_) => {
            let records = load_ned_csv(path);
            let states: Vec<StateLog> = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect();
            let typed = if matches!(extras, Some(ExtrasComparator::Ned { .. })) {
                CsvRecords::Ned(records)
            } else {
                CsvRecords::Times(states.iter().map(|s| s.time).collect())
            };
            (states, typed)
        }
        CsvReference::Euler(_) => {
            let records = load_euler_csv(path);
            let states: Vec<StateLog> = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    quaternion: Some(glam::DQuat::from_xyzw(
                        r.quaternion[1],
                        r.quaternion[2],
                        r.quaternion[3],
                        r.quaternion[0],
                    )),
                    ..Default::default()
                })
                .collect();
            let typed = if matches!(extras, Some(ExtrasComparator::Euler)) {
                CsvRecords::Euler(records)
            } else {
                CsvRecords::Times(states.iter().map(|s| s.time).collect())
            };
            (states, typed)
        }
        CsvReference::Srp(_) => {
            let records = load_srp_trajectory(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::SrpBasic(_) => {
            let records = load_srp_basic_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::Drag(_) => {
            let records = load_drag_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::SolarBeta(_) => {
            let records = load_solar_beta_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            (states, CsvRecords::SolarBeta(records))
        }
        CsvReference::Shadow(_) => {
            let records = load_shadow_calc_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::TorqueSimple(_) => {
            let records = load_torque_simple_csv(path);
            let states = records
                .iter()
                .map(|r| {
                    // CSV stores JEOD scalar-first `[q0, q1, q2, q3]`; glam
                    // expects xyzw, so reorder to `[q1, q2, q3, q0]`.
                    let q = glam::DQuat::from_xyzw(
                        r.quaternion[1],
                        r.quaternion[2],
                        r.quaternion[3],
                        r.quaternion[0],
                    );
                    StateLog {
                        time: r.time,
                        position: Some(r.position),
                        velocity: Some(r.velocity),
                        ang_vel: Some(r.ang_vel),
                        quaternion: Some(q),
                        ..Default::default()
                    }
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::AtmosTraj(_) => {
            let records = load_atmos_traj_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::AeroTraj(_) => {
            let records = load_aero_traj_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
        CsvReference::OrbInit(_) => {
            let records = load_orbinit_csv(path);
            let states = records
                .iter()
                .map(|r| StateLog {
                    time: r.time,
                    position: Some(r.position),
                    velocity: Some(r.velocity),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let times = states.iter().map(|s| s.time).collect();
            (states, CsvRecords::Times(times))
        }
    }
}

/// Per-family extras max-error accumulator used by `run_and_assert`.
///
/// Holds an entry per metric the comparator emits; `observe` updates
/// each metric's running max with the absolute error at the current
/// step. After the propagation loop, [`ExtrasAccumulator::entries`]
/// hands the report the final pairs and [`ExtrasAccumulator::assert_against`]
/// checks each against the matching tolerance row.
struct ExtrasAccumulator {
    kind: ExtrasComparator,
    entries: Vec<(&'static str, f64, &'static str)>,
}

impl ExtrasAccumulator {
    fn new(kind: &ExtrasComparator) -> Self {
        let entries: Vec<(&'static str, f64, &'static str)> = match kind {
            ExtrasComparator::Orbelem => vec![
                ("sma", 0.0, "m"),
                ("eccentricity", 0.0, ""),
                ("inclination", 0.0, "rad"),
                ("arg_periapsis", 0.0, "rad"),
                ("long_asc_node", 0.0, "rad"),
                ("true_anom", 0.0, "rad"),
                ("mean_anom", 0.0, "rad"),
            ],
            ExtrasComparator::Lvlh => {
                vec![("t_parent_this", 0.0, ""), ("ang_vel", 0.0, "rad/s")]
            }
            ExtrasComparator::Ned { .. } => vec![
                ("altitude", 0.0, "m"),
                ("latitude", 0.0, "rad"),
                ("longitude", 0.0, "rad"),
            ],
            ExtrasComparator::Euler | ExtrasComparator::DyncompEuler => vec![
                ("euler_roll", 0.0, "rad"),
                ("euler_pitch", 0.0, "rad"),
                ("euler_yaw", 0.0, "rad"),
            ],
            ExtrasComparator::SolarBeta => vec![("beta", 0.0, "rad")],
        };
        Self {
            kind: kind.clone(),
            entries,
        }
    }

    fn entries(&self) -> &[(&'static str, f64, &'static str)] {
        &self.entries
    }

    fn update_max(&mut self, name: &'static str, val: f64) {
        for entry in self.entries.iter_mut() {
            if entry.0 == name {
                entry.1 = entry.1.max(val);
                return;
            }
        }
        panic!("ExtrasAccumulator: unknown metric `{name}`");
    }

    fn observe(&mut self, body: &VehicleOutput, records: &CsvRecords, idx: usize, case_name: &str) {
        match (&self.kind, records) {
            (ExtrasComparator::Orbelem, CsvRecords::Orbelem(recs)) => {
                let r = &recs[idx];
                let oe = body.orbital_elements.as_ref().unwrap_or_else(|| {
                    panic!("{case_name}: orbital_elements not computed at idx {idx}")
                });
                self.update_max("sma", (oe.semi_major_axis - r.semi_major_axis).abs());
                self.update_max("eccentricity", (oe.e_mag - r.e_mag).abs());
                self.update_max("inclination", (oe.inclination - r.inclination).abs());
                self.update_max(
                    "arg_periapsis",
                    angle_diff(oe.arg_periapsis, r.arg_periapsis),
                );
                self.update_max(
                    "long_asc_node",
                    angle_diff(oe.long_asc_node, r.long_asc_node),
                );
                self.update_max("true_anom", angle_diff(oe.true_anom, r.true_anom));
                self.update_max("mean_anom", angle_diff(oe.mean_anom, r.mean_anom));
            }
            (ExtrasComparator::Lvlh, CsvRecords::Lvlh(recs)) => {
                let r = &recs[idx];
                let lvlh = body
                    .lvlh_frame
                    .as_ref()
                    .unwrap_or_else(|| panic!("{case_name}: lvlh_frame not computed at idx {idx}"));
                self.update_max(
                    "t_parent_this",
                    max_mat_diff(&lvlh.t_parent_this, &r.t_parent_this),
                );
                self.update_max(
                    "ang_vel",
                    (lvlh.ang_vel_this.length() - r.ang_vel_mag).abs(),
                );
            }
            (ExtrasComparator::Ned { spherical }, CsvRecords::Ned(recs)) => {
                let r = &recs[idx];
                let geo = body.geodetic_state.as_ref().unwrap_or_else(|| {
                    panic!("{case_name}: geodetic_state not computed at idx {idx}")
                });
                let (ref_alt, ref_lat, ref_lon) = if *spherical {
                    (r.sphere_altitude, r.sphere_latitude, r.sphere_longitude)
                } else {
                    (r.ellip_altitude, r.ellip_latitude, r.ellip_longitude)
                };
                self.update_max("altitude", (geo.altitude - ref_alt).abs());
                self.update_max("latitude", (geo.latitude - ref_lat).abs());
                self.update_max("longitude", angle_diff(geo.longitude, ref_lon));
            }
            (ExtrasComparator::Euler, CsvRecords::Euler(recs)) => {
                let r = &recs[idx];
                let jeod_q = jeod_sim::JeodQuat::new(
                    r.quaternion[0],
                    r.quaternion[1],
                    r.quaternion[2],
                    r.quaternion[3],
                );
                self.observe_euler_from_quat(body, &jeod_q, idx, case_name);
            }
            (ExtrasComparator::DyncompEuler, CsvRecords::Dyncomp(recs)) => {
                let r = &recs[idx];
                // composite_body.quaternion is glam-DQuat (xyzw) here.
                let q = r.composite_body.quaternion;
                let jeod_q = jeod_sim::JeodQuat::new(q.w, q.x, q.y, q.z);
                self.observe_euler_from_quat(body, &jeod_q, idx, case_name);
            }
            (ExtrasComparator::SolarBeta, CsvRecords::SolarBeta(recs)) => {
                let r = &recs[idx];
                let beta = body
                    .solar_beta
                    .unwrap_or_else(|| panic!("{case_name}: solar_beta not computed at idx {idx}"));
                // Solar beta is constrained to [-π/2, π/2] in JEOD, so
                // wrap-around isn't a real concern — plain absolute
                // difference matches what the bespoke test asserted on.
                self.update_max("beta", (beta - r.solar_beta).abs());
            }
            (kind, recs) => panic!(
                "{case_name}: ExtrasComparator {kind:?} requires the matching \
                 CsvReference variant; got records discriminant len={}",
                recs.len()
            ),
        }
    }

    /// Shared Euler-from-quaternion self-consistency observer used by
    /// both [`ExtrasComparator::Euler`] (SIM_Euler CSVs) and
    /// [`ExtrasComparator::DyncompEuler`] (SIM_dyncomp `composite_body`).
    /// Reconstructs JEOD's reference Euler triple from the logged
    /// quaternion via our own port — this is a self-consistency check
    /// of our Euler extractor against the JEOD-quaternion reference,
    /// not a comparison against JEOD's logged Euler columns (JEOD logs
    /// all 6 × 2 × 3 sequences but our DerivedState only computes one).
    fn observe_euler_from_quat(
        &mut self,
        body: &VehicleOutput,
        jeod_q: &jeod_sim::JeodQuat,
        idx: usize,
        case_name: &str,
    ) {
        use uom::si::angle::radian;
        let euler = body
            .euler_angles
            .unwrap_or_else(|| panic!("{case_name}: euler_angles not computed at idx {idx}"));
        let jeod_t = jeod_q.left_quat_to_transformation();
        let jeod_euler = jeod_math::compute_euler_angles_from_matrix_typed(
            &jeod_t,
            jeod_sim::EulerSequence::XYZ,
        );
        self.update_max(
            "euler_roll",
            angle_diff(euler[0], jeod_euler[0].get::<radian>()),
        );
        self.update_max(
            "euler_pitch",
            angle_diff(euler[1], jeod_euler[1].get::<radian>()),
        );
        self.update_max(
            "euler_yaw",
            angle_diff(euler[2], jeod_euler[2].get::<radian>()),
        );
    }

    fn assert_against(&self, tol: &[(&'static str, f64)], case_name: &str) {
        for (name, observed, _unit) in &self.entries {
            let entry = tol.iter().find(|(n, _)| *n == *name).unwrap_or_else(|| {
                panic!(
                    "{case_name}: tolerances missing entry for extra `{name}` (observed {observed:.3e})"
                )
            });
            assert!(
                *observed < entry.1,
                "{case_name}: extras `{name}` error {:.6e} exceeds tolerance {:.6e}",
                observed,
                entry.1
            );
        }
        // Surface any tolerance entries the comparator doesn't emit so
        // tests don't silently miss new metrics added to the catalog.
        for (tol_name, _) in tol {
            if !self.entries.iter().any(|(n, _, _)| n == tol_name) {
                panic!(
                    "{case_name}: tolerances declare extra `{tol_name}` but \
                     comparator does not emit a metric with that name"
                );
            }
        }
    }
}

/// Compute angular difference accounting for wraparound at 2π.
fn angle_diff(a: f64, b: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = (a - b) % tau;
    if d > std::f64::consts::PI {
        d -= tau;
    }
    if d < -std::f64::consts::PI {
        d += tau;
    }
    d.abs()
}

/// Max absolute element-wise difference between two 3×3 matrices.
fn max_mat_diff(a: &glam::DMat3, b: &glam::DMat3) -> f64 {
    let mut max_d = 0.0_f64;
    for c in 0..3 {
        for r in 0..3 {
            let d = (a.col(c)[r] - b.col(c)[r]).abs();
            max_d = max_d.max(d);
        }
    }
    max_d
}
