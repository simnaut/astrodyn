//! Parity tests between `astrodyn_runner` (arena harness) and `astrodyn_bevy`
//! (ECS adapter).
//!
//! Each integration test in `tests/` runs an identical scenario through
//! both consumers of the `astrodyn` gateway crate and asserts bit-identical
//! state. The library half (this file) provides the shared
//! [`VerificationCaseParityExt`] trait that turns any Tier 3
//! [`astrodyn_verif_jeod::verification::VerificationCase`]
//! into a one-line parity assertion (issue #389), so wrapper tests collapse
//! to:
//!
//! ```ignore
//! use astrodyn_verif_jeod::run_verification::sim_dyncomp;
//! use astrodyn_verif_parity::VerificationCaseParityExt;
//!
//! #[test]
//! fn bevy_parity_dyncomp_run2_3dof() {
//!     sim_dyncomp::run2_3dof().run_and_assert_parity::<astrodyn::Earth>();
//! }
//! ```
//!
//! ## Cadence
//!
//! Comparison happens at every reference-CSV checkpoint (typically 1–60 s),
//! same cadence the runner-vs-JEOD `run_and_assert` uses. Bit-identity at
//! the CSV cadence implies bit-identity at every intermediate integration
//! tick, since divergence between two runtimes that share the same
//! `astrodyn_*` math is monotonic — once they drift, they stay drifted —
//! so a coarser checkpoint set is equivalent in detection strength to a
//! per-tick scan.
//!
//! ## Transitivity to JEOD
//!
//! The trait carries the `bevy ≡ runner` half of the
//! `bevy ≡ runner ≈ JEOD` argument: if `runner ↔ JEOD` holds within
//! tolerance (the existing `tier3_*` tests) and `runner ↔ bevy` holds
//! bit-for-bit (these tests), then `bevy ↔ JEOD` follows by transitivity
//! within the same tolerance. Issue #389 closes the gap by making the
//! `bevy_parity_*` test set a superset of every Tier 3 topic.

use std::time::Duration;

use astrodyn_bevy::{RotationalStateC, SimulationBuilderBevyExt, TranslationalStateC};
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_verif_jeod::run_verification::{initial_conditions_from, load_reference_states};
use astrodyn_verif_jeod::tier3_csv;
use astrodyn_verif_jeod::verification::VerificationCase;
use bevy::prelude::*;
use uom::si::time::second;

mod bevy_context;
pub use bevy_context::BevySimContext;

/// Trait that turns any [`VerificationCase`] into a runner-vs-bevy
/// bit-identical state parity assertion.
///
/// This is the parity counterpart of
/// [`astrodyn_verif_jeod::VerificationCaseExt::run_and_assert`]. Both
/// consume the same scenario factory, the same reference-CSV schedule,
/// and the same propagation duration; the parity trait additionally
/// builds a Bevy [`App`] from the scenario via
/// [`SimulationBuilderBevyExt::populate_app`] and asserts every body's
/// translational + rotational state is bit-identical between the two
/// runtimes at every CSV checkpoint.
pub trait VerificationCaseParityExt {
    /// Materialize the case into both runtimes, propagate to each
    /// reference-CSV time step, and assert per-component
    /// `f64::to_bits()` equality on every body's state. Panics on the
    /// first divergence with a diagnostic naming the body, time, and
    /// component.
    ///
    /// `<P: astrodyn::Planet>` selects the planet whose
    /// [`PlanetInertial`](astrodyn::PlanetInertial) frame the bodies
    /// integrate in; today every shipped scenario is single-planet, so
    /// the call site pins `<astrodyn::Earth>` (or the relevant planet)
    /// and the bridge spawns all bodies under that frame.
    fn run_and_assert_parity<P: astrodyn::Planet>(&self);
}

impl VerificationCaseParityExt for VerificationCase {
    fn run_and_assert_parity<P: astrodyn::Planet>(&self) {
        // 1. Load the reference CSV exactly once. Only timestamps and
        //    the t=0 row matter for parity — JEOD-logged state never
        //    enters the comparison (that's the runner-vs-JEOD job in
        //    `run_and_assert`).
        let ref_path = tier3_csv::test_data_path(self.reference.file_name());
        assert!(
            ref_path.exists(),
            "JEOD reference CSV not found at {} for `{}`. Generate with: \
             docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            ref_path.display(),
            self.name,
        );
        let ref_states = load_reference_states(&self.reference, &ref_path);
        assert!(
            !ref_states.is_empty(),
            "`{}`: reference CSV {} produced 0 records",
            self.name,
            ref_path.display()
        );

        // 2. Derive `InitialConditions` from the t=0 row and build the
        //    scenario into both runtimes from the *same* factory. The
        //    fn pointer is `Copy`, so we call it twice (once per
        //    runtime) with no extra cost. The runner-side builder is
        //    also our source for `dt` — captured before `.build()`
        //    consumes it, so we don't need a third factory call just
        //    to read the integrator timestep.
        let init = initial_conditions_from(&ref_states[0]);
        let runner_builder = (self.scenario)(&init);
        let dt = runner_builder.dt;
        let mut runner_sim = runner_builder
            .build()
            .unwrap_or_else(|e| panic!("`{}`: runner build failed: {e:?}", self.name));

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        let handles = (self.scenario)(&init)
            .populate_app::<P>(&mut app)
            .unwrap_or_else(|e| panic!("`{}`: bevy populate_app failed: {e:?}", self.name));
        // Run Startup so frame-tree registration systems
        // (`register_source_frames_system`, etc.) attach
        // `FrameEntityC` / `PfixFrameEntityC` on the source entities
        // before any `pre_step` hook tries to mutate them. Without this
        // the very first `BevySimContext::set_source_position` panics
        // on the missing `FrameEntityC` lookup, since `MinimalPlugins`
        // never auto-runs `Startup` and the parity loop drives
        // `FixedUpdate` directly.
        app.world_mut().run_schedule(Startup);

        // 2b. If the case carries a pre-step factory, invoke it now so
        //     the resulting closure can capture run-once state (a loaded
        //     DE421 ephemeris, J2000 JD, source indices) the per-step
        //     body would otherwise re-derive on every call. The factory
        //     produces *two* closures from a shared captured environment
        //     so that the runner-side state and the Bevy-side state see
        //     bit-identical inputs at every record. The factory itself
        //     runs twice (`fn(&InitialConditions) -> PreStepClosure` is
        //     `Copy`) — captured state (e.g. an Ephemeris<'static> handle)
        //     is reloaded for each side, which is fine: load is
        //     deterministic and the per-step closure body sees the same
        //     numeric inputs regardless.
        let mut runner_pre_step = self.pre_step.map(|builder| builder(&init));
        let mut bevy_pre_step = self.pre_step.map(|builder| builder(&init));

        // 3. Propagate in lockstep. Rather than mirror runner's
        //    `step_until` (which would require a separate Bevy-side
        //    "step until time" helper), advance both by the same number
        //    of integration ticks per CSV record. With shared `dt` and
        //    aligned start times, the two paths reach the same sim time
        //    after the same tick count — bit-identity is the contract.
        let duration_s = self.duration.get::<second>();
        let mut current_time = ref_states[0].time;
        for record in ref_states.iter().skip(1) {
            if duration_s > 0.0 && record.time > duration_s {
                break;
            }
            // Number of ticks needed to advance from `current_time` to
            // `record.time`. Reference CSVs sample at integer multiples
            // of `dt` so the division should be exact; round to absorb
            // f64 representation jitter (e.g. 60.0 / 10.0 = 5.999…).
            let dt_steps = ((record.time - current_time) / dt).round() as usize;
            assert!(
                dt_steps > 0,
                "`{}`: reference record at t={} is not strictly after current sim time {} \
                 (scaled by dt={dt}); CSV must sample at strictly increasing times.",
                self.name,
                record.time,
                current_time,
            );
            // Run the pre-step hook on both runtimes before propagation
            // so each sees up-to-date inputs for this record. The runner
            // side mirrors `VerificationCaseExt::run_and_assert`'s call
            // shape; the Bevy side wraps `&mut World` in a
            // [`BevySimContext`] so the same closure body can mutate
            // either runtime through the [`SimContext`] surface.
            if let Some(hook) = runner_pre_step.as_mut() {
                hook(&mut runner_sim, record.time);
            }
            if let Some(hook) = bevy_pre_step.as_mut() {
                let world = app.world_mut();
                let mut ctx = BevySimContext::<P>::new(
                    world,
                    &handles.source_entities,
                    &handles.body_entities,
                );
                hook(&mut ctx, record.time);
            }

            // Advance runner.
            runner_sim
                .step_n(dt_steps)
                .unwrap_or_else(|e| panic!("`{}`: runner step_n failed: {e}", self.name));
            // Advance bevy.
            for _ in 0..dt_steps {
                app.world_mut()
                    .resource_mut::<Time<Fixed>>()
                    .advance_by(Duration::from_secs_f64(dt));
                app.world_mut().run_schedule(FixedUpdate);
            }
            current_time = record.time;

            // 4. Assert bit-identity per body, per component. The
            //    bridge `populate_app` keeps `body_entities[i]` parallel
            //    to runner's body index `i` (both consume the same
            //    `bodies` Vec), so the `i`-th entry on each side is the
            //    same body.
            for (body_idx, &entity) in handles.body_entities.iter().enumerate() {
                let runner_body = runner_sim.body(body_idx);
                let bevy_trans = astrodyn::typed_bridge::trans_typed_to_raw(
                    &app.world()
                        .get::<TranslationalStateC<P>>(entity)
                        .unwrap_or_else(|| {
                            panic!(
                                "`{}`: bevy body {body_idx} missing TranslationalStateC<{}>",
                                self.name,
                                std::any::type_name::<P>(),
                            )
                        })
                        .0,
                );
                let runner_trans_untyped =
                    astrodyn::typed_bridge::trans_typed_to_raw(&runner_body.trans);
                assert_translational_bits_eq(
                    self.name,
                    body_idx,
                    record.time,
                    &bevy_trans,
                    &runner_trans_untyped,
                );
                if let Some(runner_rot) = runner_body.rot.as_ref() {
                    let bevy_rot = astrodyn::typed_bridge::rot_typed_to_raw(
                        &app.world()
                            .get::<RotationalStateC>(entity)
                            .unwrap_or_else(|| {
                                panic!(
                                    "`{}`: bevy body {body_idx} missing RotationalStateC \
                                     (runner has rotational state)",
                                    self.name,
                                )
                            })
                            .0,
                    );
                    let runner_rot_untyped = astrodyn::typed_bridge::rot_typed_to_raw(runner_rot);
                    assert_rotational_bits_eq(
                        self.name,
                        body_idx,
                        record.time,
                        &bevy_rot,
                        &runner_rot_untyped,
                    );
                }
            }
        }
    }
}

/// Assert per-component `f64::to_bits()` equality between two
/// translational states. Failure messages name the case, body index,
/// time, and offending component so a parity drift in any single
/// scenario points straight at the broken assumption.
fn assert_translational_bits_eq(
    case_name: &str,
    body_idx: usize,
    time: f64,
    bevy: &astrodyn::TranslationalState,
    runner: &astrodyn::TranslationalState,
) {
    for i in 0..3 {
        assert_bits_eq(
            case_name,
            body_idx,
            time,
            &format!("position[{i}]"),
            bevy.position[i],
            runner.position[i],
        );
        assert_bits_eq(
            case_name,
            body_idx,
            time,
            &format!("velocity[{i}]"),
            bevy.velocity[i],
            runner.velocity[i],
        );
    }
}

/// Assert per-component `f64::to_bits()` equality between two
/// rotational states (quaternion + body-frame angular velocity).
fn assert_rotational_bits_eq(
    case_name: &str,
    body_idx: usize,
    time: f64,
    bevy: &astrodyn::RotationalState,
    runner: &astrodyn::RotationalState,
) {
    // JEOD-quat layout `[q0, q1, q2, q3]` (q0 scalar) — compare in JEOD
    // order so a mismatch on the scalar component reports the right
    // index in the failure message.
    for i in 0..4 {
        assert_bits_eq(
            case_name,
            body_idx,
            time,
            &format!("quat[{i}]"),
            bevy.quaternion.data[i],
            runner.quaternion.data[i],
        );
    }
    for i in 0..3 {
        assert_bits_eq(
            case_name,
            body_idx,
            time,
            &format!("ang_vel[{i}]"),
            bevy.ang_vel_body[i],
            runner.ang_vel_body[i],
        );
    }
}

fn assert_bits_eq(case_name: &str, body_idx: usize, time: f64, component: &str, a: f64, b: f64) {
    assert!(
        a.to_bits() == b.to_bits(),
        "`{case_name}` body {body_idx} at t={time:.6}s: {component} not bit-identical:\n  \
         bevy:   {a} (bits={:#018x})\n  \
         runner: {b} (bits={:#018x})",
        a.to_bits(),
        b.to_bits(),
    );
}
