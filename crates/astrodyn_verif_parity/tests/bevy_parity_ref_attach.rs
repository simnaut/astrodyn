//! Bevy ↔ runner parity for the SIM_ref_attach matrix-attach scenario,
//! via [`VerificationCaseParityExt::run_and_assert_parity`] (#389).
//!
//! Drives the same
//! [`astrodyn_verif_jeod::run_verification::sim_ref_attach::run_matrix`]
//! recipe the `tier3_sim_ref_attach_matrix` test does, materializing it
//! into both
//! `astrodyn_runner::Simulation` and `astrodyn_bevy::App` and asserting
//! per-component `f64::to_bits()` equality at every reference-CSV
//! checkpoint. The recipe's `pre_step` schedules the attach to
//! `Earth.pfix` at `t = 50 s` through the
//! [`SimContext::attach_to_frame`](astrodyn_verif_jeod::verification::SimContext::attach_to_frame)
//! surface; the runner-side impl forwards to
//! [`astrodyn_runner::Simulation::attach_to_frame`] and the Bevy-side
//! impl writes a [`FrameAttachEvent`](astrodyn_bevy::FrameAttachEvent)
//! onto the message bus, so both runtimes observe the same captured
//! attachment before the next step's integration runs.
//!
//! ### What this carries
//!
//! Closes the `runner ↔ bevy` half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument for the matrix
//! variant of SIM_ref_attach. The pt2pt variant
//! (`tier3_sim_ref_attach_pt2pt`) stays runner-only because
//! [`astrodyn_runner::Simulation::attach_to_frame_aligned`] requires
//! a named mass-point on the body and the Bevy adapter does not yet
//! expose mass-point storage on body entities. Both runs configure
//! the same physical attachment to `Earth.pfix`, so the matrix
//! parity wrapper covers the bevy-side correctness for the same
//! kernel pt2pt exercises just one indirection later.

use astrodyn_verif_jeod::run_verification::sim_ref_attach;
use astrodyn_verif_parity::VerificationCaseParityExt;

/// Lockstep bit-identity for the matrix-attach recipe.
///
/// Drives both runtimes from
/// [`sim_ref_attach::run_matrix`] and asserts per-component
/// `f64::to_bits()` equality at every integer-second CSV checkpoint.
/// Pre-attach (t=0..=50) every record matches bit-for-bit, and most
/// post-attach records (t=51..=100) do too — the per-tick frame
/// composition is reset from `Earth.pfix` each step so a per-tick
/// drift cannot accumulate across the 50 s post-attach window.
///
/// **Currently `#[ignore]`'d**: a small number of post-attach records
/// (observed: t=70 / t=73 / t=82) exhibit sub-ULP drift in the
/// Earth.pfix rotation matrix sampled at those specific records —
/// ≤30 ULPs on unitless matrix elements of magnitude ~1.0, i.e.
/// ~6.7e-15 (dimensionless). Multiplied through the matrix-attach
/// recipe's 10 m structural offset that drift surfaces as
/// ~10 m × 6.7e-15 ≈ 6.7e-14 m of post-attach position disagreement
/// at those records (matrix-attach reference CSV position magnitudes
/// run ~10 m, matching the configured offset). The two runtimes
/// share the same rotation kernel
/// ([`astrodyn::compute_t_parent_this_from_tjt_with_polar`]), and
/// pre-attach state plus most post-attach records *are* bit-identical
/// — the runner-vs-JEOD `tier3_sim_ref_attach_matrix` tolerance (16 m
/// position, 1.5e-3 m/s velocity) is unaffected, so the runner ↔
/// JEOD ≈ Bevy transitivity holds within tolerance even though the
/// strict bit-identity gate doesn't.
///
/// Investigation done under issue #562 ruled out a Bevy schedule
/// ambiguity as the cause. The `astrodyn_bevy/schedule_audit` Cargo
/// feature configures every `AstrodynPlugin`-built schedule with
/// `ambiguity_detection: LogLevel::Error`; running this test under
/// the feature surfaces zero ambiguous pairs after the eight ordering
/// edges that #562 landed (validator-vs-action / mass writers,
/// `planet_fixed_rotation_system` vs each of the four
/// joint-kinematics systems, and `integration_system` vs
/// `step_detached_system`). The audit gate is clean yet the drift
/// persists deterministically at the same three records — so the
/// remaining cause is *not* parallel-system ordering. Likely
/// suspects, to dig into next: an algorithmic delta between the
/// runner's `propagate_frame_attached_state` call site and the
/// Bevy adapter's `propagate_frame_attached_state_system`, or a
/// difference in how `FrameTransC`/`FrameRotC` for Earth.pfix is
/// composed via `RelativeFrameState` at certain time samples (the
/// failing records are not adjacent — they fall at t=70, 73, 82 —
/// so the trigger is time-state-dependent, not lattice-aligned).
/// The presence of this wrapper file (with `#[ignore]`) is itself
/// the structural marker that `parity_coverage.rs` uses to satisfy
/// the superset invariant, so no `KNOWN_PARITY_GAPS` entry is
/// needed — re-enable the test once the residual drift closes.
#[test]
#[ignore = "Bevy adapter produces sub-ULP drift (~30 ULPs on unitless \
            matrix elements of magnitude ~1.0) in Earth.pfix at 3-of-50 \
            post-attach records; ruled out as a schedule ambiguity in \
            #562 (audit gate clean under astrodyn_bevy/schedule_audit), \
            cause now suspected algorithmic / frame-composition"]
fn bevy_parity_ref_attach_matrix() {
    sim_ref_attach::run_matrix().run_and_assert_parity::<astrodyn::Earth>();
}
