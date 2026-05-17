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
/// **Bit-identical at every record post #562.** Issue #562's
/// investigation traced the previously-observed ~30 ULP drift at
/// t=70 / t=73 / t=82 to a missing identity-rotation fast-path in
/// `astrodyn_frames`'s `RefFrameState::incr_right` and `::negate`
/// composition primitives. JEOD's
/// `models/utils/ref_frames/src/ref_frame_state.cc` checks
/// `Numerical::compare_exact(q.scalar, 1.0)` and skips the
/// `q.multiply(...).normalize()` + `q.left_quat_to_transformation()`
/// round-trip when the operand is bit-exactly identity — a
/// no-op mathematically but a ~1–30 ULP perturbation in f64.
/// Pre-#566 the runner and Bevy adapter disagreed on hop count for
/// Earth-at-the-heliocentric-origin: the runner aliased Earth's
/// inertial directly to root and walked one hop, the Bevy adapter
/// always carried the distinct `Earth.inertial` intermediate and
/// walked two. Without the fast-path the extra round-trip drifted
/// at GMST values where its ULP error crossed a representable-value
/// boundary. Porting JEOD's fast-path into `incr_right` and `negate`
/// makes the composition hop-count-invariant — 1-hop walks and
/// 2-hop walks through identity intermediates produce bit-identical
/// `RefFrameState`s, which is the contract `JEOD_INV: RF.13`
/// documents. PR #567 then aligned the runner's frame tree to the
/// canonical `BasePlanet { inertial, pfix }` shape the Bevy adapter
/// has always used, so both runtimes now agree on the 2-hop walk;
/// the fast-path stays in place as the bit-identity contract for
/// walks through identity intermediates regardless of which runtime
/// constructed the tree.
#[test]
fn bevy_parity_ref_attach_matrix() {
    sim_ref_attach::run_matrix().run_and_assert_parity::<astrodyn::Earth>();
}
