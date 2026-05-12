//! Bevy ↔ runner parity for the SIM_dyncomp RUN_6B drag scenario with
//! a non-identity structural-to-body rotation (1 kg sphere, point-mass
//! gravity, MET atmosphere, Cd-based drag, 15° eigen rotation about
//! [1,1,1]) — a topic-alias wrapper.
//!
//! The same `sim_dyncomp::run6b_drag_rotated_struct()` recipe is
//! already exercised under `bevy_parity_dyncomp_run6.rs`, but the
//! `parity_coverage` superset check matches tier3 topics against
//! parity wrapper file stems using exact-or-prefix on
//! `bevy_parity_<topic>`. The owning tier3 file
//! (`tier3_sim_drag_rot_verif.rs`) canonicalizes to topic
//! `drag_rot_verif`, which the `dyncomp_run6` group cannot satisfy by
//! name. This file supplies the matching stem; asserting the same
//! bit-identity twice is intentional and cheap relative to the cost
//! of leaving a stale `KNOWN_PARITY_GAPS` entry in place.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_drag_rot_verif() {
    sim_dyncomp::run6b_drag_rotated_struct().run_and_assert_parity::<astrodyn::Earth>();
}
