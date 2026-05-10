//! Bevy ↔ runner parity for SIM_dyncomp RUN_4 (point-mass Earth + DE421
//! Sun/Moon third-body, 6-DOF, 8 hours). Closed by #395 — exercises the
//! per-record `pre_step` Sun/Moon update on both runtimes through the
//! shared `SimContext::set_source_position` surface.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run4_3rd_body() {
    sim_dyncomp::run4_3rd_body().run_and_assert_parity::<astrodyn::Earth>();
}
