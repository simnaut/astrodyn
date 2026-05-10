//! Bevy ↔ runner parity for SIM_dyncomp RUN_4 (3rd-body gravity from
//! Sun + Moon, per-step DE421 ephemeris updates), via the
//! `VerificationCaseParityExt` trait.
//!
//! Unblocked by issue #395's `BevySimContext`: the recipe's `pre_step`
//! drives `set_source_position` for Sun + Moon at each CSV record on
//! both runtimes; bit-identity follows from identical numeric inputs.
//!
//! The corresponding runner-vs-JEOD test lives in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_dyncomp_run4.rs`.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run4_3rd_body() {
    sim_dyncomp::run4_3rd_body().run_and_assert_parity::<astrodyn::Earth>();
}
