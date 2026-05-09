//! Bevy ↔ runner parity for SIM_3_ORBIT_1st_ORDER (DerivativeFirstOrder
//! thermal-integration SRP with per-step Sun position injection), via
//! the `VerificationCaseParityExt` trait.
//!
//! Unblocked by issue #395's `BevySimContext`: the recipe's `pre_step`
//! drives `set_source_position` for the Sun source on both runtimes
//! at each CSV record before integration runs.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_srp_1st_order_trajectory() {
    sim_srp::srp_1st_order_trajectory().run_and_assert_parity::<astrodyn::Earth>();
}
