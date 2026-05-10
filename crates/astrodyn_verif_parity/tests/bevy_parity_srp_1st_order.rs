//! Bevy ↔ runner parity for SIM_3_ORBIT_1st_ORDER — first-order
//! derivative-class thermal SRP, GEO orbit. The recipe drives the Sun
//! source via a per-record `pre_step`; the wrapper is unblocked by
//! #395's `AppSimContext::set_source_position` bridge.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_srp_1st_order_trajectory() {
    sim_srp::srp_1st_order_trajectory().run_and_assert_parity::<astrodyn::Earth>();
}
