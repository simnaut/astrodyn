//! Bevy ↔ runner parity for SIM_OrbElem (eccentric orbit with classical
//! orbital-element extras). Wrapper lands as part of #389.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_orbelem_ecc() {
    sim_derived_state::orbelem_ecc().run_and_assert_parity::<astrodyn::Earth>();
}
