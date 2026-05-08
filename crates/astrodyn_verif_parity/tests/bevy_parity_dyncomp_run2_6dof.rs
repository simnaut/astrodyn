//! Bevy ↔ runner parity for SIM_dyncomp RUN_2 (6-DOF baseline ISS, point-mass
//! Earth, Rk4, 8-hour propagation), via the `VerificationCaseParityExt`
//! trait introduced for issue #389.
//!
//! Companion to `bevy_parity_dyncomp_run2_3dof.rs`. The 6-DOF flavor adds
//! rotational state to the per-step bit-identity assertion — the parity
//! trait reads `RotationalStateC` whenever the runner reports
//! `body.rot.is_some()`, so this wrapper exercises that branch end-to-end.
//! If the 3-DOF pilot passes but this fails, the rotational-state
//! comparison or some attitude-touching system in the bridge is broken.
//!
//! The corresponding runner-vs-JEOD test is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_dyncomp_run2.rs::tier3_simulation_run2_6dof`.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run2_6dof() {
    sim_dyncomp::run2_6dof().run_and_assert_parity::<astrodyn::Earth>();
}
