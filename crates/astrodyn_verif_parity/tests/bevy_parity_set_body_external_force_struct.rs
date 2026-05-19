//! Bevy ↔ runner parity for the **structural-frame** external load
//! surface — the lockstep gate for issue #510 Part 2.
//!
//! The struct-frame load setters (`Simulation::set_body_external_force_struct`
//! and `_torque_struct`) feed `SimBody.external_force_struct` /
//! `external_torque_struct` on the runner side. The Bevy adapter mirrors
//! them through new `ExternalForceStructC` / `ExternalTorqueStructC`
//! components and an extended `force_collection_system` branch that
//! rotates struct → inertial / body using the body's current attitude
//! (`T_inertial_struct = T_struct_body^T * T_inertial_body`, mirroring
//! the runner's `simulation/step/integrate.rs:85-105` and JEOD's
//! `dyn_body_collect.cc:219-221`).
//!
//! The recipes paired below ([`force_struct_via_pre_step`],
//! [`torque_struct_via_pre_step`], [`force_and_torque_struct_via_pre_step`])
//! build a 6-DOF body with a non-trivial structural-to-body rotation
//! (30° about z) **and** a non-trivial initial inertial-body attitude
//! (45° about y), so all three frames are distinct at runtime — the
//! struct-frame load is exercised through both `T_struct_body` and
//! `T_inertial_body` non-identity factors, ruling out tests that
//! short-circuit when either matrix is identity.
//!
//! Parity asserts `runner ↔ bevy` bit-identity at every synthetic
//! record (the `analytical_tolerances()` setting in the recipe is zero
//! per-component). Any divergence here means the Bevy adapter's
//! struct-frame branch in `force_collection_system` drifted from the
//! runner's `integrate.rs` shape — the regression-protection
//! mechanism the issue specifies.

use astrodyn_verif_jeod::run_verification::sim_force_torque_response;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_set_body_external_force_struct_force_only() {
    sim_force_torque_response::force_struct_via_pre_step()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_set_body_external_force_struct_torque_only() {
    sim_force_torque_response::torque_struct_via_pre_step()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_set_body_external_force_struct_both() {
    sim_force_torque_response::force_and_torque_struct_via_pre_step()
        .run_and_assert_parity::<astrodyn::Earth>();
}
