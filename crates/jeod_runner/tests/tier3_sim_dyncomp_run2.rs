//! Tier 3: SIM_dyncomp RUN_2 — Point-mass gravity (3-DOF and 6-DOF)
//!
//! All simulation parameters (mu, step size, mass) are loaded from JEOD source
//! files rather than hardcoded, per issue #44.
//!
//! Phase 7 of #101 collapsed the per-test setup into the
//! [`run_verification::sim_dyncomp`](jeod_runner::run_verification::sim_dyncomp)
//! recipe family. The verification-case constructor encapsulates all JEOD
//! source loading and `Simulation` setup; the test body is a one-liner that
//! delegates to [`VerificationCaseExt::run_and_assert`].

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_dyncomp;

#[test]
fn tier3_simulation_run2_3dof() {
    sim_dyncomp::run2_3dof().run_and_assert();
}

#[test]
fn tier3_simulation_run2_6dof() {
    sim_dyncomp::run2_6dof().run_and_assert();
}
