//! Bevy ↔ runner parity for SIM_dyncomp RUN_6A / RUN_6B (atmospheric
//! drag — constant-density and MET, with optional structural-frame
//! rotation and aero-trajectory variants) and RUN_6C / RUN_6D (impulsive
//! maneuver burns). The drag cases share a `pre_step: None` recipe; the
//! maneuver cases carry a per-tick force pre-step. Either way the parity
//! wrapper drives both runtimes through the same scenario factory and
//! asserts bit-identical state at every reference-CSV checkpoint.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run6a_const_density_drag() {
    sim_dyncomp::run6a_const_density_drag().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run6b_drag() {
    sim_dyncomp::run6b_drag().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run6b_drag_aero_traj() {
    sim_dyncomp::run6b_drag_aero_traj().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run6b_drag_rotated_struct() {
    sim_dyncomp::run6b_drag_rotated_struct().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run6c_plane_change() {
    sim_dyncomp::run6c_plane_change().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run6d_departure() {
    sim_dyncomp::run6d_departure().run_and_assert_parity::<astrodyn::Earth>();
}
