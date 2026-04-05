//! Tier 3: Cross-validate frame propagation (structure <-> composite <-> core)
//! against JEOD SIM_dyncomp RUN_2.
//!
//! JEOD logs all three body frames (composite_body, core_body, structure).
//! We take the composite_body state and use propagate_forward/propagate_reverse
//! with the ISS mass offset to derive structure and core_body frames, then
//! compare against JEOD's logged values.
//!
//! Since the ISS sim uses a single body with no children, core_body == composite_body
//! (the core-to-composite offset is zero). The interesting test is composite <-> structure,
//! which exercises the CoM offset of (-3.0, -1.5, 4.0) meters.
//!
//! These are pure coordinate transforms (no integration), so differences should be
//! near machine precision.

use glam::{DMat3, DVec3};
use jeod_dynamics::propagation::{propagate_forward, propagate_reverse};
use jeod_dynamics::MassPointState;
use jeod_math::JeodQuat;
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::dyncomp_csv::load_dyncomp_csv;
use std::path::Path;

#[test]
fn tier3_frame_propagation_composite_to_structure() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(
        trajectory.len() >= 100,
        "Expected at least 100 data points, got {}",
        trajectory.len()
    );

    // ISS mass offset from SIM_dyncomp Modified_data/mass.py (set_mass_iss).
    // CoM position = (-3.0, -1.5, 4.0) meters in structure frame.
    // No body-frame rotation offset (eigen_angle = 0 -> identity).
    //
    // This is the structure-to-composite offset: the composite body frame
    // origin (CoM) is at this position relative to the structure frame origin.
    let struct_to_composite = MassPointState {
        position: DVec3::new(-3.0, -1.5, 4.0),
        t_parent_this: DMat3::IDENTITY,
    };

    let mut max_pos_error_rev = 0.0_f64;
    let mut max_vel_error_rev = 0.0_f64;
    let mut max_t_error_rev = 0.0_f64;
    let mut max_angvel_error_rev = 0.0_f64;

    let mut max_pos_error_fwd = 0.0_f64;
    let mut max_vel_error_fwd = 0.0_f64;
    let mut max_t_error_fwd = 0.0_f64;
    let mut max_angvel_error_fwd = 0.0_f64;

    for record in &trajectory {
        // Build the composite body RefFrameState from CSV data.
        let composite_state = jeod_frames::RefFrameState {
            trans: jeod_frames::RefFrameTrans {
                position: record.composite_body.position,
                velocity: record.composite_body.velocity,
            },
            rot: jeod_frames::RefFrameRot {
                q_parent_this: JeodQuat::from_glam(record.composite_body.quaternion),
                t_parent_this: record.composite_body.t_parent_this,
                ang_vel_this: record.composite_body.ang_vel,
            },
        };

        // --- Reverse: composite -> structure ---
        // propagate_reverse takes the derived (composite) state and the
        // forward offset (struct_to_composite) and recovers the source (structure).
        let computed_structure = propagate_reverse(&composite_state, &struct_to_composite);

        let pos_err = (computed_structure.trans.position - record.structure.position).length();
        let vel_err = (computed_structure.trans.velocity - record.structure.velocity).length();
        let angvel_err = (computed_structure.rot.ang_vel_this - record.structure.ang_vel).length();

        // Max element-wise error in the transformation matrix
        let t_diff = computed_structure.rot.t_parent_this - record.structure.t_parent_this;
        let t_err = [t_diff.x_axis, t_diff.y_axis, t_diff.z_axis]
            .iter()
            .flat_map(|col| [col.x.abs(), col.y.abs(), col.z.abs()])
            .fold(0.0_f64, f64::max);

        max_pos_error_rev = max_pos_error_rev.max(pos_err);
        max_vel_error_rev = max_vel_error_rev.max(vel_err);
        max_t_error_rev = max_t_error_rev.max(t_err);
        max_angvel_error_rev = max_angvel_error_rev.max(angvel_err);

        // --- Forward: structure -> composite ---
        // Build the structure RefFrameState from CSV data.
        let structure_state = jeod_frames::RefFrameState {
            trans: jeod_frames::RefFrameTrans {
                position: record.structure.position,
                velocity: record.structure.velocity,
            },
            rot: jeod_frames::RefFrameRot {
                q_parent_this: JeodQuat::from_glam(record.structure.quaternion),
                t_parent_this: record.structure.t_parent_this,
                ang_vel_this: record.structure.ang_vel,
            },
        };

        let computed_composite = propagate_forward(&structure_state, &struct_to_composite);

        let pos_err_f =
            (computed_composite.trans.position - record.composite_body.position).length();
        let vel_err_f =
            (computed_composite.trans.velocity - record.composite_body.velocity).length();
        let angvel_err_f =
            (computed_composite.rot.ang_vel_this - record.composite_body.ang_vel).length();

        let t_diff_f = computed_composite.rot.t_parent_this - record.composite_body.t_parent_this;
        let t_err_f = [t_diff_f.x_axis, t_diff_f.y_axis, t_diff_f.z_axis]
            .iter()
            .flat_map(|col| [col.x.abs(), col.y.abs(), col.z.abs()])
            .fold(0.0_f64, f64::max);

        max_pos_error_fwd = max_pos_error_fwd.max(pos_err_f);
        max_vel_error_fwd = max_vel_error_fwd.max(vel_err_f);
        max_t_error_fwd = max_t_error_fwd.max(t_err_f);
        max_angvel_error_fwd = max_angvel_error_fwd.max(angvel_err_f);
    }

    println!("=== Tier 3 Frame Propagation Cross-Validation (RUN_2) ===");
    println!(
        "Duration: {} s ({} data points)",
        trajectory.last().unwrap().time,
        trajectory.len()
    );
    println!();
    println!("--- Reverse: composite -> structure ---");
    println!("Max position error:      {:.6e} m", max_pos_error_rev);
    println!("Max velocity error:      {:.6e} m/s", max_vel_error_rev);
    println!("Max T matrix error:      {:.6e}", max_t_error_rev);
    println!(
        "Max angular vel error:   {:.6e} rad/s",
        max_angvel_error_rev
    );
    println!();
    println!("--- Forward: structure -> composite ---");
    println!("Max position error:      {:.6e} m", max_pos_error_fwd);
    println!("Max velocity error:      {:.6e} m/s", max_vel_error_fwd);
    println!("Max T matrix error:      {:.6e}", max_t_error_fwd);
    println!(
        "Max angular vel error:   {:.6e} rad/s",
        max_angvel_error_fwd
    );

    // Build StateLog vectors for position+velocity+angular velocity comparison
    let mut our_states = Vec::with_capacity(trajectory.len());
    let mut ref_states = Vec::with_capacity(trajectory.len());

    for record in &trajectory {
        let composite_state = jeod_frames::RefFrameState {
            trans: jeod_frames::RefFrameTrans {
                position: record.composite_body.position,
                velocity: record.composite_body.velocity,
            },
            rot: jeod_frames::RefFrameRot {
                q_parent_this: JeodQuat::from_glam(record.composite_body.quaternion),
                t_parent_this: record.composite_body.t_parent_this,
                ang_vel_this: record.composite_body.ang_vel,
            },
        };

        let computed_structure = propagate_reverse(&composite_state, &struct_to_composite);

        our_states.push(StateLog {
            time: record.time,
            position: Some(computed_structure.trans.position),
            velocity: Some(computed_structure.trans.velocity),
            ang_vel: Some(computed_structure.rot.ang_vel_this),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: record.time,
            position: Some(record.structure.position),
            velocity: Some(record.structure.velocity),
            ang_vel: Some(record.structure.ang_vel),
            acceleration: record.derivs.as_ref().map(|d| d.trans_accel),
            ang_accel: record.derivs.as_ref().map(|d| d.rot_accel),
            ..Default::default()
        });
    }

    let mut report = CrossvalReport::compute(
        "tier3_frame_propagation_composite_to_structure",
        &our_states,
        &ref_states,
    );
    report.add_extra("rev_T_matrix", max_t_error_rev, "");
    assert!(max_t_error_rev < 1.166e-15, "rev_T_matrix");
    report.add_extra("fwd_position", max_pos_error_fwd, "m");
    assert!(max_pos_error_fwd < 2.031e-9, "fwd_position");
    report.add_extra("fwd_velocity", max_vel_error_fwd, "m/s");
    assert!(max_vel_error_fwd < 1.969e-12, "fwd_velocity");
    report.add_extra("fwd_T_matrix", max_t_error_fwd, "");
    assert!(max_t_error_fwd < 1.166e-15, "fwd_T_matrix");
    report.add_extra("fwd_omega", max_angvel_error_fwd, "rad/s");
    assert!(max_angvel_error_fwd < 1e-15, "fwd_omega");
    report.write();

    // These are pure coordinate transforms, no integration involved.
    // Differences should be near machine precision.

    // Reverse: composite -> structure (assert_* covers per-component max errors)
    report.assert_position([9.779e-10, 1.956e-9, 9.779e-10]);
    report.assert_velocity([1.91e-12, 9.55e-13, 9.55e-13]);
    report.assert_ang_vel([1e-15; 3]);

    assert!(
        max_t_error_rev < 1.166e-15,
        "Reverse T matrix error {:.6e} exceeds 1.166e-15 threshold",
        max_t_error_rev
    );

    // Forward: structure -> composite
    assert!(
        max_pos_error_fwd < 2.031e-9,
        "Forward position error {:.6e} m exceeds 2.031e-9 m threshold",
        max_pos_error_fwd
    );
    assert!(
        max_vel_error_fwd < 1.969e-12,
        "Forward velocity error {:.6e} m/s exceeds 1.969e-12 m/s threshold",
        max_vel_error_fwd
    );
    assert!(
        max_t_error_fwd < 1.166e-15,
        "Forward T matrix error {:.6e} exceeds 1.166e-15 threshold",
        max_t_error_fwd
    );
    assert!(
        max_angvel_error_fwd < 1e-15,
        "Forward angular velocity error {:.6e} rad/s exceeds 1e-15 rad/s threshold",
        max_angvel_error_fwd
    );
}

#[test]
fn tier3_frame_propagation_core_equals_composite() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(
        trajectory.len() >= 100,
        "Expected at least 100 data points, got {}",
        trajectory.len()
    );

    // The ISS sim has a single body with no children, so core_body == composite_body.
    // Verify this invariant holds across the entire trajectory.
    let mut max_pos_diff = 0.0_f64;
    let mut max_vel_diff = 0.0_f64;
    let mut max_angvel_diff = 0.0_f64;
    let mut max_t_diff = 0.0_f64;

    for record in &trajectory {
        let pos_diff = (record.core_body.position - record.composite_body.position).length();
        let vel_diff = (record.core_body.velocity - record.composite_body.velocity).length();
        let angvel_diff = (record.core_body.ang_vel - record.composite_body.ang_vel).length();

        let t_diff_mat = record.core_body.t_parent_this - record.composite_body.t_parent_this;
        let t_diff = [t_diff_mat.x_axis, t_diff_mat.y_axis, t_diff_mat.z_axis]
            .iter()
            .flat_map(|col| [col.x.abs(), col.y.abs(), col.z.abs()])
            .fold(0.0_f64, f64::max);

        max_pos_diff = max_pos_diff.max(pos_diff);
        max_vel_diff = max_vel_diff.max(vel_diff);
        max_angvel_diff = max_angvel_diff.max(angvel_diff);
        max_t_diff = max_t_diff.max(t_diff);
    }

    println!("=== Core == Composite Invariant Check (RUN_2) ===");
    println!(
        "Duration: {} s ({} data points)",
        trajectory.last().unwrap().time,
        trajectory.len()
    );
    println!("Max position diff:     {:.6e} m", max_pos_diff);
    println!("Max velocity diff:     {:.6e} m/s", max_vel_diff);
    println!("Max T matrix diff:     {:.6e}", max_t_diff);
    println!("Max angular vel diff:  {:.6e} rad/s", max_angvel_diff);

    let mut report =
        CrossvalReport::compute("tier3_frame_propagation_core_equals_composite", &[], &[]);
    report.add_extra("position", max_pos_diff, "m");
    assert!(max_pos_diff < 1e-12, "position");
    report.add_extra("velocity", max_vel_diff, "m/s");
    assert!(max_vel_diff < 1e-12, "velocity");
    report.add_extra("T_matrix", max_t_diff, "");
    assert!(max_t_diff < 1.166e-15, "T_matrix");
    report.add_extra("omega", max_angvel_diff, "rad/s");
    assert!(max_angvel_diff < 1e-14, "omega");
    report.write();

    // Single body: core and composite must be identical (zero offset).
    assert!(
        max_pos_diff < 1e-12,
        "Core vs composite position diff {:.6e} m -- single body should have zero offset",
        max_pos_diff
    );
    assert!(
        max_vel_diff < 1e-12,
        "Core vs composite velocity diff {:.6e} m/s -- single body should have zero offset",
        max_vel_diff
    );
    assert!(
        max_t_diff < 1.166e-15,
        "Core vs composite T diff {:.6e} -- single body should have zero offset",
        max_t_diff
    );
    assert!(
        max_angvel_diff < 1e-14,
        "Core vs composite angular velocity diff {:.6e} rad/s -- single body should have zero offset",
        max_angvel_diff
    );
}

#[test]
fn tier3_frame_propagation_round_trip() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(
        trajectory.len() >= 100,
        "Expected at least 100 data points, got {}",
        trajectory.len()
    );

    let struct_to_composite = MassPointState {
        position: DVec3::new(-3.0, -1.5, 4.0),
        t_parent_this: DMat3::IDENTITY,
    };

    // Round-trip: composite -> structure -> composite should recover the original.
    let mut max_pos_rt = 0.0_f64;
    let mut max_vel_rt = 0.0_f64;
    let mut max_t_rt = 0.0_f64;
    let mut max_angvel_rt = 0.0_f64;

    for record in &trajectory {
        let composite_state = jeod_frames::RefFrameState {
            trans: jeod_frames::RefFrameTrans {
                position: record.composite_body.position,
                velocity: record.composite_body.velocity,
            },
            rot: jeod_frames::RefFrameRot {
                q_parent_this: JeodQuat::from_glam(record.composite_body.quaternion),
                t_parent_this: record.composite_body.t_parent_this,
                ang_vel_this: record.composite_body.ang_vel,
            },
        };

        // Forward then reverse round-trip
        let structure = propagate_reverse(&composite_state, &struct_to_composite);
        let recovered = propagate_forward(&structure, &struct_to_composite);

        let pos_err = (recovered.trans.position - composite_state.trans.position).length();
        let vel_err = (recovered.trans.velocity - composite_state.trans.velocity).length();
        let angvel_err = (recovered.rot.ang_vel_this - composite_state.rot.ang_vel_this).length();

        let t_diff = recovered.rot.t_parent_this - composite_state.rot.t_parent_this;
        let t_err = [t_diff.x_axis, t_diff.y_axis, t_diff.z_axis]
            .iter()
            .flat_map(|col| [col.x.abs(), col.y.abs(), col.z.abs()])
            .fold(0.0_f64, f64::max);

        max_pos_rt = max_pos_rt.max(pos_err);
        max_vel_rt = max_vel_rt.max(vel_err);
        max_t_rt = max_t_rt.max(t_err);
        max_angvel_rt = max_angvel_rt.max(angvel_err);
    }

    println!("=== Round-Trip Check: composite -> structure -> composite (RUN_2) ===");
    println!(
        "Duration: {} s ({} data points)",
        trajectory.last().unwrap().time,
        trajectory.len()
    );
    println!("Max position error:      {:.6e} m", max_pos_rt);
    println!("Max velocity error:      {:.6e} m/s", max_vel_rt);
    println!("Max T matrix error:      {:.6e}", max_t_rt);
    println!("Max angular vel error:   {:.6e} rad/s", max_angvel_rt);

    let mut report = CrossvalReport::compute("tier3_frame_propagation_round_trip", &[], &[]);
    report.add_extra("position", max_pos_rt, "m");
    assert!(max_pos_rt < 1e-8, "position");
    report.add_extra("velocity", max_vel_rt, "m/s");
    assert!(max_vel_rt < 1e-8, "velocity");
    report.add_extra("T_matrix", max_t_rt, "");
    assert!(max_t_rt < 1e-14, "T_matrix");
    report.add_extra("omega", max_angvel_rt, "rad/s");
    assert!(max_angvel_rt < 1e-14, "omega");
    report.write();

    // Round-trip should be near machine precision.
    assert!(
        max_pos_rt < 1e-8,
        "Round-trip position error {:.6e} m exceeds 1e-8 m threshold",
        max_pos_rt
    );
    assert!(
        max_vel_rt < 1e-8,
        "Round-trip velocity error {:.6e} m/s exceeds 1e-8 m/s threshold",
        max_vel_rt
    );
    assert!(
        max_t_rt < 1e-14,
        "Round-trip T matrix error {:.6e} exceeds 1e-14 threshold",
        max_t_rt
    );
    assert!(
        max_angvel_rt < 1e-14,
        "Round-trip angular velocity error {:.6e} rad/s exceeds 1e-14 rad/s threshold",
        max_angvel_rt
    );
}
