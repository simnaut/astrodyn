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
use std::path::Path;

/// Parsed frame state record from a single 22-column block in the JEOD CSV.
#[derive(Debug)]
struct FrameRecord {
    position: DVec3,
    velocity: DVec3,
    ang_vel: DVec3,
    t_parent_this: DMat3,
    q_parent_this: JeodQuat,
}

/// Parse a single frame's state from a CSV row given the column base offset.
///
/// Within a 22-column frame block starting at column `base`:
///   For axis i in {0,1,2}, group starts at base + i*7:
///     position[i]        = base + i*7 + 0
///     velocity[i]        = base + i*7 + 1
///     ang_vel_this[i]    = base + i*7 + 2
///     T_parent_this[i][0]= base + i*7 + 3
///     T_parent_this[i][1]= base + i*7 + 4
///     T_parent_this[i][2]= base + i*7 + 5
///     Q.vector[i]        = base + i*7 + 6
///   Q_parent_this.scalar = base + 21
fn parse_frame_record(fields: &[&str], base: usize, line_no: usize) -> FrameRecord {
    let parse = |col: usize| -> f64 {
        fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
            panic!(
                "Failed to parse CSV at line {line_no}, col {col}: {:?} ({e})",
                fields[col]
            )
        })
    };

    let position = DVec3::new(
        parse(base),
        parse(base + 7),
        parse(base + 14),
    );
    let velocity = DVec3::new(
        parse(base + 1),
        parse(base + 8),
        parse(base + 15),
    );
    let ang_vel = DVec3::new(
        parse(base + 2),
        parse(base + 9),
        parse(base + 16),
    );

    // Transformation matrix: T[row][col] where row index = axis i
    // Row 0: cols base+3, base+4, base+5
    // Row 1: cols base+10, base+11, base+12
    // Row 2: cols base+17, base+18, base+19
    //
    // JEOD stores T in row-major order: T[i][j].
    // glam DMat3 is column-major, so DMat3::from_cols takes columns.
    // Column j of the matrix = (T[0][j], T[1][j], T[2][j]).
    let t_parent_this = DMat3::from_cols(
        DVec3::new(parse(base + 3), parse(base + 10), parse(base + 17)),
        DVec3::new(parse(base + 4), parse(base + 11), parse(base + 18)),
        DVec3::new(parse(base + 5), parse(base + 12), parse(base + 19)),
    );

    let q_vec = DVec3::new(
        parse(base + 6),
        parse(base + 13),
        parse(base + 20),
    );
    let q_scalar = parse(base + 21);
    let q_parent_this = JeodQuat::new(q_scalar, q_vec.x, q_vec.y, q_vec.z);

    FrameRecord {
        position,
        velocity,
        ang_vel,
        t_parent_this,
        q_parent_this,
    }
}

/// All three frame records for a single timestep.
#[derive(Debug)]
struct ThreeFrameRecord {
    time: f64,
    composite: FrameRecord,
    core: FrameRecord,
    structure: FrameRecord,
}

/// Load all three frames from the JEOD CSV.
fn load_three_frame_trajectory(path: &Path) -> Vec<ThreeFrameRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}\n\
             Generate with: docker build -f trick/Dockerfile -t jeod-trick .. && \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 67,
            "Malformed JEOD CSV at line {}: expected at least 67 fields (time + 3 frames x 22 cols), found {}",
            i + 1,
            fields.len(),
        );

        let line_no = i + 1;
        let time: f64 = fields[0].trim().parse().unwrap_or_else(|e| {
            panic!("Failed to parse time at line {line_no}: {:?} ({e})", fields[0])
        });

        // Composite body: columns 1..22 (base=1)
        let composite = parse_frame_record(&fields, 1, line_no);
        // Core body: columns 23..44 (base=23)
        let core = parse_frame_record(&fields, 23, line_no);
        // Structure: columns 45..66 (base=45)
        let structure = parse_frame_record(&fields, 45, line_no);

        records.push(ThreeFrameRecord {
            time,
            composite,
            core,
            structure,
        });
    }
    records
}

#[test]
fn tier3_frame_propagation_composite_to_structure() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_three_frame_trajectory(&csv_path);
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
                position: record.composite.position,
                velocity: record.composite.velocity,
            },
            rot: jeod_frames::RefFrameRot {
                q_parent_this: record.composite.q_parent_this,
                t_parent_this: record.composite.t_parent_this,
                ang_vel_this: record.composite.ang_vel,
            },
        };

        // --- Reverse: composite -> structure ---
        // propagate_reverse takes the derived (composite) state and the
        // forward offset (struct_to_composite) and recovers the source (structure).
        let computed_structure = propagate_reverse(&composite_state, &struct_to_composite);

        let pos_err = (computed_structure.trans.position - record.structure.position).length();
        let vel_err = (computed_structure.trans.velocity - record.structure.velocity).length();
        let angvel_err =
            (computed_structure.rot.ang_vel_this - record.structure.ang_vel).length();

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
                q_parent_this: record.structure.q_parent_this,
                t_parent_this: record.structure.t_parent_this,
                ang_vel_this: record.structure.ang_vel,
            },
        };

        let computed_composite = propagate_forward(&structure_state, &struct_to_composite);

        let pos_err_f = (computed_composite.trans.position - record.composite.position).length();
        let vel_err_f = (computed_composite.trans.velocity - record.composite.velocity).length();
        let angvel_err_f =
            (computed_composite.rot.ang_vel_this - record.composite.ang_vel).length();

        let t_diff_f = computed_composite.rot.t_parent_this - record.composite.t_parent_this;
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
    println!("Max position error:      {:.2e} m", max_pos_error_rev);
    println!("Max velocity error:      {:.2e} m/s", max_vel_error_rev);
    println!("Max T matrix error:      {:.2e}", max_t_error_rev);
    println!("Max angular vel error:   {:.2e} rad/s", max_angvel_error_rev);
    println!();
    println!("--- Forward: structure -> composite ---");
    println!("Max position error:      {:.2e} m", max_pos_error_fwd);
    println!("Max velocity error:      {:.2e} m/s", max_vel_error_fwd);
    println!("Max T matrix error:      {:.2e}", max_t_error_fwd);
    println!("Max angular vel error:   {:.2e} rad/s", max_angvel_error_fwd);

    // These are pure coordinate transforms, no integration involved.
    // Differences should be near machine precision.

    // Reverse: composite -> structure
    assert!(
        max_pos_error_rev < 1e-6,
        "Reverse position error {:.2e} m exceeds 1e-6 m threshold",
        max_pos_error_rev
    );
    assert!(
        max_vel_error_rev < 1e-6,
        "Reverse velocity error {:.2e} m/s exceeds 1e-6 m/s threshold",
        max_vel_error_rev
    );
    assert!(
        max_t_error_rev < 1e-10,
        "Reverse T matrix error {:.2e} exceeds 1e-10 threshold",
        max_t_error_rev
    );
    assert!(
        max_angvel_error_rev < 1e-12,
        "Reverse angular velocity error {:.2e} rad/s exceeds 1e-12 rad/s threshold",
        max_angvel_error_rev
    );

    // Forward: structure -> composite
    assert!(
        max_pos_error_fwd < 1e-6,
        "Forward position error {:.2e} m exceeds 1e-6 m threshold",
        max_pos_error_fwd
    );
    assert!(
        max_vel_error_fwd < 1e-6,
        "Forward velocity error {:.2e} m/s exceeds 1e-6 m/s threshold",
        max_vel_error_fwd
    );
    assert!(
        max_t_error_fwd < 1e-10,
        "Forward T matrix error {:.2e} exceeds 1e-10 threshold",
        max_t_error_fwd
    );
    assert!(
        max_angvel_error_fwd < 1e-12,
        "Forward angular velocity error {:.2e} rad/s exceeds 1e-12 rad/s threshold",
        max_angvel_error_fwd
    );
}

#[test]
fn tier3_frame_propagation_core_equals_composite() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_three_frame_trajectory(&csv_path);
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
        let pos_diff = (record.core.position - record.composite.position).length();
        let vel_diff = (record.core.velocity - record.composite.velocity).length();
        let angvel_diff = (record.core.ang_vel - record.composite.ang_vel).length();

        let t_diff_mat = record.core.t_parent_this - record.composite.t_parent_this;
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
    println!("Max position diff:     {:.2e} m", max_pos_diff);
    println!("Max velocity diff:     {:.2e} m/s", max_vel_diff);
    println!("Max T matrix diff:     {:.2e}", max_t_diff);
    println!("Max angular vel diff:  {:.2e} rad/s", max_angvel_diff);

    // Single body: core and composite must be identical (zero offset).
    assert!(
        max_pos_diff < 1e-12,
        "Core vs composite position diff {:.2e} m -- single body should have zero offset",
        max_pos_diff
    );
    assert!(
        max_vel_diff < 1e-12,
        "Core vs composite velocity diff {:.2e} m/s -- single body should have zero offset",
        max_vel_diff
    );
    assert!(
        max_t_diff < 1e-14,
        "Core vs composite T diff {:.2e} -- single body should have zero offset",
        max_t_diff
    );
    assert!(
        max_angvel_diff < 1e-14,
        "Core vs composite angular velocity diff {:.2e} rad/s -- single body should have zero offset",
        max_angvel_diff
    );
}

#[test]
fn tier3_frame_propagation_round_trip() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_three_frame_trajectory(&csv_path);
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
                position: record.composite.position,
                velocity: record.composite.velocity,
            },
            rot: jeod_frames::RefFrameRot {
                q_parent_this: record.composite.q_parent_this,
                t_parent_this: record.composite.t_parent_this,
                ang_vel_this: record.composite.ang_vel,
            },
        };

        // Forward then reverse round-trip
        let structure = propagate_reverse(&composite_state, &struct_to_composite);
        let recovered = propagate_forward(&structure, &struct_to_composite);

        let pos_err = (recovered.trans.position - composite_state.trans.position).length();
        let vel_err = (recovered.trans.velocity - composite_state.trans.velocity).length();
        let angvel_err =
            (recovered.rot.ang_vel_this - composite_state.rot.ang_vel_this).length();

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
    println!("Max position error:      {:.2e} m", max_pos_rt);
    println!("Max velocity error:      {:.2e} m/s", max_vel_rt);
    println!("Max T matrix error:      {:.2e}", max_t_rt);
    println!("Max angular vel error:   {:.2e} rad/s", max_angvel_rt);

    // Round-trip should be near machine precision.
    assert!(
        max_pos_rt < 1e-8,
        "Round-trip position error {:.2e} m exceeds 1e-8 m threshold",
        max_pos_rt
    );
    assert!(
        max_vel_rt < 1e-8,
        "Round-trip velocity error {:.2e} m/s exceeds 1e-8 m/s threshold",
        max_vel_rt
    );
    assert!(
        max_t_rt < 1e-14,
        "Round-trip T matrix error {:.2e} exceeds 1e-14 threshold",
        max_t_rt
    );
    assert!(
        max_angvel_rt < 1e-14,
        "Round-trip angular velocity error {:.2e} rad/s exceeds 1e-14 rad/s threshold",
        max_angvel_rt
    );
}
