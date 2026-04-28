//! Tier 3: SIM_verif_attach_mass — mass tree attach/detach cross-validation.
//!
//! Reproduces 8 representative runs from JEOD's `SIM_verif_attach_mass`
//! (``models/dynamics/body_action/verif/SIM_verif_attach_mass/SET_test/``)
//! and cross-validates our `MassTree` composite mass, center of mass, and
//! inertia tensor against the `mass.out` files produced by JEOD's
//! `MassBody::print_tree()`.
//!
//! The JEOD sim itself is "initialization-only" — Trick stops at t=0 (or
//! t=2s for runs exercising runtime detach/reattach) and dumps the mass tree
//! via `parent_body.print_tree(file_name, 10)`. There is no trajectory to
//! propagate, so the reference data is the final composite properties after
//! all attach / detach actions have fired.
//!
//! Scenarios covered:
//! - RUN_01: parent + 1 child, explicit offset, Body inertia spec
//! - RUN_02: parent + 2 children (both attached to parent)
//! - RUN_03: parent + 2 children chained (child2 attached to child1)
//! - RUN_04: same topology as RUN_03, different offset along -z
//! - RUN_10: runtime detach — parent gains 3 children then detaches child2
//! - RUN_11: runtime reattach — child2 moves to new offset + rotation
//! - RUN_101: `BodyAttachAligned` via named mass points (simple)
//! - RUN_102: `BodyAttachAligned` chained across three bodies
//!
//! Supplements the analytical tests in
//! `crates/jeod_dynamics/tests/tier3_mass_attach_detach.rs` with direct
//! JEOD cross-validation.

use glam::{DMat3, DVec3};
use jeod_dynamics::{MassBodyId, MassProperties, MassTree};
use jeod_math::euler_angles::{compute_matrix_from_euler_angles_typed, EulerSequence};
use uom::si::angle::radian;
use uom::si::f64::Angle;

/// Compact f64 wrapper that mirrors the pre-Phase-10 bare-`f64` surface.
/// Lifts the radian inputs into `uom` `Angle`s and forwards to the typed
/// kernel — bit-identical numerics.
fn compute_matrix_from_euler_angles(angles: [f64; 3], sequence: EulerSequence) -> DMat3 {
    let typed = [
        Angle::new::<radian>(angles[0]),
        Angle::new::<radian>(angles[1]),
        Angle::new::<radian>(angles[2]),
    ];
    compute_matrix_from_euler_angles_typed(typed, sequence)
}
use jeod_test_data::apollo_mass_tree::{parse_print_tree, PrintedBody, PrintedTree};
use jeod_test_data::crossval::CrossvalReport;

// ════════════════════════════════════════════════════════════════════
// Test data path / loading
// ════════════════════════════════════════════════════════════════════

fn test_data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
}

fn load_reference(filename: &str) -> PrintedTree {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "JEOD reference data not found at {}.\n\
         Generate with:\n\
         docker run --rm \\\n\
           -v $(pwd)/test_data:/output \\\n\
           -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \\\n\
           jeod-trick",
        path.display()
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    parse_print_tree(&content)
}

// ════════════════════════════════════════════════════════════════════
// MassPropertiesInit porting helpers
// ════════════════════════════════════════════════════════════════════
//
// Ports the relevant `inertia_spec` branches from
// `models/dynamics/mass/src/mass_properties_init.cc:initialize_mass_properties`.
// Only the spec/orientation combinations exercised by the chosen runs are
// implemented here; see that file for the full enumeration (NoSpec, Body,
// StructCG, Struct, SpecCG, Spec).

/// Parallel axis theorem in the JEOD convention: returns the inertia of a
/// point mass located at `offset` from the reference point. Equivalent to
/// `MassBody::compute_point_mass_inertia` in `mass_point_mass_inertia.cc`.
fn point_mass_inertia(mass: f64, offset: DVec3) -> DMat3 {
    // I[i][j] = m * (r^2 * delta_ij - r[i] * r[j])
    let r_sq = offset.length_squared();
    let outer = DMat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
    DMat3::from_diagonal(DVec3::splat(r_sq)) * mass - outer * mass
}

/// Build `MassProperties` using JEOD's `Body` inertia spec: the inertia
/// tensor is already in body-frame axes through the CoM. Identity
/// structure-to-body rotation (the default for all runs we exercise here).
fn mass_body_spec(mass: f64, position: DVec3, inertia_body: DMat3) -> MassProperties {
    MassProperties::with_inertia(mass, inertia_body, position)
}

/// Build `MassProperties` using JEOD's `Struct` inertia spec: the inertia
/// tensor is in structural axes about the **structural origin**. Shift to
/// the CoM (parallel axis) and rotate to body frame (identity → no-op here).
fn mass_struct_spec(mass: f64, position: DVec3, inertia_struct_origin: DMat3) -> MassProperties {
    // JEOD: subtract point-mass-at-CoM shift, then transform by T_parent_this.
    // All runs we port use T_parent_this = I, so we only need the shift.
    let offset_inertia = point_mass_inertia(mass, position);
    let body_inertia = inertia_struct_origin - offset_inertia;
    MassProperties::with_inertia(mass, body_inertia, position)
}

/// Build `MassProperties` using JEOD's `StructCG` inertia spec: the inertia
/// tensor is in structural axes about the body CoM. Transform to body
/// (identity rotation → tensor unchanged).
fn mass_struct_cg_spec(mass: f64, position: DVec3, inertia_struct: DMat3) -> MassProperties {
    MassProperties::with_inertia(mass, inertia_struct, position)
}

// ════════════════════════════════════════════════════════════════════
// Baseline mass bodies (ported from JEOD Modified_data/*.py)
// ════════════════════════════════════════════════════════════════════

/// `parent_mass_default()` from `Modified_data/parent_mass.py`.
fn parent_default() -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(0.41666667, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.1666667),
    );
    mass_body_spec(1.0, DVec3::ZERO, inertia)
}

/// `child1_mass_default()`.
fn child1_default() -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(0.41666667, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.1666667),
    );
    mass_body_spec(1.0, DVec3::ZERO, inertia)
}

/// `child2_mass_default()` — identical geometry to child1.
fn child2_default() -> MassProperties {
    child1_default()
}

/// `child3_mass_default()`: Body spec, diag(0.3333, 0.4167, 0.0833) about CoM.
fn child3_default() -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(0.33333333, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.0833333333),
    );
    mass_body_spec(1.0, DVec3::ZERO, inertia)
}

// ════════════════════════════════════════════════════════════════════
// Validation helpers
// ════════════════════════════════════════════════════════════════════

/// Scalar error metrics between our MassTree composite and the JEOD
/// `PrintedBody`. Used both for tolerance assertions and for tracking the
/// overall max error written to the cross-validation report. Returned as
/// `(mass_err, com_err, inertia_err)` where:
/// - `mass_err`: absolute difference in composite mass (kg).
/// - `com_err`: Euclidean (L2) distance between CoM vectors (m).
/// - `inertia_err`: max over the three matrix columns of the L2 distance
///   between the column vectors (kg·m²). This aggregates a 3x3 matrix
///   error into a single scalar while staying sensitive to any one column
///   drifting — it is *not* a strict per-element max delta.
fn composite_errors(tree: &MassTree, id: MassBodyId, reference: &PrintedBody) -> (f64, f64, f64) {
    let body = tree.get(id);
    let comp = &body.composite_properties;
    let mass_err = (comp.mass - reference.composite_mass).abs();
    let com_err = (comp.position - reference.composite_cm).length();
    let inertia_err = [
        (comp.inertia.x_axis - reference.composite_inertia.x_axis).length(),
        (comp.inertia.y_axis - reference.composite_inertia.y_axis).length(),
        (comp.inertia.z_axis - reference.composite_inertia.z_axis).length(),
    ]
    .iter()
    .copied()
    .fold(0.0_f64, f64::max);
    (mass_err, com_err, inertia_err)
}

/// Assert and track errors for a single body. Tolerances must be wide enough
/// to absorb JEOD's `%20lf` formatting in `mass_print_body.cc` — width 20 with
/// the default `%f` precision of **6 digits after the decimal point** (not
/// 6 significant figures). JEOD's printed reference values therefore lose
/// precision relative to our doubles; for values near unity the comparison
/// floor is on the order of ~5e-7 per kg·m² element.
#[allow(clippy::too_many_arguments)]
fn check_body(
    run: &str,
    body_label: &str,
    tree: &MassTree,
    id: MassBodyId,
    reference: &PrintedBody,
    tol_mass: f64,
    tol_com: f64,
    tol_inertia: f64,
    max_errors: &mut MaxErrors,
) {
    let (mass_err, com_err, inertia_err) = composite_errors(tree, id, reference);
    max_errors.mass = max_errors.mass.max(mass_err);
    max_errors.com = max_errors.com.max(com_err);
    max_errors.inertia = max_errors.inertia.max(inertia_err);

    assert!(
        mass_err < tol_mass,
        "[{run}:{body_label}] composite mass diff {mass_err:.3e} >= tol {tol_mass:.3e}"
    );
    assert!(
        com_err < tol_com,
        "[{run}:{body_label}] composite CoM diff {com_err:.3e} >= tol {tol_com:.3e}"
    );
    assert!(
        inertia_err < tol_inertia,
        "[{run}:{body_label}] composite inertia diff {inertia_err:.3e} >= tol {tol_inertia:.3e}"
    );
}

struct MaxErrors {
    mass: f64,
    com: f64,
    inertia: f64,
}

impl MaxErrors {
    fn new() -> Self {
        Self {
            mass: 0.0,
            com: 0.0,
            inertia: 0.0,
        }
    }
}

/// Tolerances calibrated for JEOD `%20lf` (6 decimal-place) printf precision
/// of `mass.out`. Composite mass ~1-3 kg, inertia ~1 kg·m², so 5e-6 covers
/// the print-rounding floor with plenty of margin (5% rule applied to the
/// ~1e-6 noise floor).
const TOL_MASS: f64 = 5.0e-6;
const TOL_COM: f64 = 5.0e-6;
const TOL_INERTIA: f64 = 5.0e-6;

// ════════════════════════════════════════════════════════════════════
// Run-specific mass tree builders
// ════════════════════════════════════════════════════════════════════

/// RUN_01: attach child1 at [0, 1.5, 0] with rotation
/// `[[1,0,0],[0,0,1],[0,-1,0]]` (90° roll about +x, per `attach1_optionA`).
fn build_run_01() -> (MassTree, [(String, MassBodyId); 2]) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());

    // attach1_optionA
    let offset = DVec3::new(0.0, 1.5, 0.0);
    let t_parent_child = DMat3::from_cols(
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, -1.0),
        DVec3::new(0.0, 1.0, 0.0),
    );
    tree.attach(child1, parent, offset, t_parent_child);

    (tree, [("Parent".into(), parent), ("Child1".into(), child1)])
}

/// RUN_02: parent + child1 at [0, 1, 0] + child2 at [0, -1, 0], identity rotations.
fn build_run_02() -> (MassTree, [(String, MassBodyId); 3]) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());
    let child2 = tree.add_body("Child2".into(), child2_default());

    tree.attach(child1, parent, DVec3::new(0.0, 1.0, 0.0), DMat3::IDENTITY);
    tree.attach(child2, parent, DVec3::new(0.0, -1.0, 0.0), DMat3::IDENTITY);

    (
        tree,
        [
            ("Parent".into(), parent),
            ("Child1".into(), child1),
            ("Child2".into(), child2),
        ],
    )
}

/// RUN_03: child1 to parent at [1, 0, 0]; child2 to child1 at [1, 0, 0].
fn build_run_03() -> (MassTree, [(String, MassBodyId); 3]) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());
    let child2 = tree.add_body("Child2".into(), child2_default());

    tree.attach(child1, parent, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);
    tree.attach(child2, child1, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

    (
        tree,
        [
            ("Parent".into(), parent),
            ("Child1".into(), child1),
            ("Child2".into(), child2),
        ],
    )
}

/// RUN_04: child1 to parent at [0, 0, -2]; child2 to child1 at [0, 0, -2].
fn build_run_04() -> (MassTree, [(String, MassBodyId); 3]) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());
    let child2 = tree.add_body("Child2".into(), child2_default());

    tree.attach(child1, parent, DVec3::new(0.0, 0.0, -2.0), DMat3::IDENTITY);
    tree.attach(child2, child1, DVec3::new(0.0, 0.0, -2.0), DMat3::IDENTITY);

    (
        tree,
        [
            ("Parent".into(), parent),
            ("Child1".into(), child1),
            ("Child2".into(), child2),
        ],
    )
}

/// RUN_10: parent (StructCG spec option B), child1 (Body spec option C),
/// child2 (Struct spec option B), child3 (Body spec default). Attach all
/// three to parent at init, then runtime-detach child2 at t=1s, print tree
/// at shutdown (t=2s). Final expected tree: parent + child1 + child3.
fn build_run_10() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();

    // parent: default mass, then inertia_spec becomes StructCG (same values
    // since T_parent_this = I → StructCG transform is a no-op).
    let parent_inertia_structcg = DMat3::from_cols(
        DVec3::new(0.33333333, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.0833333333),
    );
    let parent_mass = mass_struct_cg_spec(1.0, DVec3::ZERO, parent_inertia_structcg);
    let parent = tree.add_root("Parent".into(), parent_mass);

    // child1: inertia_spec = Body (optionC)
    let child1_inertia = DMat3::from_cols(
        DVec3::new(0.33333333, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.0833333333),
    );
    let child1_mass = mass_body_spec(1.0, DVec3::ZERO, child1_inertia);
    let child1 = tree.add_body("Child1".into(), child1_mass);

    // child2: inertia_spec = Struct (optionB) — inertia about structural
    // origin. Position is [0,0,0] so parallel-axis shift is zero.
    let child2_inertia_struct = DMat3::from_cols(
        DVec3::new(1.33333333, 0.0, -0.5),
        DVec3::new(0.0, 1.66666667, 0.0),
        DVec3::new(-0.5, 0.0, 0.33333333),
    );
    let child2_mass = mass_struct_spec(1.0, DVec3::ZERO, child2_inertia_struct);
    let child2 = tree.add_body("Child2".into(), child2_mass);

    // child3: default Body spec
    let child3_mass = child3_default();
    let child3 = tree.add_body("Child3".into(), child3_mass);

    // attach1: child1 → parent at [-1, 0, 0]
    tree.attach(child1, parent, DVec3::new(-1.0, 0.0, 0.0), DMat3::IDENTITY);
    // attach2: child2 → parent at [2.5, 0, 1] with Euler Yaw_Pitch_Roll [0, 180°, 0]
    // Yaw_Pitch_Roll = ZYX sequence in jeod_math.
    let euler_deg: [f64; 3] = [0.0, 180.0, 0.0];
    let angles = [
        euler_deg[0].to_radians(),
        euler_deg[1].to_radians(),
        euler_deg[2].to_radians(),
    ];
    let t_parent_child2 = compute_matrix_from_euler_angles(angles, EulerSequence::ZYX);
    tree.attach(child2, parent, DVec3::new(2.5, 0.0, 1.0), t_parent_child2);
    // attach3: child3 → parent at [1, 0, 0]
    tree.attach(child3, parent, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

    // Runtime detach at t=1: remove child2.
    tree.detach(child2);

    // print_tree starts from root (parent) and recursively prints children
    // — child2 is detached so it is not in the reference file. Ordering:
    // parent, child1, child3 (JEOD iterates children in insertion order,
    // skipping the detached one).
    let ids = vec![
        ("Parent".to_string(), parent),
        ("Child1".to_string(), child1),
        ("Child3".to_string(), child3),
    ];
    (tree, ids)
}

/// RUN_11: parent (StructCG), child1 (Body optionC), child2 (Struct optionB
/// with non-zero structural position [0.5, 0, 1]), child3 (not attached).
/// Attach1: child1 → parent at [1, 0, 0].
/// Attach2: child2 → parent at [1.5, 0, -1].
/// Reattach at t=1: child2 to new offset [1.5, 0, -2] with Euler YPR [0, -90°, 0].
fn build_run_11() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();

    let parent_inertia = DMat3::from_cols(
        DVec3::new(0.33333333, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.0833333333),
    );
    let parent_mass = mass_struct_cg_spec(1.0, DVec3::ZERO, parent_inertia);
    let parent = tree.add_root("Parent".into(), parent_mass);

    let child1_inertia = DMat3::from_cols(
        DVec3::new(0.33333333, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.0833333333),
    );
    let child1_mass = mass_body_spec(1.0, DVec3::ZERO, child1_inertia);
    let child1 = tree.add_body("Child1".into(), child1_mass);

    // child2: Struct spec, position = [0.5, 0, 1.0] — parallel axis shift is
    // non-zero.
    let child2_inertia_struct = DMat3::from_cols(
        DVec3::new(1.33333333, 0.0, -0.5),
        DVec3::new(0.0, 1.66666667, 0.0),
        DVec3::new(-0.5, 0.0, 0.33333333),
    );
    let child2_mass = mass_struct_spec(1.0, DVec3::new(0.5, 0.0, 1.0), child2_inertia_struct);
    let child2 = tree.add_body("Child2".into(), child2_mass);

    // Initial attachments.
    tree.attach(child1, parent, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);
    tree.attach(child2, parent, DVec3::new(1.5, 0.0, -1.0), DMat3::IDENTITY);

    // Reattach child2 at t=1s: new offset + rotation. `BodyReattach` does
    // not walk the subtree — it simply updates the structure_point and
    // triggers a composite recompute via the parent's mass tree. We model
    // this as detach + attach with the new pose.
    tree.detach(child2);
    let angles = [
        0.0_f64.to_radians(),
        (-90.0_f64).to_radians(),
        0.0_f64.to_radians(),
    ];
    let t_reattach = compute_matrix_from_euler_angles(angles, EulerSequence::ZYX);
    tree.attach(child2, parent, DVec3::new(1.5, 0.0, -2.0), t_reattach);

    let ids = vec![
        ("Parent".to_string(), parent),
        ("Child1".to_string(), child1),
        ("Child2".to_string(), child2),
    ];
    (tree, ids)
}

/// RUN_101: parent + child1 attached via named mass points
/// (parent_mass_points_1 + child1_mass_points_1A). Attach subject point
/// "right_to_top" to parent point "top_to_right".
fn build_run_101() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());

    // parent_mass_points_1: "top_to_right" at [0, 0.5, 0],
    //   T = [[0,1,0],[0,0,1],[1,0,0]]
    let t_parent_pt = DMat3::from_cols(
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
    );
    tree.add_mass_point(
        parent,
        "top_to_right",
        DVec3::new(0.0, 0.5, 0.0),
        t_parent_pt,
    );

    // child1_mass_points_1A: "right_to_top" at [0, 0, 1],
    //   T = [[0,0,1],[0,-1,0],[1,0,0]]
    let t_child_pt = DMat3::from_cols(
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
    );
    tree.add_mass_point(
        child1,
        "right_to_top",
        DVec3::new(0.0, 0.0, 1.0),
        t_child_pt,
    );

    tree.attach_aligned(child1, "right_to_top", parent, "top_to_right");

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
        ],
    )
}

/// RUN_102: parent + child1 + child2; child1 attached to parent, child2
/// attached to parent (NOT to child1; from `pt_attach2_default` which sets
/// parent_body = parent). Uses Euler orientations for mass points.
fn build_run_102() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());
    let child2 = tree.add_body("Child2".into(), child2_default());

    // parent_mass_points_2: "bottom_to_top" at [0, -0.5, 0] with YPR [-90°, 0, 0];
    //                      "top_to_bottom" at [0, 0.5, 0]  with YPR [ 90°, 0, 0].
    let ypr_neg90 =
        compute_matrix_from_euler_angles([(-90.0_f64).to_radians(), 0.0, 0.0], EulerSequence::ZYX);
    let ypr_pos90 =
        compute_matrix_from_euler_angles([(90.0_f64).to_radians(), 0.0, 0.0], EulerSequence::ZYX);
    tree.add_mass_point(
        parent,
        "bottom_to_top",
        DVec3::new(0.0, -0.5, 0.0),
        ypr_neg90,
    );
    tree.add_mass_point(
        parent,
        "top_to_bottom",
        DVec3::new(0.0, 0.5, 0.0),
        ypr_pos90,
    );

    // child1_mass_points_1B: "bottom_to_top" at [0, -0.5, 0], YPR [-90°, 0, 0].
    tree.add_mass_point(
        child1,
        "bottom_to_top",
        DVec3::new(0.0, -0.5, 0.0),
        ypr_neg90,
    );

    // child2_mass_points_1B: "top_to_bottom" at [0, 0.5, 0], YPR [90°, 0, 0].
    tree.add_mass_point(
        child2,
        "top_to_bottom",
        DVec3::new(0.0, 0.5, 0.0),
        ypr_pos90,
    );

    // pt_attach1: child1.bottom_to_top → parent.top_to_bottom
    tree.attach_aligned(child1, "bottom_to_top", parent, "top_to_bottom");
    // pt_attach2: child2.top_to_bottom → parent.bottom_to_top
    tree.attach_aligned(child2, "top_to_bottom", parent, "bottom_to_top");

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child2".to_string(), child2),
        ],
    )
}

// ════════════════════════════════════════════════════════════════════
// Shared validation driver
// ════════════════════════════════════════════════════════════════════

/// Validate every body in the reference file against the constructed tree
/// and accumulate max errors. JEOD writes the tree starting from the root
/// and recurses; body name lookup is by string.
fn validate_run<S: AsRef<str>>(
    run_label: &str,
    tree: &MassTree,
    ids: &[(S, MassBodyId)],
    reference: &PrintedTree,
    max_errors: &mut MaxErrors,
) {
    assert!(
        !reference.bodies.is_empty(),
        "[{run_label}] reference file has no bodies"
    );
    for ref_body in &reference.bodies {
        let id = ids
            .iter()
            .find(|(n, _)| n.as_ref() == ref_body.name)
            .map(|(_, id)| *id)
            .unwrap_or_else(|| {
                panic!(
                    "[{run_label}] reference body '{}' not found in constructed tree",
                    ref_body.name
                )
            });
        check_body(
            run_label,
            &ref_body.name,
            tree,
            id,
            ref_body,
            TOL_MASS,
            TOL_COM,
            TOL_INERTIA,
            max_errors,
        );
    }
}

// ════════════════════════════════════════════════════════════════════
// Tier 3 test: iterates every run, asserts tolerances, emits report
// ════════════════════════════════════════════════════════════════════

// non-recipe: SIM_verif_attach_mass uses 1 kg JEOD test-fixture bodies
// (parent_default/child{1,2,3}_default) ported from
// `Modified_data/*.py`; not Apollo-class. The whole test runs against
// `MassTree::print_tree` parity, not the full simulation pipeline.
#[test]
fn tier3_sim_attach_mass() {
    let mut errors = MaxErrors::new();

    {
        let (tree, ids) = build_run_01();
        let reference = load_reference("attach_mass_01_mass.out");
        validate_run("RUN_01", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_02();
        let reference = load_reference("attach_mass_02_mass.out");
        validate_run("RUN_02", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_03();
        let reference = load_reference("attach_mass_03_mass.out");
        validate_run("RUN_03", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_04();
        let reference = load_reference("attach_mass_04_mass.out");
        validate_run("RUN_04", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_10();
        let reference = load_reference("attach_mass_10_mass.out");
        validate_run("RUN_10", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_11();
        let reference = load_reference("attach_mass_11_mass.out");
        validate_run("RUN_11", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_101();
        let reference = load_reference("attach_mass_101_mass.out");
        validate_run("RUN_101", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_102();
        let reference = load_reference("attach_mass_102_mass.out");
        validate_run("RUN_102", &tree, &ids, &reference, &mut errors);
    }

    let mut report = CrossvalReport::compute("tier3_sim_attach_mass", &[], &[]);
    report.add_extra("composite_mass", errors.mass, "kg");
    report.add_extra("composite_com", errors.com, "m");
    report.add_extra("composite_inertia", errors.inertia, "kg*m^2");
    report.write();
}
