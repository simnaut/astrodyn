//! Tier 3: SIM_verif_attach_mass — mass tree attach/detach cross-validation.
//!
//! Reproduces representative runs from JEOD's `SIM_verif_attach_mass`
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
//! - RUN_05: single parent, `Struct` inertia spec, non-zero CoM (no attach)
//! - RUN_06: parent (`StructCG`) + child1 (`Spec`), offset attach [0,0,2]
//! - RUN_07: parent (`StructCG`) + child1 (`SpecCG`), offset attach [-1,0,0]
//! - RUN_10: runtime detach — parent gains 3 children then detaches child2
//! - RUN_11: runtime reattach — child2 moves to new offset + rotation
//! - RUN_101: `BodyAttachAligned` via named mass points (simple)
//! - RUN_102: `BodyAttachAligned` chained across three bodies
//! - RUN_103: three bodies chained via named points (inline Euler spec)
//! - RUN_104: three bodies chained via named points (`InputMatrix` spec)
//! - RUN_106: parent (`StructCG`) + child1 (`Spec`) via named points
//! - RUN_107: parent (`StructCG`) + child1 (`SpecCG`) via named points
//! - RUN_110: named-point attach of 3 children then runtime detach of child2
//! - RUN_111: named-point + offset attach, runtime reattach of child2
//! - RUN_09: non-identity parent struct→body orientation. JEOD reports the
//!   composite inertia (`C.M.P. Ib`) in the parent **body** frame while the
//!   composite CoM stays in the struct frame; `recompute_composites` produces
//!   the composite in the body frame (the parent's `StructCG` inertia is rotated
//!   struct→body at init), so both compare directly against `mass.out`.
//! - RUN_109: named-point attach (RUN_107 layout) combined with the RUN_09
//!   non-identity parent struct→body orientation; child1 uses the `Body`
//!   option-C inertia. The named-point geometry resolves to the same
//!   parent←child1 edge as RUN_09 (struct origin at [-1,0,0], identity
//!   struct→struct), so the composite matches RUN_09's `mass.out`.
//! - RUN_08: a child (child3) is attached to child2, then a later body action
//!   attaches child3 to the parent. Because child3 is no longer a root, JEOD's
//!   `MassBody::attach_to` (`mass_attach.cc:181-223`) re-roots child3's existing
//!   tree: it finds child3's root (child1), recasts the requested child3→parent
//!   offset/rotation into the child1→parent frame, and attaches child1. The
//!   final topology is the chain parent←child1←child2←child3. Modeled with
//!   [`MassTree::attach_with_reroot`].
//! - RUN_108: the same final chain as RUN_08, but every attach is via named
//!   mass points. The re-root path is exercised through the named-point
//!   geometry (computed inline) fed into `attach_with_reroot`.
//!
//! Supplements the analytical tests in
//! `crates/astrodyn_dynamics/tests/tier3_mass_attach_detach.rs` with direct
//! JEOD cross-validation.

use astrodyn::{compute_matrix_from_euler_angles_typed, EulerSequence};
use astrodyn::{MassBodyId, MassProperties, MassTree};
use glam::{DMat3, DVec3};
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
use astrodyn_verif_jeod::apollo_mass_tree::{parse_print_tree, PrintedBody, PrintedTree};
use astrodyn_verif_jeod::crossval::CrossvalReport;

// ════════════════════════════════════════════════════════════════════
// Test data path / loading
// ════════════════════════════════════════════════════════════════════

fn test_data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

fn load_reference(filename: &str) -> PrintedTree {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "JEOD reference data not found at {}.\n\
         Generate with:\n\
         docker run --rm \\\n\
           -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \\\n\
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

/// Build `MassProperties` using JEOD's `SpecCG` inertia spec: the inertia
/// tensor is given in a *user-specified* frame (`inertia_orientation`) about
/// the body CoM. Port of the `SpecCG` branch in
/// `mass_properties_init.cc:125-135`: `I_body = T_io · I_user · T_ioᵀ`, where
/// `T_io = inertia_orientation.trans` (struct→user transform) and
/// `transform_matrix(T, A) = T · A · Tᵀ`.
fn mass_spec_cg_spec(
    mass: f64,
    position: DVec3,
    inertia_orientation: DMat3,
    inertia_user: DMat3,
) -> MassProperties {
    let body_inertia = inertia_orientation * inertia_user * inertia_orientation.transpose();
    MassProperties::with_inertia(mass, body_inertia, position)
}

/// Build `MassProperties` using JEOD's `Spec` inertia spec: the inertia tensor
/// is given in a user-specified frame (`inertia_orientation`) about a
/// user-specified origin (`inertia_offset`). Port of the `Spec` branch in
/// `mass_properties_init.cc:141-158`: shift the tensor from the user origin to
/// the CoM via the parallel-axis theorem (the offset point mass is computed in
/// the *user* frame), then rotate to body: `I_body = T_io · (I_user −
/// pmi(m, inertia_offset)) · T_ioᵀ`.
fn mass_spec_spec(
    mass: f64,
    position: DVec3,
    inertia_orientation: DMat3,
    inertia_offset: DVec3,
    inertia_user: DMat3,
) -> MassProperties {
    let offset_inertia = point_mass_inertia(mass, inertia_offset);
    let shifted = inertia_user - offset_inertia;
    let body_inertia = inertia_orientation * shifted * inertia_orientation.transpose();
    MassProperties::with_inertia(mass, body_inertia, position)
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

/// `parent_mass_inertia_optionB()` / `child*_mass_inertia_optionC()`: the
/// canonical box inertia diag(0.3333, 0.4167, 0.0833) about the CoM, in body
/// (= struct) axes. `optionB` declares it via the `StructCG` spec; `optionC`
/// declares the identical tensor directly via the `Body` spec. With an
/// identity struct→body transform both produce this same body-frame tensor.
fn box_inertia_diag() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(0.33333333, 0.0, 0.0),
        DVec3::new(0.0, 0.41666667, 0.0),
        DVec3::new(0.0, 0.0, 0.0833333333),
    )
}

/// `child1_mass_inertia_optionA()` inputs (the `Spec` declaration). The inertia
/// tensor, offset point, and orientation matrix are taken verbatim from
/// `Modified_data/child1_mass.py`.
fn child1_spec_option_a() -> (DMat3, DVec3, DMat3) {
    let inertia_user = DMat3::from_cols(
        DVec3::new(1.4166667, 0.144338, -0.433013),
        DVec3::new(0.144338, 1.58333, 0.25),
        DVec3::new(-0.433013, 0.25, 0.333333),
    );
    let inertia_offset = DVec3::new(0.433013, -0.25, 1.0);
    // inertia_orientation.trans (row-major in JEOD) → DMat3 columns.
    let inertia_orientation = DMat3::from_cols(
        DVec3::new(0.8660254, 0.5, 0.0),
        DVec3::new(-0.5, 0.8660254, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    (inertia_user, inertia_offset, inertia_orientation)
}

/// `child1_mass_inertia_optionB()` inputs (the `SpecCG` declaration). The
/// orientation is *not* reset by `optionB`, so a run that calls `optionA`
/// then `optionB` (RUN_07, RUN_107) keeps the `optionA` orientation matrix.
fn child1_spec_option_b_inertia() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(0.35416666666667, 0.03608439182435, 0.0),
        DVec3::new(0.03608439182435, 0.39583333333333, 0.0),
        DVec3::new(0.0, 0.0, 0.08333333333333),
    )
}

/// `child1` built per `child1_mass_inertia_optionA` (Spec spec).
fn child1_spec_a() -> MassProperties {
    let (inertia_user, inertia_offset, inertia_orientation) = child1_spec_option_a();
    mass_spec_spec(
        1.0,
        DVec3::ZERO,
        inertia_orientation,
        inertia_offset,
        inertia_user,
    )
}

/// `child1` built per `child1_mass_inertia_optionA` then `optionB` (SpecCG
/// spec, keeping the optionA orientation matrix).
fn child1_spec_b() -> MassProperties {
    let (_, _, inertia_orientation) = child1_spec_option_a();
    mass_spec_cg_spec(
        1.0,
        DVec3::ZERO,
        inertia_orientation,
        child1_spec_option_b_inertia(),
    )
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
    let comp = &tree.get(id).composite_properties;
    // `recompute_composites` stores the composite inertia in the body frame
    // (JEOD's `C.M.P. Ib tensor` convention) and the CoM in the struct frame
    // (`C.M.P. CM vector`), so both compare directly against `mass.out`.
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

/// RUN_05: single parent body, `Struct` inertia spec (option A) about the
/// structural origin, CoM offset to [0.5, 0, 1]. No attachments — the
/// composite equals the core. Exercises the parallel-axis shift in
/// `mass_struct_spec` with a non-zero CoM position.
fn build_run_05() -> (MassTree, [(String, MassBodyId); 1]) {
    let mut tree = MassTree::new();
    // parent_mass_inertia_optionA: Struct spec about structural origin.
    let inertia_struct = DMat3::from_cols(
        DVec3::new(1.33333333, 0.0, -0.5),
        DVec3::new(0.0, 1.66666667, 0.0),
        DVec3::new(-0.5, 0.0, 0.33333333),
    );
    let parent_mass = mass_struct_spec(1.0, DVec3::new(0.5, 0.0, 1.0), inertia_struct);
    let parent = tree.add_root("Parent".into(), parent_mass);
    (tree, [("Parent".into(), parent)])
}

/// RUN_06: parent (`StructCG` option B) + child1 (`Spec` option A) attached at
/// offset [0, 0, 2] with identity orientation (the `attach1_default`
/// quaternion is left at identity; only the offset is overridden).
fn build_run_06() -> (MassTree, [(String, MassBodyId); 2]) {
    let mut tree = MassTree::new();
    let parent_mass = mass_struct_cg_spec(1.0, DVec3::ZERO, box_inertia_diag());
    let parent = tree.add_root("Parent".into(), parent_mass);
    let child1 = tree.add_body("Child1".into(), child1_spec_a());

    tree.attach(child1, parent, DVec3::new(0.0, 0.0, 2.0), DMat3::IDENTITY);

    (tree, [("Parent".into(), parent), ("Child1".into(), child1)])
}

/// RUN_07: parent (`StructCG` option B) + child1 (`SpecCG` option B, retaining
/// the option-A orientation) attached at offset [-1, 0, 0], identity
/// orientation.
fn build_run_07() -> (MassTree, [(String, MassBodyId); 2]) {
    let mut tree = MassTree::new();
    let parent_mass = mass_struct_cg_spec(1.0, DVec3::ZERO, box_inertia_diag());
    let parent = tree.add_root("Parent".into(), parent_mass);
    let child1 = tree.add_body("Child1".into(), child1_spec_b());

    tree.attach(child1, parent, DVec3::new(-1.0, 0.0, 0.0), DMat3::IDENTITY);

    (tree, [("Parent".into(), parent), ("Child1".into(), child1)])
}

/// Parent struct→body orientation from `parent_mass_orientation_optionA`
/// (`Modified_data/parent_mass.py`), JEOD `T_parent_this` (row-major in the
/// `.py`, transposed into glam column-major here). Used by RUN_09 both as the
/// parent's [`MassProperties::t_parent_this`] and to rotate its `StructCG`
/// inertia struct→body; JEOD reports the root's composite in this body frame.
#[allow(
    clippy::approx_constant,
    reason = "0.70710678118655 is the verbatim optionA matrix value from JEOD's \
              Modified_data/parent_mass.py; substituting FRAC_1_SQRT_2 would diverge \
              from the transcribed JEOD source matrix"
)]
fn parent_option_a() -> DMat3 {
    // Row-major JEOD matrix → DMat3::from_cols takes columns.
    DMat3::from_cols(
        DVec3::new(0.8660254, 0.35355339059327, -0.35355339059327),
        DVec3::new(-0.5, 0.61237243569579, -0.61237243569579),
        DVec3::new(0.0, 0.70710678118655, 0.70710678118655),
    )
}

/// RUN_09: parent (`StructCG` option B inertia + **non-identity struct→body
/// orientation** `optionA`) + child1 (`Body` option C) attached at offset
/// [-1, 0, 0], identity attach. First run where the **root body's own**
/// struct→body transform is non-identity, so it is the case that distinguishes
/// the struct and body frames in `recompute_composites`.
///
/// The `StructCG` spec gives the inertia in struct axes; JEOD's init
/// (`mass_properties_init.cc:103`) rotates it struct→body via `T_parent_this`,
/// so the stored **core** inertia is `T · box_diag · Tᵀ` (= JEOD's `M.P. Ib`
/// for the oriented parent). `recompute_composites` then accumulates the
/// composite in the parent body frame, so `composite_properties.inertia`
/// matches JEOD's `C.M.P. Ib tensor` directly (the CoM stays struct-frame,
/// matching `C.M.P. CM vector`).
fn build_run_09() -> (MassTree, [(String, MassBodyId); 2]) {
    let mut tree = MassTree::new();
    // StructCG init: rotate the struct-axes option-B inertia into the body
    // frame via the parent's struct→body transform (JEOD mass_properties_init.cc).
    let t = parent_option_a();
    let parent_core_body = t * box_inertia_diag() * t.transpose();
    let parent = tree.add_root(
        "Parent".into(),
        MassProperties::with_inertia(1.0, parent_core_body, DVec3::ZERO).with_t_parent_this(t),
    );
    let child1 = tree.add_body(
        "Child1".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    tree.attach(child1, parent, DVec3::new(-1.0, 0.0, 0.0), DMat3::IDENTITY);
    (tree, [("Parent".into(), parent), ("Child1".into(), child1)])
}

/// RUN_109: named-point attach combined with the RUN_09 non-identity parent
/// struct→body orientation. Same named-point layout as RUN_107 (parent
/// "front_to_back" at [-0.5,0,0], child1 "back_to_front" at [0.5,0,0]), but the
/// parent carries `parent_mass_orientation_optionA` (non-identity struct→body)
/// on top of its `StructCG` option-B inertia, and child1 uses the `Body`
/// option-C inertia. The docking-aligned named-point attach resolves to the
/// same parent←child1 edge as RUN_09 (child1 struct origin at parent-struct
/// [-1,0,0], identity struct→struct), so the composite matches RUN_09's
/// `mass.out` exactly. Distinguishes the named-point path from the non-identity
/// root-orientation handling.
fn build_run_109() -> (MassTree, [(String, MassBodyId); 2]) {
    let mut tree = MassTree::new();
    // Parent: StructCG option-B inertia rotated struct→body via optionA, plus
    // the optionA struct→body transform itself (identical to RUN_09's parent).
    let t = parent_option_a();
    let parent_core_body = t * box_inertia_diag() * t.transpose();
    let parent = tree.add_root(
        "Parent".into(),
        MassProperties::with_inertia(1.0, parent_core_body, DVec3::ZERO).with_t_parent_this(t),
    );
    // Child1: Body option-C inertia, identity struct→body.
    let child1 = tree.add_body(
        "Child1".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );

    // parent_mass_points_1 renamed in input.py to "front_to_back" at [-0.5,0,0]
    // with T=[[-1,0,0],[0,-1,0],[0,0,1]].
    tree.add_mass_point(
        parent,
        "front_to_back",
        DVec3::new(-0.5, 0.0, 0.0),
        jeod_trans([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    // child1_mass_points_1A renamed to "back_to_front" at [0.5,0,0], identity.
    tree.add_mass_point(
        child1,
        "back_to_front",
        DVec3::new(0.5, 0.0, 0.0),
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );

    tree.attach_aligned(child1, "back_to_front", parent, "front_to_back");

    (tree, [("Parent".into(), parent), ("Child1".into(), child1)])
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
    // Yaw_Pitch_Roll = ZYX sequence in astrodyn_math.
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

/// Convert a JEOD row-major 3×3 orientation matrix (`trans[i][j]`) into a
/// `DMat3`. JEOD's `trans` rows are listed `[[row0], [row1], [row2]]`; the
/// `DMat3` columns are `column_j = [trans[0][j], trans[1][j], trans[2][j]]`
/// (the same convention the existing RUN_101/102 builders apply by hand).
fn jeod_trans(rows: [[f64; 3]; 3]) -> DMat3 {
    DMat3::from_cols(
        DVec3::new(rows[0][0], rows[1][0], rows[2][0]),
        DVec3::new(rows[0][1], rows[1][1], rows[2][1]),
        DVec3::new(rows[0][2], rows[1][2], rows[2][2]),
    )
}

/// RUN_103: three bodies (all default `Body` inertia) chained via named mass
/// points — child1 → parent and child2 → child1. Mass-point orientations are
/// given inline in the input file via `Yaw_Pitch_Roll` Euler angles (= ZYX).
fn build_run_103() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());
    let child2 = tree.add_body("Child2".into(), child2_default());

    let ypr_0 = compute_matrix_from_euler_angles([0.0, 0.0, 0.0], EulerSequence::ZYX);
    let ypr_180 =
        compute_matrix_from_euler_angles([180.0_f64.to_radians(), 0.0, 0.0], EulerSequence::ZYX);

    // parent: "back_to_front" at [0.5, 0, 0], YPR [0,0,0].
    tree.add_mass_point(parent, "back_to_front", DVec3::new(0.5, 0.0, 0.0), ypr_0);
    // child1: "front_to_back" at [-0.5,0,0] YPR[180,0,0]; "back_to_front" at [0.5,0,0] YPR[0,0,0].
    tree.add_mass_point(child1, "front_to_back", DVec3::new(-0.5, 0.0, 0.0), ypr_180);
    tree.add_mass_point(child1, "back_to_front", DVec3::new(0.5, 0.0, 0.0), ypr_0);
    // child2: "front_to_back" at [-0.5,0,0] YPR[180,0,0].
    tree.add_mass_point(child2, "front_to_back", DVec3::new(-0.5, 0.0, 0.0), ypr_180);

    tree.attach_aligned(child1, "front_to_back", parent, "back_to_front");
    tree.attach_aligned(child2, "front_to_back", child1, "back_to_front");

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child2".to_string(), child2),
        ],
    )
}

/// RUN_104: three default-inertia bodies chained via named points using
/// `InputMatrix` orientations — child1 → parent and child2 → child1.
fn build_run_104() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root("Parent".into(), parent_default());
    let child1 = tree.add_body("Child1".into(), child1_default());
    let child2 = tree.add_body("Child2".into(), child2_default());

    let t_left_right = jeod_trans([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
    let t_right_left = jeod_trans([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]);

    // parent: point0 renamed "left_to_right" at [0,0,-1].
    tree.add_mass_point(
        parent,
        "left_to_right",
        DVec3::new(0.0, 0.0, -1.0),
        t_left_right,
    );
    // child1_mass_points_2: "right_to_left" at [0,0,1]; "left_to_right" at [0,0,-1].
    tree.add_mass_point(
        child1,
        "right_to_left",
        DVec3::new(0.0, 0.0, 1.0),
        t_right_left,
    );
    tree.add_mass_point(
        child1,
        "left_to_right",
        DVec3::new(0.0, 0.0, -1.0),
        t_left_right,
    );
    // child2_mass_points_1A: "right_to_left" at [0,0,1].
    tree.add_mass_point(
        child2,
        "right_to_left",
        DVec3::new(0.0, 0.0, 1.0),
        t_right_left,
    );

    tree.attach_aligned(child1, "right_to_left", parent, "left_to_right");
    tree.attach_aligned(child2, "right_to_left", child1, "left_to_right");

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child2".to_string(), child2),
        ],
    )
}

/// RUN_106: parent (`StructCG` option B) + child1 (`Spec` option A) attached
/// via named points using `InputMatrix` orientations.
fn build_run_106() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root(
        "Parent".into(),
        mass_struct_cg_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child1 = tree.add_body("Child1".into(), child1_spec_a());

    // parent: point0 renamed "right_to_left" at [0,0,1].
    let t_right_left = jeod_trans([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]);
    tree.add_mass_point(
        parent,
        "right_to_left",
        DVec3::new(0.0, 0.0, 1.0),
        t_right_left,
    );
    // child1: point0 renamed "left_to_right" at [0,0,-1].
    let t_left_right = jeod_trans([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
    tree.add_mass_point(
        child1,
        "left_to_right",
        DVec3::new(0.0, 0.0, -1.0),
        t_left_right,
    );

    tree.attach_aligned(child1, "left_to_right", parent, "right_to_left");

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
        ],
    )
}

/// RUN_107: parent (`StructCG` option B) + child1 (`SpecCG` option B, keeping
/// the option-A orientation) attached via named points.
fn build_run_107() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root(
        "Parent".into(),
        mass_struct_cg_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child1 = tree.add_body("Child1".into(), child1_spec_b());

    // parent: point0 renamed "front_to_back" at [-0.5,0,0].
    let t_front_back = jeod_trans([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]);
    tree.add_mass_point(
        parent,
        "front_to_back",
        DVec3::new(-0.5, 0.0, 0.0),
        t_front_back,
    );
    // child1: point0 renamed "back_to_front" at [0.5,0,0], identity orientation.
    let t_back_front = jeod_trans([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    tree.add_mass_point(
        child1,
        "back_to_front",
        DVec3::new(0.5, 0.0, 0.0),
        t_back_front,
    );

    tree.attach_aligned(child1, "back_to_front", parent, "front_to_back");

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
        ],
    )
}

/// RUN_110: parent (`StructCG`) with three named points, child1 (`Body` option
/// C), child2 (`Struct` option B, CoM at [0.5,0,1]) and child3 (default) each
/// attached to the parent via named points; then child2 is runtime-detached at
/// t=1s. Print at shutdown (t=2s) → parent + child1 + child3.
fn build_run_110() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root(
        "Parent".into(),
        mass_struct_cg_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child1 = tree.add_body(
        "Child1".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );

    // child2: Struct spec about structural origin, CoM at [0.5, 0, 1].
    let child2_inertia_struct = DMat3::from_cols(
        DVec3::new(1.33333333, 0.0, -0.5),
        DVec3::new(0.0, 1.66666667, 0.0),
        DVec3::new(-0.5, 0.0, 0.33333333),
    );
    let child2 = tree.add_body(
        "Child2".into(),
        mass_struct_spec(1.0, DVec3::new(0.5, 0.0, 1.0), child2_inertia_struct),
    );
    let child3 = tree.add_body("Child3".into(), child3_default());

    // parent_mass_points_3:
    //   "front_to_back" at [-0.5,0,0] T=[[-1,0,0],[0,-1,0],[0,0,1]]
    //   "back_to_front" at [ 0.5,0,0] T=identity
    //   "parent_to_child2" at [2.5,0,1] T=[[1,0,0],[0,0,-1],[0,1,0]]
    tree.add_mass_point(
        parent,
        "front_to_back",
        DVec3::new(-0.5, 0.0, 0.0),
        jeod_trans([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    tree.add_mass_point(
        parent,
        "back_to_front",
        DVec3::new(0.5, 0.0, 0.0),
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    tree.add_mass_point(
        parent,
        "parent_to_child2",
        DVec3::new(2.5, 0.0, 1.0),
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    );

    // child1: point0 "back_to_front" at [0.5,0,0] identity.
    tree.add_mass_point(
        child1,
        "back_to_front",
        DVec3::new(0.5, 0.0, 0.0),
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    // child2: point0 "child2_to_parent" at [0,0,0] T=[[1,0,0],[0,0,-1],[0,1,0]].
    tree.add_mass_point(
        child2,
        "child2_to_parent",
        DVec3::ZERO,
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    );
    // child3_mass_points_1: "front_to_back" at [-0.5,0,0] T=[[-1,0,0],[0,-1,0],[0,0,1]].
    tree.add_mass_point(
        child3,
        "front_to_back",
        DVec3::new(-0.5, 0.0, 0.0),
        jeod_trans([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
    );

    tree.attach_aligned(child1, "back_to_front", parent, "front_to_back");
    tree.attach_aligned(child2, "child2_to_parent", parent, "parent_to_child2");
    tree.attach_aligned(child3, "front_to_back", parent, "back_to_front");

    // Runtime detach child2 at t=1s; tree printed at shutdown.
    tree.detach(child2);

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child3".to_string(), child3),
        ],
    )
}

/// RUN_111: parent (`StructCG`) + child1 (`Body` option C) attached via named
/// points; child2 (`Struct` option B, CoM at [0.5,0,1]) offset-attached to the
/// parent at [1.5,0,-1] (identity), then runtime-reattached at t=1s to a new
/// offset [1.5,0,-2] with `Yaw_Pitch_Roll` [0,-90°,0].
fn build_run_111() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root(
        "Parent".into(),
        mass_struct_cg_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child1 = tree.add_body(
        "Child1".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );

    let child2_inertia_struct = DMat3::from_cols(
        DVec3::new(1.33333333, 0.0, -0.5),
        DVec3::new(0.0, 1.66666667, 0.0),
        DVec3::new(-0.5, 0.0, 0.33333333),
    );
    let child2 = tree.add_body(
        "Child2".into(),
        mass_struct_spec(1.0, DVec3::new(0.5, 0.0, 1.0), child2_inertia_struct),
    );

    // parent: point0 "back_to_front" at [0.5,0,0], identity.
    tree.add_mass_point(
        parent,
        "back_to_front",
        DVec3::new(0.5, 0.0, 0.0),
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    // child1: point0 "front_to_back" at [-0.5,0,0] T=[[-1,0,0],[0,-1,0],[0,0,1]].
    tree.add_mass_point(
        child1,
        "front_to_back",
        DVec3::new(-0.5, 0.0, 0.0),
        jeod_trans([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
    );

    // pt_attach1: child1.front_to_back → parent.back_to_front.
    tree.attach_aligned(child1, "front_to_back", parent, "back_to_front");
    // attach2: child2 → parent at offset [1.5,0,-1] identity.
    tree.attach(child2, parent, DVec3::new(1.5, 0.0, -1.0), DMat3::IDENTITY);

    // Reattach child2 at t=1s: offset [1.5,0,-2], YPR [0,-90°,0].
    tree.detach(child2);
    let t_reattach =
        compute_matrix_from_euler_angles([0.0, (-90.0_f64).to_radians(), 0.0], EulerSequence::ZYX);
    tree.attach(child2, parent, DVec3::new(1.5, 0.0, -2.0), t_reattach);

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child2".to_string(), child2),
        ],
    )
}

/// RUN_08: chained attaches where the third action targets an already-attached
/// (non-root) child, exercising JEOD's re-root path
/// (`mass_attach.cc:181-223`).
///
/// All four bodies use the `Body` option-C box inertia diag(0.3333, 0.4167,
/// 0.0833) about the CoM (child2's `optionA` == optionC values).
///
/// Body actions, in order:
/// 1. `attach1_optionB`: child2 → child1 at offset [0.5, 0, 2.5] with
///    `Pitch_Yaw_Roll` (= YZX) Euler [-90°, 0, 0].
/// 2. `attach2_optionA`: child3 → child2 at offset [-1, 0, 0], identity.
/// 3. `attach3_optionA`: child3 → parent at offset [-0.5, 0, 1.5] with
///    `Pitch_Yaw_Roll` [-90°, 0, 0]. child3 is non-root by now (root = child1),
///    so JEOD re-roots: it recasts the child3→parent offset/rotation into the
///    child1→parent frame and attaches child1. The net child1→parent edge is
///    offset [-1, 0, 0], identity struct→struct (verified against `mass.out`).
///
/// Final topology: parent ← child1 ← child2 ← child3. Modeled with
/// [`MassTree::attach_with_reroot`] for the third action, which performs the
/// same root-frame recast as JEOD.
fn build_run_08() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root(
        "Parent".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child1 = tree.add_body(
        "Child1".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child2 = tree.add_body(
        "Child2".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child3 = tree.add_body(
        "Child3".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );

    // attach1_optionB: child2 → child1, Pitch_Yaw_Roll (YZX) [-90°, 0, 0].
    let t_attach1 =
        compute_matrix_from_euler_angles([(-90.0_f64).to_radians(), 0.0, 0.0], EulerSequence::YZX);
    tree.attach(child2, child1, DVec3::new(0.5, 0.0, 2.5), t_attach1);

    // attach2_optionA: child3 → child2 at [-1, 0, 0], identity.
    tree.attach(child3, child2, DVec3::new(-1.0, 0.0, 0.0), DMat3::IDENTITY);

    // attach3_optionA: requested child3 → parent at [-0.5, 0, 1.5],
    // Pitch_Yaw_Roll [-90°, 0, 0]. child3 is non-root → re-root child1.
    let t_attach3 =
        compute_matrix_from_euler_angles([(-90.0_f64).to_radians(), 0.0, 0.0], EulerSequence::YZX);
    tree.attach_with_reroot(child3, parent, DVec3::new(-0.5, 0.0, 1.5), t_attach3);

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child2".to_string(), child2),
            ("Child3".to_string(), child3),
        ],
    )
}

/// RUN_108: the same final chain as RUN_08 (parent ← child1 ← child2 ← child3),
/// but every attach is via named mass points, and the third action again
/// targets the already-attached child3 → re-root child1 under the parent.
///
/// All four bodies use the `Body` option-C box inertia.
///
/// Named points (from `Modified_data` + the RUN_108 input.py renames):
/// - parent point "front_center_right" at [-0.5, 0, 1],
///   T=[[0,0,1],[0,1,0],[-1,0,0]].
/// - child1 point "back_center_right" at [0.5, 0, 1],
///   T=[[0,0,1],[0,1,0],[-1,0,0]].
/// - child2 `mass_points_2`: "front_to_back" at [-0.5,0,0]
///   T=[[-1,0,0],[0,-1,0],[0,0,1]]; "ghost_front_to_back" at [-1.5,0,0] same T.
/// - child3 `mass_points_2`: "front_to_back" at [-0.5,0,0]
///   T=[[-1,0,0],[0,-1,0],[0,0,1]]; "back_to_front" at [0.5,0,0] identity.
///
/// Body actions, in order:
/// 1. pt_attach1: child2.ghost_front_to_back → child1.back_center_right.
/// 2. pt_attach2: child3.back_to_front → child2.front_to_back.
/// 3. pt_attach3: child3.front_to_back → parent.front_center_right. child3 is
///    non-root → re-root child1 under parent. The named-point attach geometry
///    (docking-aligned) is computed inline via [`attach_aligned_offset`] and
///    fed to [`MassTree::attach_with_reroot`].
fn build_run_108() -> (MassTree, Vec<(String, MassBodyId)>) {
    let mut tree = MassTree::new();
    let parent = tree.add_root(
        "Parent".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child1 = tree.add_body(
        "Child1".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child2 = tree.add_body(
        "Child2".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );
    let child3 = tree.add_body(
        "Child3".into(),
        mass_body_spec(1.0, DVec3::ZERO, box_inertia_diag()),
    );

    // parent point "front_center_right" at [-0.5,0,1], T=[[0,0,1],[0,1,0],[-1,0,0]].
    let t_fcr = jeod_trans([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]);
    tree.add_mass_point(
        parent,
        "front_center_right",
        DVec3::new(-0.5, 0.0, 1.0),
        t_fcr,
    );

    // child1 point "back_center_right" at [0.5,0,1], same T as parent point.
    tree.add_mass_point(
        child1,
        "back_center_right",
        DVec3::new(0.5, 0.0, 1.0),
        t_fcr,
    );

    // child2 mass_points_2.
    let t_ftb = jeod_trans([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]);
    tree.add_mass_point(child2, "front_to_back", DVec3::new(-0.5, 0.0, 0.0), t_ftb);
    tree.add_mass_point(
        child2,
        "ghost_front_to_back",
        DVec3::new(-1.5, 0.0, 0.0),
        t_ftb,
    );

    // child3 mass_points_2.
    tree.add_mass_point(child3, "front_to_back", DVec3::new(-0.5, 0.0, 0.0), t_ftb);
    tree.add_mass_point(
        child3,
        "back_to_front",
        DVec3::new(0.5, 0.0, 0.0),
        jeod_trans([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );

    // pt_attach1: child2.ghost_front_to_back → child1.back_center_right.
    tree.attach_aligned(child2, "ghost_front_to_back", child1, "back_center_right");
    // pt_attach2: child3.back_to_front → child2.front_to_back.
    tree.attach_aligned(child3, "back_to_front", child2, "front_to_back");

    // pt_attach3: child3.front_to_back → parent.front_center_right. child3 is
    // non-root now → compute the docking-aligned (offset, T) the named points
    // resolve to, then re-root child1 under parent with that geometry.
    let (offset3, t3) =
        attach_aligned_offset(&tree, child3, "front_to_back", parent, "front_center_right");
    tree.attach_with_reroot(child3, parent, offset3, t3);

    (
        tree,
        vec![
            ("Parent".to_string(), parent),
            ("Child1".to_string(), child1),
            ("Child2".to_string(), child2),
            ("Child3".to_string(), child3),
        ],
    )
}

/// Compute the docking-aligned `(offset, T_parent_child)` that JEOD's
/// named-point attach (`mass_attach.cc:66-136`) resolves to for
/// `child.child_point → parent.parent_point`, expressed as the child
/// structural origin in the parent structural frame and the parent→child
/// structural rotation. This mirrors the geometry inside
/// [`MassTree::attach_aligned`] (the 180° docking yaw `diag(-1,-1,1)`),
/// extracted so the caller can route the result through
/// [`MassTree::attach_with_reroot`] for the non-root-subject case.
fn attach_aligned_offset(
    tree: &MassTree,
    child_id: MassBodyId,
    child_point_name: &str,
    parent_id: MassBodyId,
    parent_point_name: &str,
) -> (DVec3, DMat3) {
    let child_point = tree
        .find_mass_point(child_id, child_point_name)
        .expect("child mass point: must exist for RUN_108 named-point reroot");
    let child_pt_pos = child_point.position;
    let child_pt_t = child_point.t_parent_this;
    let parent_point = tree
        .find_mass_point(parent_id, parent_point_name)
        .expect("parent mass point: must exist for RUN_108 named-point reroot");
    let parent_pt_pos = parent_point.position;
    let parent_pt_t = parent_point.t_parent_this;

    // Invert child point: child_struct in child_point frame.
    let inv_pos = -(child_pt_t * child_pt_pos);
    let inv_t = child_pt_t.transpose();
    // 180° docking yaw (JEOD mass_attach.cc:112-115).
    let t_yaw = DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    let pos_after_yaw = t_yaw * inv_pos;
    let offset = parent_pt_t.transpose() * pos_after_yaw + parent_pt_pos;
    let t_parent_child = inv_t * t_yaw * parent_pt_t;
    (offset, t_parent_child)
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
        let (tree, ids) = build_run_05();
        let reference = load_reference("attach_mass_05_mass.out");
        validate_run("RUN_05", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_06();
        let reference = load_reference("attach_mass_06_mass.out");
        validate_run("RUN_06", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_07();
        let reference = load_reference("attach_mass_07_mass.out");
        validate_run("RUN_07", &tree, &ids, &reference, &mut errors);
    }
    {
        // RUN_09: the root parent has a non-identity struct→body orientation
        // (optionA), so `recompute_composites` produces the composite inertia
        // in the parent body frame (JEOD's `C.M.P. Ib` convention) — compared
        // directly. Child1 is identity-oriented.
        let (tree, ids) = build_run_09();
        let reference = load_reference("attach_mass_09_mass.out");
        validate_run("RUN_09", &tree, &ids, &reference, &mut errors);
    }
    {
        // RUN_109: named-point attach + the RUN_09 non-identity parent
        // struct→body orientation. The docking-aligned named-point geometry
        // resolves to RUN_09's parent←child1 edge, so the composite matches.
        let (tree, ids) = build_run_109();
        let reference = load_reference("attach_mass_109_mass.out");
        validate_run("RUN_109", &tree, &ids, &reference, &mut errors);
    }
    {
        // RUN_08: third action attaches an already-attached child → JEOD
        // re-roots the subject's tree under the new parent.
        let (tree, ids) = build_run_08();
        let reference = load_reference("attach_mass_08_mass.out");
        validate_run("RUN_08", &tree, &ids, &reference, &mut errors);
    }
    {
        // RUN_108: named-point variant of RUN_08; same re-root path via the
        // named-point geometry.
        let (tree, ids) = build_run_108();
        let reference = load_reference("attach_mass_108_mass.out");
        validate_run("RUN_108", &tree, &ids, &reference, &mut errors);
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
    {
        let (tree, ids) = build_run_103();
        let reference = load_reference("attach_mass_103_mass.out");
        validate_run("RUN_103", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_104();
        let reference = load_reference("attach_mass_104_mass.out");
        validate_run("RUN_104", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_106();
        let reference = load_reference("attach_mass_106_mass.out");
        validate_run("RUN_106", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_107();
        let reference = load_reference("attach_mass_107_mass.out");
        validate_run("RUN_107", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_110();
        let reference = load_reference("attach_mass_110_mass.out");
        validate_run("RUN_110", &tree, &ids, &reference, &mut errors);
    }
    {
        let (tree, ids) = build_run_111();
        let reference = load_reference("attach_mass_111_mass.out");
        validate_run("RUN_111", &tree, &ids, &reference, &mut errors);
    }

    let mut report = CrossvalReport::compute("tier3_sim_attach_mass", &[], &[]);
    report.add_extra("composite_mass", errors.mass, "kg");
    report.add_extra("composite_com", errors.com, "m");
    report.add_extra("composite_inertia", errors.inertia, "kg*m^2");
    report.write();
}
