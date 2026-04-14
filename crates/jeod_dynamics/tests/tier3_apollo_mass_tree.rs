//! Tier 3: Apollo stack mass tree attach/detach cross-validation.
//!
//! Validates the full 12-phase Apollo mission mass tree sequence against
//! JEOD SIM_Apollo reference data (Docker-generated `.out` files).
//!
//! This test exercises:
//! - Named attachment points (`attach_aligned`)
//! - 8-body hierarchical mass tree (S1, S2, S3, LES, CM, SM, LM, DM)
//! - 7 initial attachments forming the launch stack
//! - 12 mission phases with detach/re-attach operations
//! - Composite mass, CoM, and inertia validation at each phase
//!
//! Reference: `sims/SIM_Apollo/SET_test/RUN_test/input.py` in JEOD v5.4.

use glam::{DMat3, DVec3};
use jeod_dynamics::{MassProperties, MassTree};

// ── Unit conversion constants ──

/// Pounds-mass to kilograms (exact).
const LB_TO_KG: f64 = 0.453_592_37;
/// Feet to meters (exact).
const FT_TO_M: f64 = 0.3048;
/// lb*ft^2 to kg*m^2 (lbm * ft^2).
const LB_FT2_TO_KG_M2: f64 = LB_TO_KG * FT_TO_M * FT_TO_M;

// ── 180° yaw rotation about Z (used frequently for attachment points) ──

fn yaw_180() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    )
}

// ── Apollo body definitions (from JEOD Modified_data/mass/*.py) ──

/// Create mass properties for a body with diagonal inertia.
/// `mass_lb`: mass in pounds, `cm_x_ft`: CoM X in feet (Y,Z = 0),
/// `ixx`, `iyy`, `izz` in lb*ft^2.
fn apollo_mass(mass_lb: f64, cm_x_ft: f64, ixx: f64, iyy: f64, izz: f64) -> MassProperties {
    MassProperties::with_inertia(
        mass_lb * LB_TO_KG,
        DMat3::from_diagonal(DVec3::new(
            ixx * LB_FT2_TO_KG_M2,
            iyy * LB_FT2_TO_KG_M2,
            izz * LB_FT2_TO_KG_M2,
        )),
        DVec3::new(cm_x_ft * FT_TO_M, 0.0, 0.0),
    )
}

/// Build the full Apollo mass tree with all 8 bodies and their attachment points.
///
/// Returns (tree, body IDs): (tree, cm, sm, lm, dm, s3, s2, s1, les).
fn build_apollo_tree() -> (
    MassTree,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let mut tree = MassTree::new();

    // Command Module: 12,807 lb, CoM at 8.7 ft
    // Source: Modified_data/mass/command_module.py
    let cm = tree.add_root(
        "cm".into(),
        apollo_mass(12807.0, 8.7, 157372.0, 64624.0, 64624.0),
    );

    // Service Module: 54,064 lb, CoM at 12.3 ft
    let sm = tree.add_body(
        "sm".into(),
        apollo_mass(54064.0, 12.3, 1107231.0, 1235227.0, 1235227.0),
    );

    // Lunar Module (Ascent): 10,582 lb, CoM at 5.45 ft
    let lm = tree.add_body(
        "lm".into(),
        apollo_mass(10582.0, 5.45, 259259.0, 155822.0, 155822.0),
    );

    // Descent Module: 25,640 lb, CoM at 5.0 ft
    let dm = tree.add_body(
        "dm".into(),
        apollo_mass(25640.0, 5.0, 628180.0, 367506.0, 367506.0),
    );

    // Stage 3: 274,171 lb, CoM at 30.65 ft
    let s3 = tree.add_body(
        "s3".into(),
        apollo_mass(274171.0, 30.65, 16138048.0, 29532558.0, 29532558.0),
    );

    // Stage 2: 1,083,480 lb, CoM at 40.75 ft
    let s2 = tree.add_body(
        "s2".into(),
        apollo_mass(1083480.0, 40.75, 147488715.0, 223676545.0, 223676545.0),
    );

    // Stage 1: 5,031,023 lb, CoM at 69.0 ft
    let s1 = tree.add_body(
        "s1".into(),
        apollo_mass(5031023.0, 69.0, 684848006.0, 2338482378.0, 2338482378.0),
    );

    // Launch Escape System: 9,200 lb, CoM at 16.25 ft
    let les = tree.add_body(
        "les".into(),
        apollo_mass(9200.0, 16.25, 5566.0, 205231.0, 205231.0),
    );

    // ── Attachment points (from Modified_data/attach/*.py) ──

    // CM: "SM interface" at (11.6, 0, 0) ft, 0° rotation
    //     "CM docking port" at (4.0, 0, 0) ft, 180° about Z
    tree.add_mass_point(
        cm,
        "SM interface",
        DVec3::new(11.6 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    tree.add_mass_point(
        cm,
        "CM docking port",
        DVec3::new(4.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );

    // SM: "Stage 3 interface" at (-20.9, 0, 0) ft, 180° about Z
    //     "CM interface" at (24.6, 0, 0) ft, 0° rotation
    tree.add_mass_point(
        sm,
        "Stage 3 interface",
        DVec3::new(-20.9 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    tree.add_mass_point(
        sm,
        "CM interface",
        DVec3::new(24.6 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );

    // LM (Ascent Module):
    //   "LM docking port" at (10.9, 0, 0) ft, 0° rotation
    //   "Descent Module interface" at (0, 0, 0) ft, 180° about Z
    //   "Stage 3 interface" at (-10.0, 0, 0) ft, 180° about Z
    tree.add_mass_point(
        lm,
        "LM docking port",
        DVec3::new(10.9 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    tree.add_mass_point(lm, "Descent Module interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        lm,
        "Stage 3 interface",
        DVec3::new(-10.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );

    // DM (Descent Module):
    //   "Ascent Module interface" at (0, 0, 0) ft, 0° rotation
    //   "Stage 3 interface" at (-10.0, 0, 0) ft, 180° about Z
    tree.add_mass_point(dm, "Ascent Module interface", DVec3::ZERO, DMat3::IDENTITY);
    tree.add_mass_point(
        dm,
        "Stage 3 interface",
        DVec3::new(-10.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );

    // S3: "Stage 2 interface" at (0, 0, 0) ft, 180° about Z
    //     "LEM/SM/CM interface" at (61.3, 0, 0) ft, 0° rotation
    tree.add_mass_point(s3, "Stage 2 interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        s3,
        "LEM/SM/CM interface",
        DVec3::new(61.3 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );

    // S2: "Stage 1 interface" at (0, 0, 0) ft, 180° about Z
    //     "Stage 3 interface" at (81.5, 0, 0) ft, 0° rotation
    tree.add_mass_point(s2, "Stage 1 interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        s2,
        "Stage 3 interface",
        DVec3::new(81.5 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );

    // S1: "Stage 2 interface" at (138.0, 0, 0) ft, 0° rotation
    tree.add_mass_point(
        s1,
        "Stage 2 interface",
        DVec3::new(138.0 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );

    // LES: "CM interface" at (0, 0, 0) ft, 180° about Z
    tree.add_mass_point(les, "CM interface", DVec3::ZERO, yaw_180());

    (tree, cm, sm, lm, dm, s3, s2, s1, les)
}

/// Assemble the full launch stack (7 attachments).
///
/// Source: Modified_data/attach/launch_stack.py
#[allow(clippy::too_many_arguments)]
fn assemble_launch_stack(
    tree: &mut MassTree,
    _cm: usize,
    sm: usize,
    lm: usize,
    dm: usize,
    s3: usize,
    s2: usize,
    s1: usize,
    les: usize,
) {
    // 1. DM "Ascent Module interface" -> LM "Descent Module interface"
    tree.attach_aligned(
        dm,
        "Ascent Module interface",
        lm,
        "Descent Module interface",
    );
    // 2. SM "CM interface" -> CM "SM interface"
    tree.attach_aligned(sm, "CM interface", _cm, "SM interface");
    // 3. S3 "LEM/SM/CM interface" -> SM "Stage 3 interface"
    tree.attach_aligned(s3, "LEM/SM/CM interface", sm, "Stage 3 interface");
    // 4. LM "Stage 3 interface" -> S3 "LEM/SM/CM interface"
    tree.attach_aligned(lm, "Stage 3 interface", s3, "LEM/SM/CM interface");
    // 5. S2 "Stage 3 interface" -> S3 "Stage 2 interface"
    tree.attach_aligned(s2, "Stage 3 interface", s3, "Stage 2 interface");
    // 6. S1 "Stage 2 interface" -> S2 "Stage 1 interface"
    tree.attach_aligned(s1, "Stage 2 interface", s2, "Stage 1 interface");
    // 7. LES "CM interface" -> CM "CM docking port"
    tree.attach_aligned(les, "CM interface", _cm, "CM docking port");
}

fn test_data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
}

fn load_reference(filename: &str) -> jeod_test_data::apollo_mass_tree::PrintedTree {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "Apollo reference data not found at {}. \
         Generate it with: docker run ... trick/generate_references.sh",
        path.display()
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    jeod_test_data::apollo_mass_tree::parse_print_tree(&content)
}

/// Assert that a body's composite properties match the reference.
fn assert_composite_match(
    tree: &MassTree,
    body_id: usize,
    ref_body: &jeod_test_data::apollo_mass_tree::PrintedBody,
    phase: &str,
) {
    let body = tree.get(body_id);
    let comp = &body.composite_properties;
    let tol_mass = 1e-3; // kg (absolute)
    let tol_pos = 1e-6; // m (absolute)
    let tol_inertia = 1e-2; // kg*m^2 (absolute)

    assert!(
        (comp.mass - ref_body.composite_mass).abs() < tol_mass,
        "[{phase}] {}: composite mass {:.6} != ref {:.6} (diff={:.2e})",
        body.name,
        comp.mass,
        ref_body.composite_mass,
        (comp.mass - ref_body.composite_mass).abs()
    );

    let pos_diff = (comp.position - ref_body.composite_cm).length();
    assert!(
        pos_diff < tol_pos,
        "[{phase}] {}: composite CoM diff {pos_diff:.2e} m exceeds tolerance {tol_pos:.0e}",
        body.name
    );

    for (col_idx, (our_col, ref_col)) in [
        (comp.inertia.x_axis, ref_body.composite_inertia.x_axis),
        (comp.inertia.y_axis, ref_body.composite_inertia.y_axis),
        (comp.inertia.z_axis, ref_body.composite_inertia.z_axis),
    ]
    .iter()
    .enumerate()
    {
        let diff = (*our_col - *ref_col).length();
        assert!(
            diff < tol_inertia,
            "[{phase}] {}: composite inertia col {col_idx} diff {diff:.2e} exceeds tolerance {tol_inertia:.0e}",
            body.name
        );
    }
}

#[test]
fn tier3_apollo_mass_tree() {
    let (mut tree, cm, sm, lm, dm, s3, s2, s1, les) = build_apollo_tree();
    assemble_launch_stack(&mut tree, cm, sm, lm, dm, s3, s2, s1, les);

    // Phase 0: Full stack
    let ref_full = load_reference("apollo_Full_Stack.out");
    if let Some(ref_cm) = ref_full.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Full_Stack");
    }

    // Phase 1: First stage separation
    tree.detach(s1);
    let ref_1 = load_reference("apollo_1st_Stage_Sep.out");
    if let Some(ref_cm) = ref_1.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "1st_Stage_Sep");
    }

    // Phase 2: Second stage separation
    tree.detach(s2);
    let ref_2 = load_reference("apollo_2nd_Stage_Sep.out");
    if let Some(ref_cm) = ref_2.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "2nd_Stage_Sep");
    }

    // Phase 3: LES jettison
    tree.detach(les);
    let ref_3 = load_reference("apollo_LES_Jettison.out");
    if let Some(ref_cm) = ref_3.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "LES_Jettison");
    }

    // Phase 4: Third stage separation
    tree.detach(s3);
    let ref_4 = load_reference("apollo_3rd_Stage_Sep.out");
    if let Some(ref_cm) = ref_4.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "3rd_Stage_Sep");
    }

    // Phase 5: LM separation (LM detaches from CM's subtree)
    tree.detach(lm);
    let ref_5_lem = load_reference("apollo_LEM_Sep.out");
    if let Some(ref_lm) = ref_5_lem.find("lm") {
        assert_composite_match(&tree, lm, ref_lm, "LEM_Sep/lm");
    }
    let ref_5_apollo = load_reference("apollo_Apollo.out");
    if let Some(ref_cm) = ref_5_apollo.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Apollo/cm");
    }

    // Phase 6: LM docks to CM (trans-lunar configuration)
    // input.py: cm_dyn.dyn_body.attach_child("CM docking port", "LM docking port", lm_dyn.dyn_body)
    // attach_child(parent_point, child_point, child) is equivalent to
    // child.attach_to(child_point, parent_point, parent)
    tree.attach_aligned(lm, "LM docking port", cm, "CM docking port");
    let ref_6 = load_reference("apollo_Trans_Lunar.out");
    if let Some(ref_cm) = ref_6.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Trans_Lunar");
    }

    // Phase 7: LM undocks for lunar descent
    tree.detach(lm);
    let ref_7_lm = load_reference("apollo_LM_Descent.out");
    if let Some(ref_lm) = ref_7_lm.find("lm") {
        assert_composite_match(&tree, lm, ref_lm, "LM_Descent/lm");
    }
    let ref_7_cm = load_reference("apollo_Lunar_Orbit.out");
    if let Some(ref_cm) = ref_7_cm.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Lunar_Orbit/cm");
    }

    // Phase 8: Descent module separation
    tree.detach(dm);
    let ref_8 = load_reference("apollo_LM_Ascent.out");
    if let Some(ref_lm) = ref_8.find("lm") {
        assert_composite_match(&tree, lm, ref_lm, "LM_Ascent");
    }

    // Phase 9: LM re-docks to CM (lunar rendezvous)
    // input.py: lm_dyn.dyn_body.attach_to("LM docking port", "CM docking port", cm_dyn.dyn_body)
    tree.attach_aligned(lm, "LM docking port", cm, "CM docking port");
    let ref_9 = load_reference("apollo_Lunar_Rendezvous.out");
    if let Some(ref_cm) = ref_9.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Lunar_Rendezvous");
    }

    // Phase 10: LM final separation
    tree.detach(lm);
    let ref_10 = load_reference("apollo_Return.out");
    if let Some(ref_cm) = ref_10.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Return");
    }

    // Phase 11: SM jettison (entry configuration)
    tree.detach(sm);
    let ref_11 = load_reference("apollo_Entry.out");
    if let Some(ref_cm) = ref_11.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Entry");
    }
    // Final.out should match Entry.out for the CM (printed twice in input.py)
    let ref_final = load_reference("apollo_Final.out");
    if let Some(ref_cm) = ref_final.find("cm") {
        assert_composite_match(&tree, cm, ref_cm, "Final");
    }
}
