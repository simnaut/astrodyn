//! Tier 3: Mass tree attach/detach edge case tests.
//!
//! These tests exercise the `MassTree` attach/detach API directly, verifying
//! composite mass, center of mass, and inertia against analytical formulas
//! (parallel axis theorem, mass-weighted CoM averaging). No JEOD reference
//! data is needed -- all expected values are computed from first principles.

use glam::{DMat3, DVec3};
use jeod_dynamics::{MassProperties, MassTree};

// ── Helpers ──

/// Assert two DVec3 are approximately equal.
fn assert_vec3_close(a: DVec3, b: DVec3, tol: f64, msg: &str) {
    let diff = (a - b).length();
    assert!(
        diff < tol,
        "{msg}: diff {diff:.2e} exceeds tolerance {tol:.2e}"
    );
}

/// Assert two DMat3 are approximately equal (per-column check).
fn assert_mat3_close(a: DMat3, b: DMat3, tol: f64, msg: &str) {
    for (i, (ca, cb)) in [
        (a.x_axis, b.x_axis),
        (a.y_axis, b.y_axis),
        (a.z_axis, b.z_axis),
    ]
    .iter()
    .enumerate()
    {
        let diff = (*ca - *cb).length();
        assert!(
            diff < tol,
            "{msg}: column {i} diff {diff:.2e} exceeds tolerance {tol:.2e}"
        );
    }
}

/// Assert a matrix is symmetric: M[i][j] == M[j][i].
fn assert_symmetric(m: DMat3, tol: f64, msg: &str) {
    assert_mat3_close(m, m.transpose(), tol, msg);
}

/// Parallel axis theorem: inertia of a point mass at offset r from reference.
/// I[i][j] = mass * (r^2 * delta_ij - r[i] * r[j])
fn point_mass_inertia(mass: f64, offset: DVec3) -> DMat3 {
    let r_sq = offset.length_squared();
    let outer = DMat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
    DMat3::from_diagonal(DVec3::splat(r_sq)) * mass - outer * mass
}

// ── Tests ──

#[test]
fn tier3_mass_single_attach_composite() {
    // Create parent (1000 kg) and child (500 kg)
    // Attach child at offset [1, 0, 0] m
    // Verify composite mass = 1500 kg
    // Verify composite CoM shifted toward child
    // Verify composite inertia uses parallel axis theorem
    let mut tree = MassTree::new();

    let parent_core = MassProperties::new(1000.0);
    let pid = tree.add_root("parent".into(), parent_core);

    let child_core = MassProperties::new(500.0);
    let cid = tree.add_body("child".into(), child_core);

    tree.attach(cid, pid, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

    let comp = &tree.get(pid).composite_properties;

    // Composite mass = 1500 kg
    assert!(
        (comp.mass - 1500.0).abs() < 1e-12,
        "composite mass = {}, expected 1500",
        comp.mass
    );

    // Composite CoM: (1000*0 + 500*1) / 1500 = 1/3 along x
    let expected_cm = DVec3::new(1.0 / 3.0, 0.0, 0.0);
    assert_vec3_close(comp.position, expected_cm, 1e-12, "composite CoM");

    // Composite inertia via parallel axis theorem:
    //   Parent: I_core = 1000*I, offset from composite CoM = [-1/3, 0, 0]
    //     I_parent = 1000*I + point_mass(1000, [-1/3, 0, 0])
    //     point_mass(1000, [-1/3,0,0]) = 1000 * diag(0, 1/9, 1/9) = diag(0, 111.11, 111.11)
    //     I_parent = diag(1000, 1111.11, 1111.11)
    //
    //   Child: I_core = 500*I, offset from composite CoM = [2/3, 0, 0]
    //     point_mass(500, [2/3,0,0]) = 500 * diag(0, 4/9, 4/9) = diag(0, 222.22, 222.22)
    //     I_child = diag(500, 722.22, 722.22)
    //
    //   Total = diag(1500, 1833.33, 1833.33)
    let parent_shift = point_mass_inertia(1000.0, DVec3::new(-1.0 / 3.0, 0.0, 0.0));
    let child_shift = point_mass_inertia(500.0, DVec3::new(2.0 / 3.0, 0.0, 0.0));
    let expected_inertia =
        DMat3::IDENTITY * 1000.0 + parent_shift + DMat3::IDENTITY * 500.0 + child_shift;
    assert_mat3_close(comp.inertia, expected_inertia, 1e-8, "composite inertia");
}

#[test]
fn tier3_mass_detach_recovers_original() {
    // Attach child to parent, then detach.
    // Verify parent returns to original mass/CoM/inertia.
    let mut tree = MassTree::new();

    let parent_inertia = DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0));
    let parent_core = MassProperties::with_inertia(50.0, parent_inertia, DVec3::new(0.5, 0.0, 0.0));
    let pid = tree.add_root("parent".into(), parent_core);

    let orig_mass = tree.get(pid).composite_properties.mass;
    let orig_pos = tree.get(pid).composite_properties.position;
    let orig_inertia = tree.get(pid).composite_properties.inertia;

    let child_core = MassProperties::new(25.0);
    let cid = tree.add_body("child".into(), child_core);

    tree.attach(cid, pid, DVec3::new(3.0, 1.0, -0.5), DMat3::IDENTITY);

    // After attach, composite should differ from original
    assert!(
        (tree.get(pid).composite_properties.mass - orig_mass).abs() > 1.0,
        "mass should change after attach"
    );

    tree.detach(cid);

    // After detach, parent should recover original properties
    assert!(
        (tree.get(pid).composite_properties.mass - orig_mass).abs() < 1e-12,
        "mass not recovered after detach"
    );
    assert_vec3_close(
        tree.get(pid).composite_properties.position,
        orig_pos,
        1e-12,
        "position not recovered after detach",
    );
    assert_mat3_close(
        tree.get(pid).composite_properties.inertia,
        orig_inertia,
        1e-12,
        "inertia not recovered after detach",
    );
}

#[test]
fn tier3_mass_multi_level_hierarchy() {
    // Parent -> Child -> Grandchild
    // Verify composite properties account for all three levels
    // Detach grandchild: verify intermediate composite updates
    let mut tree = MassTree::new();

    let ma = 100.0;
    let mb = 50.0;
    let mc = 25.0;
    let a = tree.add_root("A".into(), MassProperties::new(ma));
    let b = tree.add_body("B".into(), MassProperties::new(mb));
    let c = tree.add_body("C".into(), MassProperties::new(mc));

    // B at [2, 0, 0] on A; C at [1, 0, 0] on B (so C at [3, 0, 0] in A's frame)
    tree.attach(b, a, DVec3::new(2.0, 0.0, 0.0), DMat3::IDENTITY);
    tree.attach(c, b, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

    // A composite mass = 175
    assert!(
        (tree.get(a).composite_properties.mass - 175.0).abs() < 1e-12,
        "A composite mass"
    );

    // B composite mass = 75 (B + C)
    assert!(
        (tree.get(b).composite_properties.mass - 75.0).abs() < 1e-12,
        "B composite mass"
    );

    // B composite CoM (in B's struct frame): (50*0 + 25*1) / 75 = 1/3
    let b_comp_cm = DVec3::new(1.0 / 3.0, 0.0, 0.0);
    assert_vec3_close(
        tree.get(b).composite_properties.position,
        b_comp_cm,
        1e-12,
        "B composite CoM",
    );

    // A composite CoM: B's composite CoM in A's frame = [2 + 1/3, 0, 0]
    // A CoM = (100*0 + 75*(2 + 1/3)) / 175 = 75 * 7/3 / 175 = 175/175 = 1.0
    let a_expected_cm_x = (ma * 0.0 + 75.0 * (2.0 + 1.0 / 3.0)) / 175.0;
    assert_vec3_close(
        tree.get(a).composite_properties.position,
        DVec3::new(a_expected_cm_x, 0.0, 0.0),
        1e-10,
        "A composite CoM",
    );

    // Now detach grandchild C from B
    tree.detach(c);

    // B composite should now be just B alone
    assert!(
        (tree.get(b).composite_properties.mass - 50.0).abs() < 1e-12,
        "B mass after grandchild detach"
    );
    assert_vec3_close(
        tree.get(b).composite_properties.position,
        DVec3::ZERO,
        1e-12,
        "B CoM after grandchild detach",
    );

    // A composite should update: A + B only
    assert!(
        (tree.get(a).composite_properties.mass - 150.0).abs() < 1e-12,
        "A mass after grandchild detach"
    );
    // A CoM = (100*0 + 50*2) / 150 = 100/150 = 2/3
    assert_vec3_close(
        tree.get(a).composite_properties.position,
        DVec3::new(2.0 / 3.0, 0.0, 0.0),
        1e-12,
        "A CoM after grandchild detach",
    );
}

#[test]
fn tier3_mass_symmetric_children_zero_com_shift() {
    // Parent with two identical children at +x and -x
    // Composite CoM should be at parent CoM (symmetry)
    // Composite mass = parent + 2*child
    let mut tree = MassTree::new();

    let parent_mass = 200.0;
    let child_mass = 50.0;
    let pid = tree.add_root("parent".into(), MassProperties::new(parent_mass));
    let c1 = tree.add_body("child+x".into(), MassProperties::new(child_mass));
    let c2 = tree.add_body("child-x".into(), MassProperties::new(child_mass));

    let offset = 3.0;
    tree.attach(c1, pid, DVec3::new(offset, 0.0, 0.0), DMat3::IDENTITY);
    tree.attach(c2, pid, DVec3::new(-offset, 0.0, 0.0), DMat3::IDENTITY);

    let comp = &tree.get(pid).composite_properties;

    // Composite mass = 300
    assert!(
        (comp.mass - 300.0).abs() < 1e-12,
        "composite mass = {}",
        comp.mass
    );

    // Composite CoM at origin (symmetry cancels)
    assert_vec3_close(comp.position, DVec3::ZERO, 1e-12, "composite CoM symmetry");

    // Inertia should also be symmetric about the origin
    // I_xx: no contribution from x-offsets (parallel axis for x-offset only adds to yy, zz)
    // I_yy = I_zz (by symmetry of +x/-x offsets)
    let inertia = comp.inertia;
    assert!(
        (inertia.y_axis.y - inertia.z_axis.z).abs() < 1e-10,
        "I_yy should equal I_zz by symmetry"
    );
    assert_symmetric(inertia, 1e-12, "symmetric children inertia");
}

#[test]
fn tier3_mass_parallel_axis_theorem() {
    // Solid sphere (known inertia I = 2/5*m*r^2) attached at offset d
    // Composite inertia should include I + m*d^2 (parallel axis)
    // Verify against analytical formula
    let mut tree = MassTree::new();

    let parent_mass = 100.0;
    let pid = tree.add_root("parent".into(), MassProperties::new(parent_mass));

    // Sphere: mass=10 kg, radius=0.5 m
    // I_sphere = 2/5 * 10 * 0.25 = 1.0 kg*m^2 (isotropic)
    let sphere_mass = 10.0;
    let sphere_radius = 0.5;
    let i_sphere = 2.0 / 5.0 * sphere_mass * sphere_radius * sphere_radius;
    let sphere_inertia = DMat3::from_diagonal(DVec3::splat(i_sphere));
    let sphere_core = MassProperties::with_inertia(sphere_mass, sphere_inertia, DVec3::ZERO);
    let sid = tree.add_body("sphere".into(), sphere_core);

    let d = DVec3::new(5.0, 0.0, 0.0);
    tree.attach(sid, pid, d, DMat3::IDENTITY);

    let comp = &tree.get(pid).composite_properties;
    let total_mass = parent_mass + sphere_mass;

    // Composite CoM
    let cm = (parent_mass * DVec3::ZERO + sphere_mass * d) / total_mass;
    assert_vec3_close(comp.position, cm, 1e-12, "composite CoM");

    // Analytical composite inertia:
    //   Parent contribution: I_parent_core + point_mass(parent_mass, -cm)
    //   Sphere contribution: I_sphere + point_mass(sphere_mass, d - cm)
    let parent_offset = DVec3::ZERO - cm;
    let sphere_offset = d - cm;
    let expected_inertia = DMat3::IDENTITY * parent_mass
        + point_mass_inertia(parent_mass, parent_offset)
        + sphere_inertia
        + point_mass_inertia(sphere_mass, sphere_offset);

    assert_mat3_close(
        comp.inertia,
        expected_inertia,
        1e-8,
        "parallel axis inertia",
    );
}

#[test]
fn tier3_mass_reattach_different_position() {
    // Attach child at position A, detach, reattach at position B
    // Verify composite properties change correctly
    let mut tree = MassTree::new();

    let parent_mass = 80.0;
    let child_mass = 20.0;
    let pid = tree.add_root("parent".into(), MassProperties::new(parent_mass));
    let cid = tree.add_body("child".into(), MassProperties::new(child_mass));

    let total_mass = parent_mass + child_mass;

    // Position A: [2, 0, 0]
    let pos_a = DVec3::new(2.0, 0.0, 0.0);
    tree.attach(cid, pid, pos_a, DMat3::IDENTITY);

    let cm_a = (parent_mass * DVec3::ZERO + child_mass * pos_a) / total_mass;
    assert_vec3_close(
        tree.get(pid).composite_properties.position,
        cm_a,
        1e-12,
        "CoM at position A",
    );

    tree.detach(cid);

    // Position B: [0, 3, 0]
    let pos_b = DVec3::new(0.0, 3.0, 0.0);
    tree.attach(cid, pid, pos_b, DMat3::IDENTITY);

    let cm_b = (parent_mass * DVec3::ZERO + child_mass * pos_b) / total_mass;
    assert_vec3_close(
        tree.get(pid).composite_properties.position,
        cm_b,
        1e-12,
        "CoM at position B",
    );

    // Verify inertia changed (different offset direction)
    // At pos_a the offset was along x, so I_yy and I_zz got the contribution.
    // At pos_b the offset is along y, so I_xx and I_zz get the contribution.
    let inertia_b = tree.get(pid).composite_properties.inertia;
    let parent_offset_b = DVec3::ZERO - cm_b;
    let child_offset_b = pos_b - cm_b;
    let expected_inertia_b = DMat3::IDENTITY * parent_mass
        + point_mass_inertia(parent_mass, parent_offset_b)
        + DMat3::IDENTITY * child_mass
        + point_mass_inertia(child_mass, child_offset_b);
    assert_mat3_close(inertia_b, expected_inertia_b, 1e-8, "inertia at position B");
}

#[test]
fn tier3_mass_many_children_composite() {
    // Attach 10 identical children at various offsets
    // Verify total composite mass = parent + 10*child
    // Verify CoM is mass-weighted average
    let mut tree = MassTree::new();

    let parent_mass = 500.0;
    let child_mass = 10.0;
    let n = 10;
    let pid = tree.add_root("parent".into(), MassProperties::new(parent_mass));

    let mut children = Vec::new();
    let mut offsets = Vec::new();

    for i in 0..n {
        let cid = tree.add_body(format!("child_{i}"), MassProperties::new(child_mass));
        // Spread children along a helix
        let angle = i as f64 * std::f64::consts::TAU / n as f64;
        let offset = DVec3::new(i as f64 * 0.5, 2.0 * angle.cos(), 2.0 * angle.sin());
        tree.attach(cid, pid, offset, DMat3::IDENTITY);
        children.push(cid);
        offsets.push(offset);
    }

    let comp = &tree.get(pid).composite_properties;
    let total_mass = parent_mass + n as f64 * child_mass;

    // Total mass
    assert!(
        (comp.mass - total_mass).abs() < 1e-10,
        "total mass = {}, expected {total_mass}",
        comp.mass
    );

    // CoM = mass-weighted average
    let mut weighted_sum = parent_mass * DVec3::ZERO; // parent at origin
    for offset in &offsets {
        weighted_sum += child_mass * *offset;
    }
    let expected_cm = weighted_sum / total_mass;
    assert_vec3_close(comp.position, expected_cm, 1e-10, "many children CoM");
}

#[test]
fn tier3_mass_negligible_mass_child() {
    // Attach a near-zero-mass child (structural attachment point).
    // Should barely change parent composite properties.
    // Note: MassProperties::new() requires mass > 0, so we use a tiny mass.
    let mut tree = MassTree::new();

    let parent_mass = 1000.0;
    let parent_inertia = DMat3::from_diagonal(DVec3::new(500.0, 600.0, 700.0));
    let parent_core = MassProperties::with_inertia(parent_mass, parent_inertia, DVec3::ZERO);
    let pid = tree.add_root("parent".into(), parent_core);

    let epsilon = 1e-10;
    let tiny_core = MassProperties::new(epsilon);
    let cid = tree.add_body("structural_point".into(), tiny_core);

    tree.attach(cid, pid, DVec3::new(5.0, 3.0, -2.0), DMat3::IDENTITY);

    let comp = &tree.get(pid).composite_properties;

    // Mass should be essentially unchanged
    assert!(
        (comp.mass - parent_mass).abs() < 1e-6,
        "mass changed significantly: {}",
        comp.mass
    );

    // CoM should be essentially unchanged
    assert_vec3_close(comp.position, DVec3::ZERO, 1e-6, "CoM shifted by tiny mass");

    // Inertia should be essentially unchanged
    assert_mat3_close(
        comp.inertia,
        parent_inertia,
        1e-4,
        "inertia changed by tiny mass",
    );
}

#[test]
fn tier3_mass_inertia_tensor_symmetry() {
    // After various attach/detach operations, composite inertia must be symmetric.
    // Use asymmetric offsets and rotations to stress the symmetry property.
    let mut tree = MassTree::new();

    let parent_core = MassProperties::with_inertia(
        50.0,
        DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0)),
        DVec3::new(0.1, -0.2, 0.3),
    );
    let pid = tree.add_root("parent".into(), parent_core);

    // Child 1: asymmetric offset
    let c1_core = MassProperties::with_inertia(
        20.0,
        DMat3::from_diagonal(DVec3::new(10.0, 30.0, 50.0)),
        DVec3::new(-0.1, 0.05, 0.0),
    );
    let c1 = tree.add_body("child1".into(), c1_core);

    // 30-degree rotation about Y
    let angle = 30.0_f64.to_radians();
    let c = angle.cos();
    let s = angle.sin();
    let rot_y_30 = DMat3::from_cols(
        DVec3::new(c, 0.0, -s),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(s, 0.0, c),
    );

    tree.attach(c1, pid, DVec3::new(1.5, -0.7, 2.1), rot_y_30);
    assert_symmetric(
        tree.get(pid).composite_properties.inertia,
        1e-10,
        "after first attach",
    );

    // Child 2: another asymmetric offset and rotation (60 deg about X)
    let c2_core = MassProperties::with_inertia(
        15.0,
        DMat3::from_diagonal(DVec3::new(5.0, 15.0, 25.0)),
        DVec3::ZERO,
    );
    let c2 = tree.add_body("child2".into(), c2_core);

    let angle2 = 60.0_f64.to_radians();
    let c2a = angle2.cos();
    let s2a = angle2.sin();
    let rot_x_60 = DMat3::from_cols(
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, c2a, s2a),
        DVec3::new(0.0, -s2a, c2a),
    );

    tree.attach(c2, pid, DVec3::new(-0.3, 1.8, -0.9), rot_x_60);
    assert_symmetric(
        tree.get(pid).composite_properties.inertia,
        1e-10,
        "after second attach",
    );

    // Detach child 1
    tree.detach(c1);
    assert_symmetric(
        tree.get(pid).composite_properties.inertia,
        1e-10,
        "after detach child1",
    );

    // Detach child 2
    tree.detach(c2);
    assert_symmetric(
        tree.get(pid).composite_properties.inertia,
        1e-10,
        "after detach all",
    );
}

#[test]
fn tier3_mass_detach_all_children() {
    // Attach 5 children, then detach all one by one.
    // After all detached, parent should match original properties exactly.
    let mut tree = MassTree::new();

    let parent_inertia = DMat3::from_diagonal(DVec3::new(400.0, 500.0, 600.0));
    let parent_core =
        MassProperties::with_inertia(200.0, parent_inertia, DVec3::new(0.1, -0.05, 0.0));
    let pid = tree.add_root("parent".into(), parent_core);

    let orig_mass = tree.get(pid).composite_properties.mass;
    let orig_pos = tree.get(pid).composite_properties.position;
    let orig_inertia = tree.get(pid).composite_properties.inertia;

    let mut child_ids = Vec::new();
    for i in 0..5 {
        let child_mass = 10.0 + i as f64 * 5.0; // 10, 15, 20, 25, 30 kg
        let cid = tree.add_body(format!("child_{i}"), MassProperties::new(child_mass));
        let offset = DVec3::new(
            (i as f64 - 2.0) * 1.5,
            ((i * 37) as f64 % 7.0 - 3.0) * 0.5,
            ((i * 53) as f64 % 11.0 - 5.0) * 0.3,
        );
        tree.attach(cid, pid, offset, DMat3::IDENTITY);
        child_ids.push(cid);
    }

    // Composite mass should be larger
    assert!(
        tree.get(pid).composite_properties.mass > orig_mass + 50.0,
        "mass should increase with children"
    );

    // Detach all children one by one
    for cid in child_ids {
        tree.detach(cid);
    }

    // Should recover original properties
    assert!(
        (tree.get(pid).composite_properties.mass - orig_mass).abs() < 1e-12,
        "mass not recovered: {} vs {}",
        tree.get(pid).composite_properties.mass,
        orig_mass
    );
    assert_vec3_close(
        tree.get(pid).composite_properties.position,
        orig_pos,
        1e-12,
        "position not recovered after detaching all",
    );
    assert_mat3_close(
        tree.get(pid).composite_properties.inertia,
        orig_inertia,
        1e-12,
        "inertia not recovered after detaching all",
    );
}
