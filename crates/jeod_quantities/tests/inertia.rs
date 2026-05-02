//! Integration tests for `InertiaTensor<F>` — exercises the public surface
//! (frame-tagged construction, rotational invariance, parallel-axis
//! composition, frame-mismatch detection at the type level).

use glam::{DMat3, DVec3};
use jeod_quantities::prelude::*;

const TOL: f64 = 1e-12;

fn approx_mat_eq(a: DMat3, b: DMat3, tol: f64) -> bool {
    for c in 0..3 {
        for r in 0..3 {
            if (a.col(c)[r] - b.col(c)[r]).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[test]
fn from_components_is_symmetric_and_diagonal_matches_principal() {
    let i_full = InertiaTensor::<RootInertial>::from_components(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
    let i_diag = InertiaTensor::<RootInertial>::from_principal(1.0, 2.0, 3.0);
    assert_eq!(i_full, i_diag);
}

#[test]
fn rotation_preserves_trace() {
    // Trace of an inertia tensor is rotation-invariant. Cook a
    // non-trivial rotation (45° about (1,1,1)/√3) and verify.
    let i = InertiaTensor::<RootInertial>::from_components(2.0, 3.0, 5.0, 0.1, 0.2, 0.3);
    let axis = DVec3::new(1.0, 1.0, 1.0).normalize();
    let rot = DMat3::from_axis_angle(axis, std::f64::consts::FRAC_PI_4);

    let rotated = i.transform(&rot);

    let original_trace = {
        let m = i.as_dmat3();
        m.col(0).x + m.col(1).y + m.col(2).z
    };
    let rotated_trace = {
        let m = rotated.as_dmat3();
        m.col(0).x + m.col(1).y + m.col(2).z
    };
    assert!(
        (original_trace - rotated_trace).abs() < TOL,
        "trace not preserved: {} vs {}",
        original_trace,
        rotated_trace
    );
}

#[test]
fn rotation_round_trip_recovers_original() {
    let i = InertiaTensor::<RootInertial>::from_components(2.0, 3.0, 5.0, 0.1, 0.2, 0.3);
    let axis = DVec3::new(1.0, 0.5, -0.3).normalize();
    let rot = DMat3::from_axis_angle(axis, 0.7);
    let rot_inv = rot.transpose();

    let i_rotated = i.transform(&rot);
    let i_back = i_rotated.transform(&rot_inv);

    assert!(
        approx_mat_eq(i.as_dmat3(), i_back.as_dmat3(), 1e-13),
        "round trip failed:\n  orig = {:?}\n  back = {:?}",
        i.as_dmat3(),
        i_back.as_dmat3()
    );
}

#[test]
fn parallel_axis_addition_matches_steiner() {
    // Steiner: inertia about parent axis = inertia about CoM axis +
    //   m * ([d]² · I − d ⊗ d). Build the offset contribution by
    //   hand and verify InertiaTensor::add reproduces it.
    let m = 5.0;
    let d = DVec3::new(1.0, 2.0, -0.5);

    let d_sq = d.length_squared();
    let outer = DMat3::from_cols(
        DVec3::new(d.x * d.x, d.y * d.x, d.z * d.x),
        DVec3::new(d.x * d.y, d.y * d.y, d.z * d.y),
        DVec3::new(d.x * d.z, d.y * d.z, d.z * d.z),
    );
    let offset_contribution =
        InertiaTensor::<RootInertial>::from_dmat3_unchecked(m * (DMat3::IDENTITY * d_sq - outer));

    let i_com = InertiaTensor::<RootInertial>::from_principal(2.0, 3.0, 4.0);
    let i_about_parent = i_com + offset_contribution;

    // Spot check: the (0,0) entry should be Ixx + m·(dy² + dz²).
    let expected_xx = 2.0 + m * (d.y * d.y + d.z * d.z);
    assert!(
        (i_about_parent.as_dmat3().col(0).x - expected_xx).abs() < TOL,
        "xx component: got {}, expected {}",
        i_about_parent.as_dmat3().col(0).x,
        expected_xx
    );
}

#[test]
fn scalar_mul_is_distributive() {
    let a = InertiaTensor::<RootInertial>::from_principal(1.0, 2.0, 3.0);
    let b = InertiaTensor::<RootInertial>::from_principal(4.0, 5.0, 6.0);
    let lhs = (a + b) * 2.0;
    let rhs = a * 2.0 + b * 2.0;
    assert_eq!(lhs, rhs);
}

#[test]
fn neg_then_add_is_zero() {
    let i = InertiaTensor::<RootInertial>::from_components(1.0, 2.0, 3.0, 0.4, 0.5, 0.6);
    let z = i + (-i);
    assert_eq!(z, InertiaTensor::<RootInertial>::zero());
}

#[test]
fn default_is_zero() {
    let z: InertiaTensor<RootInertial> = Default::default();
    assert_eq!(z, InertiaTensor::<RootInertial>::zero());
}

// `InertiaTensor<Ecef> + InertiaTensor<RootInertial>` must NOT compile —
// frame mismatch is a type error. The compile-fail case is asserted
// via a `compile_fail` doctest on `InertiaTensor` itself (see
// `crates/jeod_quantities/src/inertia.rs`); this test confirms the
// same-frame happy path.
#[test]
fn add_within_same_frame_compiles() {
    let a = InertiaTensor::<Ecef>::from_principal(1.0, 1.0, 1.0);
    let b = InertiaTensor::<Ecef>::from_principal(2.0, 2.0, 2.0);
    let _sum = a + b;
}
