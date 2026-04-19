//! `FrameTransform` composition and round-trip tests.

use glam::DVec3;
use jeod_quantities::prelude::*;

// A sample vehicle tag for body-frame tests. Define it locally in a test-only
// module using the sealed-trait pattern is not possible (Sealed is private),
// so frame composition tests use pairs of already-defined frames.

#[test]
fn identity_apply_is_identity() {
    let t: FrameTransform<Inertial, Inertial> = FrameTransform::<Inertial, Inertial>::identity();
    let p: Position<Inertial> = Qty3::from_raw_si(DVec3::new(1.0, 2.0, 3.0));
    let p2 = t.apply(p);
    assert_eq!(p.raw_si(), p2.raw_si());
}

#[test]
fn inverse_of_inverse_is_identity() {
    // Rotation by 45° about Z: non-trivial, tests the sign/convention stuff.
    let c = (core::f64::consts::FRAC_PI_4 / 2.0).cos(); // cos(θ/2) with θ = 45°
    let s = (core::f64::consts::FRAC_PI_4 / 2.0).sin();
    let q = NormalizedQuat::new(JeodQuat::from_array([c, 0.0, 0.0, s]))
        .expect("normalized rotation quaternion");
    let t: FrameTransform<Inertial, Ecef> = FrameTransform::from_quat(q);
    let back = t.inverse().inverse();

    let p: Position<Inertial> = Qty3::from_raw_si(DVec3::new(1.0, 0.0, 0.0));
    let rotated_once = t.apply(p);
    let rotated_twice = back.apply(p);
    assert!((rotated_once.raw_si() - rotated_twice.raw_si()).length() < 1e-12);
}

#[test]
fn apply_then_inverse_returns_original() {
    let c = (core::f64::consts::FRAC_PI_4 / 2.0).cos();
    let s = (core::f64::consts::FRAC_PI_4 / 2.0).sin();
    let q = NormalizedQuat::new(JeodQuat::from_array([c, 0.0, 0.0, s])).unwrap();
    let t: FrameTransform<Inertial, Ecef> = FrameTransform::from_quat(q);

    let p: Position<Inertial> = Qty3::from_raw_si(DVec3::new(1.0, 2.0, 3.0));
    let in_ecef = t.apply(p);
    let back = t.inverse().apply(in_ecef);
    let diff = (p.raw_si() - back.raw_si()).length();
    assert!(diff < 1e-12, "round-trip drift = {diff}");
}

#[test]
fn composition_round_trip() {
    // Use three *distinct* frame pairs so the type checker verifies composition.
    let q1 = NormalizedQuat::new(JeodQuat::from_array([
        (core::f64::consts::FRAC_PI_4 / 2.0).cos(),
        0.0,
        0.0,
        (core::f64::consts::FRAC_PI_4 / 2.0).sin(),
    ]))
    .unwrap();
    let q2_inv = NormalizedQuat::new(JeodQuat::from_array([
        (core::f64::consts::FRAC_PI_4 / 2.0).cos(),
        0.0,
        0.0,
        -(core::f64::consts::FRAC_PI_4 / 2.0).sin(),
    ]))
    .unwrap();
    let t_i_to_e: FrameTransform<Inertial, Ecef> = FrameTransform::from_quat(q1);
    let t_e_to_i: FrameTransform<Ecef, Inertial> = FrameTransform::from_quat(q2_inv);
    let composed: FrameTransform<Inertial, Inertial> = t_i_to_e * t_e_to_i;

    let p: Position<Inertial> = Qty3::from_raw_si(DVec3::new(5.0, -3.0, 1.0));
    let out = composed.apply(p);
    let diff = (p.raw_si() - out.raw_si()).length();
    assert!(diff < 1e-10, "compose(A→B, B→A).apply(p) drift = {diff}");
}

#[test]
fn identity_composition_does_not_change_frame() {
    let id_i: FrameTransform<Inertial, Inertial> = FrameTransform::<Inertial, Inertial>::identity();
    let composed: FrameTransform<Inertial, Inertial> = id_i * id_i;
    let p: Position<Inertial> = Qty3::from_raw_si(DVec3::new(1.0, 2.0, 3.0));
    assert_eq!(composed.apply(p).raw_si(), p.raw_si());
}

#[test]
fn matrix_cache_consistent_with_quat_rotation() {
    let q = NormalizedQuat::new(JeodQuat::from_array([
        (core::f64::consts::FRAC_PI_4 / 2.0).cos(),
        0.0,
        0.0,
        (core::f64::consts::FRAC_PI_4 / 2.0).sin(),
    ]))
    .unwrap();
    let t: FrameTransform<Inertial, Ecef> = FrameTransform::from_quat(q);

    let p = DVec3::new(1.0, 0.0, 0.0);
    let via_matrix = t.matrix() * p;

    let p_typed: Position<Inertial> = Qty3::from_raw_si(p);
    let via_apply = t.apply(p_typed).raw_si();

    let diff = (via_matrix - via_apply).length();
    assert!(diff < 1e-12, "matrix vs apply drift = {diff}");
}
