//! Integration tests for `Qty3` arithmetic across dimensions and frames.

use glam::DVec3;
use jeod_quantities::prelude::*;
use uom::si::{
    f64::{Length, Time, Velocity as ScalarVelocity},
    length::{kilometer, meter},
    time::second,
    velocity::meter_per_second,
};

// ---- helpers ----

fn pos_inertial(x: f64, y: f64, z: f64) -> Position<RootInertial> {
    Position::<RootInertial>::new(
        Length::new::<meter>(x),
        Length::new::<meter>(y),
        Length::new::<meter>(z),
    )
}

fn pos_ecef(x: f64, y: f64, z: f64) -> Position<Ecef> {
    Position::<Ecef>::new(
        Length::new::<meter>(x),
        Length::new::<meter>(y),
        Length::new::<meter>(z),
    )
}

fn vel_inertial(x: f64, y: f64, z: f64) -> jeod_quantities::Velocity<RootInertial> {
    Qty3::<_, RootInertial>::new(
        ScalarVelocity::new::<meter_per_second>(x),
        ScalarVelocity::new::<meter_per_second>(y),
        ScalarVelocity::new::<meter_per_second>(z),
    )
}

// ---- Add / Sub / Neg ----

#[test]
fn position_add_same_frame() {
    let a = pos_inertial(1.0, 2.0, 3.0);
    let b = pos_inertial(10.0, 20.0, 30.0);
    let s = a + b;
    assert_eq!(s.raw_si(), DVec3::new(11.0, 22.0, 33.0));
}

#[test]
fn position_sub_same_frame() {
    let a = pos_inertial(10.0, 10.0, 10.0);
    let b = pos_inertial(1.0, 2.0, 3.0);
    let d = a - b;
    assert_eq!(d.raw_si(), DVec3::new(9.0, 8.0, 7.0));
}

#[test]
fn position_neg() {
    let a = pos_inertial(1.0, -2.0, 3.0);
    assert_eq!((-a).raw_si(), DVec3::new(-1.0, 2.0, -3.0));
}

#[test]
fn velocity_add_same_frame() {
    let a = vel_inertial(100.0, 0.0, 0.0);
    let b = vel_inertial(0.0, 50.0, 0.0);
    assert_eq!((a + b).raw_si(), DVec3::new(100.0, 50.0, 0.0));
}

#[test]
fn zero_vector() {
    let z: Position<RootInertial> = Qty3::zero();
    assert_eq!(z.raw_si(), DVec3::ZERO);
}

// ---- Scalar multiply / divide ----

#[test]
fn scalar_multiply_position() {
    let a = pos_inertial(1.0, 2.0, 3.0);
    assert_eq!((a * 10.0).raw_si(), DVec3::new(10.0, 20.0, 30.0));
}

#[test]
fn scalar_divide_position() {
    let a = pos_inertial(10.0, 20.0, 30.0);
    assert_eq!((a / 5.0).raw_si(), DVec3::new(2.0, 4.0, 6.0));
}

#[test]
fn scalar_multiply_preserves_frame() {
    let a = pos_ecef(2.0, 4.0, 6.0);
    let b = a * 3.0;
    // Compile-time proof it's still Ecef: assign to typed slot.
    let _: Position<Ecef> = b;
    assert_eq!(b.raw_si(), DVec3::new(6.0, 12.0, 18.0));
}

// ---- Cross-dimension multiply: Velocity × Time → Position ----

#[test]
fn velocity_times_time_yields_position_like() {
    let v = vel_inertial(10.0, 0.0, 0.0);
    let t = Time::new::<second>(5.0);
    let r = v * t; // Qty3<velocity * time, RootInertial> = Qty3<length, RootInertial>
    assert_eq!(r.raw_si(), DVec3::new(50.0, 0.0, 0.0));
}

// ---- .magnitude() ----

#[test]
fn magnitude_345() {
    let a = pos_inertial(3.0, 4.0, 0.0);
    assert!((a.magnitude().value - 5.0).abs() < 1e-12);
}

#[test]
fn magnitude_unit_axes() {
    let x = pos_inertial(1.0, 0.0, 0.0);
    let y = pos_inertial(0.0, 1.0, 0.0);
    let z = pos_inertial(0.0, 0.0, 1.0);
    for v in [x, y, z] {
        assert!((v.magnitude().value - 1.0).abs() < 1e-12);
    }
}

#[test]
fn magnitude_kilometer_input() {
    // Build by km unit and ensure magnitude reports meters.
    let a = Position::<RootInertial>::new(
        Length::new::<kilometer>(3.0),
        Length::new::<kilometer>(4.0),
        Length::new::<kilometer>(0.0),
    );
    assert!((a.magnitude().value - 5000.0).abs() < 1e-9);
}

// ---- Frame is a compile-time tag: different frames are distinct types ----

#[test]
fn different_frames_are_distinct_types() {
    let i = pos_inertial(1.0, 0.0, 0.0);
    let e = pos_ecef(1.0, 0.0, 0.0);
    // Values match, types differ:
    assert_eq!(i.raw_si(), e.raw_si());
    // The line below, if uncommented, must fail to compile:
    //   let _ = i + e;
}

// ---- Layout static asserts: Qty3 is just three f64s ----

#[test]
fn qty3_size_and_align_match_vec3() {
    use core::mem::{align_of, size_of};
    assert_eq!(size_of::<Position<RootInertial>>(), size_of::<DVec3>());
    assert_eq!(size_of::<Position<RootInertial>>(), 24);
    assert_eq!(align_of::<Position<RootInertial>>(), 8);
    assert_eq!(size_of::<jeod_quantities::Velocity<Ecef>>(), 24);
}

#[test]
fn raw_si_round_trip() {
    let v = DVec3::new(1.23, -4.56, 7.89);
    let p: Position<RootInertial> = Qty3::from_raw_si(v);
    assert_eq!(p.raw_si(), v);
}

// ---- More dimension/frame combos to hit the >=25 Qty3-op target ----

#[test]
fn add_in_ecef() {
    let a = pos_ecef(1.0, 2.0, 3.0);
    let b = pos_ecef(4.0, 5.0, 6.0);
    assert_eq!((a + b).raw_si(), DVec3::new(5.0, 7.0, 9.0));
}

#[test]
fn mul_by_negative_scalar() {
    let a = pos_inertial(1.0, 1.0, 1.0);
    assert_eq!((a * -2.0).raw_si(), DVec3::new(-2.0, -2.0, -2.0));
}

#[test]
fn div_by_unity_is_identity() {
    let a = pos_inertial(3.0, 4.0, 5.0);
    assert_eq!((a / 1.0).raw_si(), a.raw_si());
}

#[test]
fn double_neg_is_identity() {
    let a = pos_inertial(7.0, -8.0, 9.0);
    assert_eq!((-(-a)).raw_si(), a.raw_si());
}

#[test]
fn debug_format_mentions_frame() {
    let a = pos_ecef(1.0, 2.0, 3.0);
    let rendered = format!("{a:?}");
    assert!(rendered.contains("Ecef"), "got: {rendered}");
    // Make sure the debug output is also distinct for RootInertial — we use
    // `type_name` so the fully-qualified path appears, which comfortably
    // differs between `Ecef` and `RootInertial`.
    let b = pos_inertial(1.0, 2.0, 3.0);
    let rendered_b = format!("{b:?}");
    assert!(rendered_b.contains("RootInertial"), "got: {rendered_b}");
    assert_ne!(rendered, rendered_b);
}

#[test]
fn partial_eq_reflects_raw_si() {
    let a = pos_inertial(1.0, 2.0, 3.0);
    let b = pos_inertial(1.0, 2.0, 3.0);
    let c = pos_inertial(1.0, 2.0, 3.5);
    assert!(a == b);
    assert!(a != c);
}

#[test]
fn vel_plus_vel_in_ecef() {
    let a = Qty3::<_, Ecef>::new(
        ScalarVelocity::new::<meter_per_second>(1.0),
        ScalarVelocity::new::<meter_per_second>(0.0),
        ScalarVelocity::new::<meter_per_second>(0.0),
    );
    let b = Qty3::<_, Ecef>::new(
        ScalarVelocity::new::<meter_per_second>(0.0),
        ScalarVelocity::new::<meter_per_second>(1.0),
        ScalarVelocity::new::<meter_per_second>(0.0),
    );
    assert_eq!((a + b).raw_si(), DVec3::new(1.0, 1.0, 0.0));
}

#[test]
fn acceleration_times_time_is_velocity_like() {
    use uom::si::{acceleration, f64::Acceleration as ScalarAccel};
    let a = Qty3::<acceleration::Dimension, RootInertial>::new(
        ScalarAccel::new::<uom::si::acceleration::meter_per_second_squared>(9.81),
        ScalarAccel::new::<uom::si::acceleration::meter_per_second_squared>(0.0),
        ScalarAccel::new::<uom::si::acceleration::meter_per_second_squared>(0.0),
    );
    let t = Time::new::<second>(2.0);
    let v = a * t;
    assert!((v.raw_si().x - 19.62).abs() < 1e-9);
}

#[test]
fn magnitude_is_translation_invariant_under_negation() {
    let a = pos_inertial(3.0, 4.0, 12.0);
    assert!((a.magnitude().value - (-a).magnitude().value).abs() < 1e-12);
    assert!((a.magnitude().value - 13.0).abs() < 1e-12);
}

#[test]
fn from_raw_si_negative_components() {
    let v = DVec3::new(-1.0, -2.0, -3.0);
    let p: Position<RootInertial> = Qty3::from_raw_si(v);
    assert_eq!(p.raw_si(), v);
}

#[test]
fn mul_then_div_round_trip() {
    let a = pos_inertial(1.5, 2.5, 3.5);
    let b = (a * 7.0) / 7.0;
    let diff = (a.raw_si() - b.raw_si()).length();
    assert!(diff < 1e-12, "diff = {diff}");
}

#[test]
fn scalar_multiply_by_zero_is_zero() {
    let a = pos_inertial(99.0, -99.0, 99.0);
    assert_eq!((a * 0.0).raw_si(), DVec3::ZERO);
}

#[test]
fn add_zero_is_identity() {
    let a = pos_inertial(1.0, 2.0, 3.0);
    let z: Position<RootInertial> = Qty3::zero();
    assert_eq!((a + z).raw_si(), a.raw_si());
}

#[test]
fn sub_self_is_zero() {
    let a = pos_inertial(1.0, 2.0, 3.0);
    let z: Position<RootInertial> = Qty3::zero();
    assert_eq!((a - a).raw_si(), z.raw_si());
}

// ---- dot / cross / Div<Quantity> ----

#[test]
fn position_dot_position_is_area_magnitude() {
    use uom::si::area::square_meter;
    let a = pos_inertial(3.0, 0.0, 0.0);
    let b = pos_inertial(4.0, 0.0, 0.0);
    let d = a.dot(&b);
    // Output dimension is length², i.e. area; value in m² is 12.
    assert!((d.get::<square_meter>() - 12.0).abs() < 1e-12);
}

#[test]
fn position_cross_position_is_zero_for_colinear() {
    let a = pos_inertial(1.0, 0.0, 0.0);
    let b = pos_inertial(2.0, 0.0, 0.0);
    let c = a.cross(&b);
    assert_eq!(c.raw_si(), DVec3::ZERO);
}

#[test]
fn cross_perpendicular_vectors_is_unit_area() {
    let x = pos_inertial(1.0, 0.0, 0.0);
    let y = pos_inertial(0.0, 1.0, 0.0);
    let z = x.cross(&y);
    // Cross of unit length × unit length along perpendicular axes is (0,0,1) m².
    assert_eq!(z.raw_si(), DVec3::new(0.0, 0.0, 1.0));
}

#[test]
fn qty3_div_time_is_velocity_like() {
    let p = pos_inertial(100.0, 0.0, 0.0);
    let t = uom::si::f64::Time::new::<uom::si::time::second>(10.0);
    let v = p / t;
    assert_eq!(v.raw_si(), DVec3::new(10.0, 0.0, 0.0));
}
