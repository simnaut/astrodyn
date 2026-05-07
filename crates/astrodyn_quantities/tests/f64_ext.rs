//! `F64Ext` unit-conversion round-trip tests.

use astrodyn_quantities::prelude::*;
use uom::si::{
    acceleration::meter_per_second_squared, angle::radian, angular_velocity::radian_per_second,
    area::square_meter, force::newton, length::meter, mass::kilogram, time::second,
    torque::newton_meter, velocity::meter_per_second,
};

#[test]
fn length_km_is_1000_m() {
    let x = 1.0.km();
    assert!((x.get::<meter>() - 1000.0).abs() < 1e-9);
}

#[test]
fn length_cm_is_0_01_m() {
    assert!((1.0.cm().get::<meter>() - 0.01).abs() < 1e-12);
}

#[test]
fn length_mm_is_0_001_m() {
    assert!((1.0.mm().get::<meter>() - 0.001).abs() < 1e-12);
}

#[test]
fn length_nmi_is_1852_m() {
    assert!((1.0.nmi().get::<meter>() - 1852.0).abs() < 1e-9);
}

#[test]
fn velocity_km_per_s_is_1000_m_per_s() {
    assert!((1.0.km_per_s().get::<meter_per_second>() - 1000.0).abs() < 1e-9);
}

#[test]
fn acceleration_standard_gravity_is_9_80665() {
    // Standard gravity: 9.80665 m/s² exactly.
    assert!((1.0.g().get::<meter_per_second_squared>() - 9.806_65).abs() < 1e-9);
}

#[test]
fn angle_deg_to_rad() {
    // 180° = π rad.
    assert!((180.0.deg().get::<radian>() - core::f64::consts::PI).abs() < 1e-12);
}

#[test]
fn angle_arcsec_to_rad() {
    // 1 arcsec = π / 648 000 rad ≈ 4.848e-6 rad.
    let expected = core::f64::consts::PI / 648_000.0;
    assert!((1.0.arcsec().get::<radian>() - expected).abs() < 1e-15);
}

#[test]
fn mass_kg_round_trip() {
    assert!((420_000.0.kg().get::<kilogram>() - 420_000.0).abs() < 1e-9);
}

#[test]
fn time_hours_to_seconds() {
    assert!((1.0.hours().get::<second>() - 3600.0).abs() < 1e-9);
}

#[test]
fn time_days_to_seconds() {
    assert!((1.0.days().get::<second>() - 86_400.0).abs() < 1e-9);
}

#[test]
fn time_weeks_is_7_days() {
    // weeks() is implemented as 7 × days.
    assert!((1.0.weeks().get::<second>() - 7.0 * 86_400.0).abs() < 1e-6);
}

#[test]
fn force_newton_round_trip() {
    assert!((100.0.n().get::<newton>() - 100.0).abs() < 1e-12);
}

#[test]
fn torque_newton_meter_round_trip() {
    assert!((5.0.n_m().get::<newton_meter>() - 5.0).abs() < 1e-12);
}

#[test]
fn angular_velocity_rpm_to_rad_per_s() {
    // 60 rpm = 1 rev/s = 2π rad/s.
    let expected = 2.0 * core::f64::consts::PI;
    assert!((60.0.rpm().get::<radian_per_second>() - expected).abs() < 1e-9);
}

#[test]
fn grav_param_km3_to_m3_per_s2() {
    // Earth μ ≈ 398 600 km³/s² = 3.986e14 m³/s². We build via .km3_per_s2().
    let mu: GravParam<Earth> = 398_600.0.km3_per_s2();
    let expected = 398_600.0 * 1.0e9;
    assert!((mu.value - expected).abs() < 1e3, "mu = {}", mu.value);
}

#[test]
fn grav_param_planet_pinned_factory_round_trip() {
    let mu = 3.986_004_418e14_f64.m3_per_s2_for::<Earth>();
    assert!((mu.value - 3.986_004_418e14).abs() < 1.0);
}

#[test]
fn specific_ang_mom_km2_per_s_is_1e6_m2_per_s() {
    let h = 1.0.km2_per_s();
    assert!((h.value - 1.0e6).abs() < 1e-6);
}

#[test]
fn specific_energy_km2_per_s2_is_1e6_m2_per_s2() {
    let eps = 1.0.km2_per_s2();
    assert!((eps.value - 1.0e6).abs() < 1e-6);
}

#[test]
fn mass_flow_rate_round_trip() {
    let mdot = 1.5.kg_per_s();
    assert!((mdot.value - 1.5).abs() < 1e-12);
}

#[test]
fn area_km2_is_1e6_m2() {
    assert!((1.0.km2().get::<square_meter>() - 1.0e6).abs() < 1e-9);
}

#[test]
fn ratio_percent_is_0_01() {
    use uom::si::ratio::ratio;
    assert!((1.0.percent().get::<ratio>() - 0.01).abs() < 1e-12);
}

#[test]
fn vec3_ext_m_at_produces_position_inertial() {
    let v = glam::DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>();
    // Compile-time check: assign to a typed slot.
    let p: Position<RootInertial> = v;
    assert_eq!(p.raw_si().x, 7_000_000.0);
}

#[test]
fn vec3_ext_km_at_scales_by_1000() {
    let v = glam::DVec3::new(7_000.0, 0.0, 0.0).km_at::<Ecef>();
    let p: Position<Ecef> = v;
    assert!((p.raw_si().x - 7_000_000.0).abs() < 1e-6);
}

#[test]
fn array3_ext_km_at_scales_by_1000() {
    let v: Position<RootInertial> = [7000.0, 0.0, 0.0].km_at::<RootInertial>();
    assert!((v.raw_si().x - 7_000_000.0).abs() < 1e-6);
}
