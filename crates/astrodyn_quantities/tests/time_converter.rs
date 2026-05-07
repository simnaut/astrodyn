//! `TimeConverter` round-trip tests for the two constant-offset scale pairs
//! exposed by this crate (TAI↔TT and GPS↔TAI).

use astrodyn_quantities::prelude::*;
use astrodyn_quantities::time_scale::{GPS_TO_TAI_OFFSET_S, TAI_TO_TT_OFFSET_S};

#[test]
fn tai_to_tt_adds_32_184() {
    let tai = SecondsSince::<TAI>::from_seconds(1000.0);
    let tt = TimeConverter::<TAI, TT>::TAI_TO_TT.apply(tai);
    assert!((tt.as_seconds() - (1000.0 + TAI_TO_TT_OFFSET_S)).abs() < 1e-12);
}

#[test]
fn tt_to_tai_subtracts_32_184() {
    let tt = SecondsSince::<TT>::from_seconds(1000.0);
    let tai = TimeConverter::<TT, TAI>::TT_TO_TAI.apply(tt);
    assert!((tai.as_seconds() - (1000.0 - TAI_TO_TT_OFFSET_S)).abs() < 1e-12);
}

#[test]
fn tai_to_tt_to_tai_round_trip() {
    let tai0 = SecondsSince::<TAI>::from_seconds(123_456.789);
    let tt = TimeConverter::<TAI, TT>::TAI_TO_TT.apply(tai0);
    let tai1 = TimeConverter::<TT, TAI>::TT_TO_TAI.apply(tt);
    assert!((tai0.as_seconds() - tai1.as_seconds()).abs() < 1e-9);
}

#[test]
fn gps_to_tai_adds_19() {
    let gps = SecondsSince::<GPS>::from_seconds(1000.0);
    let tai = TimeConverter::<GPS, TAI>::GPS_TO_TAI.apply(gps);
    assert!((tai.as_seconds() - (1000.0 + GPS_TO_TAI_OFFSET_S)).abs() < 1e-12);
}

#[test]
fn tai_to_gps_subtracts_19() {
    let tai = SecondsSince::<TAI>::from_seconds(1000.0);
    let gps = TimeConverter::<TAI, GPS>::TAI_TO_GPS.apply(tai);
    assert!((gps.as_seconds() - (1000.0 - GPS_TO_TAI_OFFSET_S)).abs() < 1e-12);
}

#[test]
fn gps_to_tai_round_trip() {
    let gps0 = SecondsSince::<GPS>::from_seconds(987_654.321);
    let tai = TimeConverter::<GPS, TAI>::GPS_TO_TAI.apply(gps0);
    let gps1 = TimeConverter::<TAI, GPS>::TAI_TO_GPS.apply(tai);
    assert!((gps0.as_seconds() - gps1.as_seconds()).abs() < 1e-9);
}

#[test]
fn inverse_flips_offset_sign() {
    let c = TimeConverter::<TAI, TT>::TAI_TO_TT;
    let c_inv = c.inverse();
    assert!((c.offset_seconds() + c_inv.offset_seconds()).abs() < 1e-12);
}
