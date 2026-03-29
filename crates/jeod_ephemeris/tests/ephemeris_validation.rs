//! Ephemeris validation tests using DE421.bsp (same kernel as JEOD).
//!
//! Requires `test_data/de421.bsp` to be present. Download with:
//!   curl -Lo test_data/de421.bsp https://public-data.nyxspace.com/anise/de421.bsp

use jeod_ephemeris::{Ephemeris, EphemerisBody};
use std::path::Path;

fn load_de421() -> Ephemeris {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/de421.bsp");
    assert!(
        path.exists(),
        "DE421.bsp not found at {}. Download with: curl -Lo test_data/de421.bsp https://public-data.nyxspace.com/anise/de421.bsp",
        path.display()
    );
    Ephemeris::from_bsp(&path).expect("Failed to load DE421.bsp")
}

/// J2000.0 TDB Julian Date = 2451545.0
const J2000_TDB_JD: f64 = 2_451_545.0;

/// Earth-Moon distance at J2000.0 should be approximately 402,449 km.
/// Tolerance: < 1 km (per PLAN.md exit criterion).
#[test]
fn earth_moon_distance_at_j2000() {
    let ephem = load_de421();
    let (pos_m, _vel) = ephem
        .get_earth_centered_state(EphemerisBody::Moon, J2000_TDB_JD)
        .expect("Failed to query Moon state");

    let distance_km = pos_m.length() / 1000.0;
    eprintln!("  Earth-Moon distance at J2000: {:.1} km", distance_km);

    // JPL Horizons: ~402,448.6 km
    assert!(
        (distance_km - 402_449.0).abs() < 1.0,
        "Earth-Moon distance at J2000: {:.1} km, expected ~402,449 km (tolerance < 1 km)",
        distance_km,
    );
}

/// Sun direction at vernal equinox (2000-03-20 ~07:35 UTC).
/// The Sun should be near RA = 0°, Dec = 0° (crossing the equatorial plane).
/// Tolerance: < 0.01° in RA (per PLAN.md exit criterion).
#[test]
fn sun_direction_at_vernal_equinox_2000() {
    let ephem = load_de421();

    // Vernal equinox 2000: 2000-03-20 07:35 UTC ≈ JD 2451623.816
    let equinox_jd = 2_451_623.816;

    let (pos_m, _vel) = ephem
        .get_earth_centered_state(EphemerisBody::Sun, equinox_jd)
        .expect("Failed to query Sun state");

    // Right ascension = atan2(y, x)
    let ra_rad = pos_m.y.atan2(pos_m.x);
    let ra_deg = ra_rad.to_degrees();

    // At the vernal equinox, the Sun's RA should be near 0° (or 360°).
    // Handle wrapping: RA could be slightly negative.
    let ra_deg_wrapped = if ra_deg < -180.0 {
        ra_deg + 360.0
    } else if ra_deg > 180.0 {
        ra_deg - 360.0
    } else {
        ra_deg
    };

    eprintln!("  Sun RA at vernal equinox 2000: {:.4}°", ra_deg_wrapped);

    assert!(
        ra_deg_wrapped.abs() < 0.01,
        "Sun RA at equinox: {:.4}° (expected near 0°, tolerance < 0.01°)",
        ra_deg_wrapped,
    );
}

/// Basic sanity: Sun-Earth distance at J2000 should be ~1 AU (149.6 million km).
#[test]
fn sun_earth_distance_at_j2000() {
    let ephem = load_de421();
    let (pos_m, _vel) = ephem
        .get_earth_centered_state(EphemerisBody::Sun, J2000_TDB_JD)
        .expect("Failed to query Sun state");

    let distance_au = pos_m.length() / 1.496e11;
    eprintln!("  Sun-Earth distance at J2000: {:.4} AU", distance_au);

    // At J2000.0 (Jan 1.5, 2000) Earth is near perihelion, so the actual
    // distance is ~0.9833 AU, not 1.0 AU. Use the physical value.
    assert!(
        (distance_au - 0.9833).abs() < 0.005,
        "Sun-Earth distance: {:.4} AU (expected ~0.9833 AU at perihelion, tolerance < 0.005 AU)",
        distance_au,
    );
}

/// Moon orbital velocity at J2000 should be ~1.02 km/s relative to Earth.
#[test]
fn moon_velocity_at_j2000() {
    let ephem = load_de421();
    let (_pos, vel) = ephem
        .get_earth_centered_state(EphemerisBody::Moon, J2000_TDB_JD)
        .expect("Failed to query Moon state");

    let speed_km_s = vel.length() / 1000.0;
    eprintln!("  Moon velocity at J2000: {:.4} km/s", speed_km_s);

    assert!(
        (speed_km_s - 1.02).abs() < 0.1,
        "Moon velocity at J2000: {:.4} km/s, expected ~1.02 km/s (tolerance 0.1 km/s)",
        speed_km_s,
    );
}

/// Mars distance from Earth at J2000 should be ~1.4-2.7 AU.
#[test]
fn mars_distance_at_j2000() {
    let ephem = load_de421();
    let (pos_m, _vel) = ephem
        .get_earth_centered_state(EphemerisBody::Mars, J2000_TDB_JD)
        .expect("Failed to query Mars state");

    let distance_au = pos_m.length() / 1.496e11;
    eprintln!("  Earth-Mars distance at J2000: {:.4} AU", distance_au);

    assert!(
        distance_au > 1.4 && distance_au < 2.7,
        "Earth-Mars distance at J2000: {:.4} AU, expected 1.4-2.7 AU",
        distance_au,
    );
}

/// Sun velocity relative to Earth at J2000 should be ~30 km/s.
#[test]
fn sun_velocity_at_j2000() {
    let ephem = load_de421();
    let (_pos, vel) = ephem
        .get_earth_centered_state(EphemerisBody::Sun, J2000_TDB_JD)
        .expect("Failed to query Sun state");

    let speed_km_s = vel.length() / 1000.0;
    eprintln!("  Sun velocity at J2000: {:.4} km/s", speed_km_s);

    assert!(
        (speed_km_s - 30.0).abs() < 5.0,
        "Sun velocity at J2000: {:.4} km/s, expected ~30 km/s (tolerance 5 km/s)",
        speed_km_s,
    );
}
