//! JEOD `PlanetFixedPosition` verification cases.
//!
//! Parses the three explicit test points defined in
//! `models/utils/planet_fixed/planet_fixed_posn/verif/SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py`.
//!
//! The JEOD SIM exercises [`PlanetFixedPosition`] using the WGS84 default
//! Earth shape (`r_eq = 6378.137 km`, `flat_inv = 298.257223563`, set in
//! `environment/planet/data/include/earth.hh`). It triggers three
//! `add_read` snapshots:
//!
//! 1. `update_from_cart` with a Cartesian PCPF position.
//! 2. `update_from_spher` with a spherical altitude/latitude/longitude.
//! 3. `update_from_ellip` with an elliptical altitude/latitude/longitude.
//!
//! JEOD's verification methodology for these conversions is round-trip
//! closure (see `verif/unit_tests/Cartesian_to_AltLatLong_to_Cartesian/main.cc`,
//! which sweeps random vectors and checks `cart -> spher -> cart` and
//! `cart -> ellip -> cart` magnitude errors). We mirror that methodology
//! against the SIM's three explicit input points so the test inputs are
//! sourced from JEOD itself (per CLAUDE.md "Tier 3 Cross-Validation"
//! guidance: initial conditions may come from JEOD source files).

use glam::DVec3;
use regex::Regex;

/// One verification case from `SIM_PFIXPOSN_VERIF`.
///
/// Each variant carries the `add_read` time tag from `input.py` for
/// traceability, plus the seed coordinates used to drive the conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanetFixedSeed {
    /// `update_from_cart`: Cartesian PCPF position (m).
    Cartesian { read_time: f64, cart_m: DVec3 },
    /// `update_from_spher`: Spherical altitude (m), latitude (rad),
    /// longitude (rad).
    Spherical {
        read_time: f64,
        altitude_m: f64,
        latitude_rad: f64,
        longitude_rad: f64,
    },
    /// `update_from_ellip`: Elliptical (geodetic) altitude (m),
    /// latitude (rad), longitude (rad).
    Elliptical {
        read_time: f64,
        altitude_m: f64,
        latitude_rad: f64,
        longitude_rad: f64,
    },
}

/// Load all three `SIM_PFIXPOSN_VERIF` seed cases from JEOD source.
///
/// # Panics
/// Panics if the `input.py` cannot be read or fewer than three cases parse.
pub fn load_planet_fixed_verif_cases(jeod_root: &std::path::Path) -> Vec<PlanetFixedSeed> {
    let path = jeod_root.join(
        "models/utils/planet_fixed/planet_fixed_posn/\
         verif/SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py",
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    // Each `add_read` block is structured like:
    //   read = <time>
    //   trick.add_read(read, """
    //   <body>
    //   """)
    // We capture the time and the body and then sniff the body for which
    // update_from_* call lives inside it.
    let block_re = Regex::new(
        r#"(?ms)read\s*=\s*([0-9eE+.\-]+)\s*\n\s*trick\.add_read\s*\(\s*read\s*,\s*"""(?P<body>.*?)"""\s*\)"#,
    )
    .expect("regex for add_read block");

    let mut cases = Vec::new();
    for caps in block_re.captures_iter(&content) {
        let read_time: f64 = caps[1].parse().unwrap_or_else(|e| {
            panic!("malformed read time {:?}: {e}", &caps[1]);
        });
        let body = &caps["body"];

        if body.contains("update_from_cart") {
            let xyz = parse_array3(body, r"earth\.cartesian_pos\s*=\s*\[([^\]]+)\]")
                .unwrap_or_else(|| panic!("cartesian_pos not parseable in:\n{body}"));
            cases.push(PlanetFixedSeed::Cartesian {
                read_time,
                cart_m: DVec3::new(xyz[0], xyz[1], xyz[2]),
            });
        } else if body.contains("update_from_spher") {
            let alt = parse_assign(body, r"earth\.spherical_pos\.altitude\s*=\s*([\-0-9eE+.]+)");
            let lat = parse_assign(body, r"earth\.spherical_pos\.latitude\s*=\s*([\-0-9eE+.]+)");
            let lon = parse_assign(
                body,
                r"earth\.spherical_pos\.longitude\s*=\s*([\-0-9eE+.]+)",
            );
            cases.push(PlanetFixedSeed::Spherical {
                read_time,
                altitude_m: alt,
                latitude_rad: lat,
                longitude_rad: lon,
            });
        } else if body.contains("update_from_ellip") {
            let alt = parse_assign(
                body,
                r"earth\.elliptical_pos\.altitude\s*=\s*([\-0-9eE+.]+)",
            );
            let lat = parse_assign(
                body,
                r"earth\.elliptical_pos\.latitude\s*=\s*([\-0-9eE+.]+)",
            );
            let lon = parse_assign(
                body,
                r"earth\.elliptical_pos\.longitude\s*=\s*([\-0-9eE+.]+)",
            );
            cases.push(PlanetFixedSeed::Elliptical {
                read_time,
                altitude_m: alt,
                latitude_rad: lat,
                longitude_rad: lon,
            });
        }
    }

    assert!(
        cases.len() >= 3,
        "expected three cases in {}, found {}",
        path.display(),
        cases.len()
    );
    cases
}

fn parse_array3(text: &str, pattern: &str) -> Option<[f64; 3]> {
    let re = Regex::new(pattern).ok()?;
    let caps = re.captures(text)?;
    let parts: Vec<f64> = caps[1]
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if parts.len() != 3 {
        return None;
    }
    Some([parts[0], parts[1], parts[2]])
}

fn parse_assign(text: &str, pattern: &str) -> f64 {
    let re = Regex::new(pattern).expect("valid regex");
    let caps = re
        .captures(text)
        .unwrap_or_else(|| panic!("pattern {pattern:?} did not match in:\n{text}"));
    caps[1]
        .parse()
        .unwrap_or_else(|e| panic!("failed to parse {:?}: {e}", &caps[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jeod_path;

    #[test]
    fn loads_three_cases_from_jeod_source() {
        let root = jeod_path();
        if !root.exists() {
            // Loader is exercised by the Tier 2 test harness; this guard
            // mirrors the convention used by other jeod_test_data loaders.
            return;
        }
        let cases = load_planet_fixed_verif_cases(&root);
        assert_eq!(cases.len(), 3, "SIM_PFIXPOSN_VERIF defines three reads");

        // Sanity-check the first case (the Cartesian seed at ISS-ish radius).
        match cases[0] {
            PlanetFixedSeed::Cartesian { cart_m, .. } => {
                assert!((cart_m.x - 6_778_136.3).abs() < 1e-6);
                assert_eq!(cart_m.y, 0.0);
                assert_eq!(cart_m.z, 0.0);
            }
            _ => panic!("first case must be Cartesian: {:?}", cases[0]),
        }
    }
}
