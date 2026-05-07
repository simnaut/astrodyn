//! JEOD `PlanetFixedPosition` verification seeds (committed JSON).
//!
//! The three explicit test points originate in JEOD's
//! `models/utils/planet_fixed/planet_fixed_posn/verif/SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py`,
//! which exercises `PlanetFixedPosition` against the WGS84 default Earth
//! shape (`r_eq = 6378.137 km`, `flat_inv = 298.257223563`, set in
//! `environment/planet/data/include/earth.hh`). The SIM triggers three
//! `add_read` snapshots:
//!
//! 1. `update_from_cart` with a Cartesian PCPF position.
//! 2. `update_from_spher` with a spherical altitude/latitude/longitude.
//! 3. `update_from_ellip` with an elliptical altitude/latitude/longitude.
//!
//! The seeds were extracted once, committed at `test_data/planet_pfixposn_seeds.json`,
//! and read back here without touching JEOD source. Regenerate with
//! `cargo run -p astrodyn_planet --bin extract_planet_pfixposn -- --jeod-home $JEOD_HOME`.
//!
//! JEOD's verification methodology for these conversions is round-trip
//! closure (see `verif/unit_tests/Cartesian_to_AltLatLong_to_Cartesian/main.cc`,
//! which sweeps random vectors and checks `cart -> spher -> cart` and
//! `cart -> ellip -> cart` magnitude errors). We mirror that methodology
//! against the SIM's three explicit input points so the test inputs are
//! sourced from JEOD itself.

use glam::DVec3;

/// One verification case from `SIM_PFIXPOSN_VERIF`.
///
/// Each variant carries the `add_read` time tag from `input.py` for
/// traceability, plus the seed coordinates used to drive the conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanetFixedSeed {
    /// `update_from_cart`: Cartesian PCPF position (m).
    Cartesian {
        /// `add_read` time tag from the JEOD `input.py`.
        read_time: f64,
        /// Seed Cartesian PCPF position in metres.
        cart_m: DVec3,
    },
    /// `update_from_spher`: Spherical altitude (m), latitude (rad),
    /// longitude (rad).
    Spherical {
        /// `add_read` time tag from the JEOD `input.py`.
        read_time: f64,
        /// Spherical altitude above the mean equatorial radius, in metres.
        altitude_m: f64,
        /// Geocentric latitude in radians.
        latitude_rad: f64,
        /// Longitude in radians.
        longitude_rad: f64,
    },
    /// `update_from_ellip`: Elliptical (geodetic) altitude (m),
    /// latitude (rad), longitude (rad).
    Elliptical {
        /// `add_read` time tag from the JEOD `input.py`.
        read_time: f64,
        /// Geodetic altitude above the reference ellipsoid, in metres.
        altitude_m: f64,
        /// Geodetic latitude in radians.
        latitude_rad: f64,
        /// Longitude in radians.
        longitude_rad: f64,
    },
}

/// Load the three `SIM_PFIXPOSN_VERIF` seed cases from the committed
/// `test_data/planet_pfixposn_seeds.json`.
///
/// # Panics
/// Panics if the JSON file is missing or malformed; the panic names the
/// regen command per CLAUDE.md "Fail Loudly".
pub fn load_planet_fixed_verif_cases() -> Vec<PlanetFixedSeed> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data/planet_pfixposn_seeds.json");
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read {}: {e}. Regenerate with `cargo run -p astrodyn_planet \
             --bin extract_planet_pfixposn -- --jeod-home $JEOD_HOME`.",
            path.display(),
        )
    });
    parse_seeds_json(&content).unwrap_or_else(|e| {
        panic!(
            "Malformed seeds in {}: {e}. Regenerate with `cargo run -p astrodyn_verif_jeod \
             --bin extract_planet_pfixposn -- --jeod-home $JEOD_HOME`.",
            path.display(),
        )
    })
}

/// Hand-rolled JSON parser for the `cases` array. Mirrors the
/// no-`serde_json` style used by `tier3_baseline_diff` and `tier3_report`.
fn parse_seeds_json(s: &str) -> Result<Vec<PlanetFixedSeed>, String> {
    let cases_idx = s
        .find("\"cases\"")
        .ok_or_else(|| "missing top-level \"cases\" key".to_string())?;
    let after = &s[cases_idx..];
    let arr_start = after
        .find('[')
        .ok_or_else(|| "no opening `[` after \"cases\"".to_string())?;
    let bytes = after.as_bytes();
    // Walk to the matching `]` tracking depth so commas inside nested
    // objects/arrays do not split entries.
    let mut depth = 1_i32;
    let mut i = arr_start + 1;
    let mut entry_starts = vec![i];
    let mut entries: Vec<&str> = Vec::new();
    let mut in_string = false;
    while i < bytes.len() && depth > 0 {
        let c = bytes[i];
        if in_string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    let last_start = *entry_starts.last().unwrap();
                    entries.push(&after[last_start..i]);
                    break;
                }
            }
            b',' if depth == 1 => {
                let last_start = *entry_starts.last().unwrap();
                entries.push(&after[last_start..i]);
                entry_starts.push(i + 1);
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return Err("unterminated cases array".into());
    }

    let mut out = Vec::with_capacity(entries.len());
    for raw in entries {
        let entry = raw.trim();
        if entry.is_empty() {
            continue;
        }
        out.push(parse_single_case(entry)?);
    }
    Ok(out)
}

fn parse_single_case(entry: &str) -> Result<PlanetFixedSeed, String> {
    let kind = parse_str_field(entry, "kind")
        .ok_or_else(|| format!("missing \"kind\" in entry: {entry}"))?;
    let read_time = parse_num_field(entry, "read_time")
        .ok_or_else(|| format!("missing \"read_time\" in entry: {entry}"))?;
    match kind.as_str() {
        "cartesian" => {
            let xyz = parse_array3_field(entry, "cart_m")
                .ok_or_else(|| format!("missing \"cart_m\" in entry: {entry}"))?;
            Ok(PlanetFixedSeed::Cartesian {
                read_time,
                cart_m: DVec3::new(xyz[0], xyz[1], xyz[2]),
            })
        }
        "spherical" => Ok(PlanetFixedSeed::Spherical {
            read_time,
            altitude_m: parse_num_field(entry, "altitude_m")
                .ok_or_else(|| format!("missing altitude_m in {entry}"))?,
            latitude_rad: parse_num_field(entry, "latitude_rad")
                .ok_or_else(|| format!("missing latitude_rad in {entry}"))?,
            longitude_rad: parse_num_field(entry, "longitude_rad")
                .ok_or_else(|| format!("missing longitude_rad in {entry}"))?,
        }),
        "elliptical" => Ok(PlanetFixedSeed::Elliptical {
            read_time,
            altitude_m: parse_num_field(entry, "altitude_m")
                .ok_or_else(|| format!("missing altitude_m in {entry}"))?,
            latitude_rad: parse_num_field(entry, "latitude_rad")
                .ok_or_else(|| format!("missing latitude_rad in {entry}"))?,
            longitude_rad: parse_num_field(entry, "longitude_rad")
                .ok_or_else(|| format!("missing longitude_rad in {entry}"))?,
        }),
        other => Err(format!("unknown kind \"{other}\" in entry: {entry}")),
    }
}

fn parse_str_field(s: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let idx = s.find(&needle)?;
    let rest = &s[idx + needle.len()..];
    let colon = rest.find(':')?;
    let after_colon = rest[colon + 1..].trim_start();
    let bytes = after_colon.as_bytes();
    if bytes.is_empty() || bytes[0] != b'"' {
        return None;
    }
    // Find closing quote (no escapes expected for kind/source values used here).
    let end = after_colon[1..].find('"')?;
    Some(after_colon[1..1 + end].to_string())
}

fn parse_num_field(s: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let idx = s.find(&needle)?;
    let rest = &s[idx + needle.len()..];
    let colon = rest.find(':')?;
    let after_colon = rest[colon + 1..].trim_start();
    let end = after_colon
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(after_colon.len());
    after_colon[..end].trim().parse().ok()
}

fn parse_array3_field(s: &str, key: &str) -> Option<[f64; 3]> {
    let needle = format!("\"{key}\"");
    let idx = s.find(&needle)?;
    let rest = &s[idx + needle.len()..];
    let lb = rest.find('[')?;
    let rb = rest[lb..].find(']')?;
    let inner = &rest[lb + 1..lb + rb];
    let parts: Vec<f64> = inner
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if parts.len() != 3 {
        return None;
    }
    Some([parts[0], parts[1], parts[2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_three_cases_from_committed_json() {
        let cases = load_planet_fixed_verif_cases();
        assert_eq!(cases.len(), 3, "SIM_PFIXPOSN_VERIF defines three reads");

        // Sanity-check the first case (the Cartesian seed at ISS-ish radius).
        match cases[0] {
            PlanetFixedSeed::Cartesian { cart_m, read_time } => {
                assert!((cart_m.x - 6_778_136.3).abs() < 1e-6);
                assert_eq!(cart_m.y, 0.0);
                assert_eq!(cart_m.z, 0.0);
                assert_eq!(read_time, 1.0);
            }
            _ => panic!("first case must be Cartesian: {:?}", cases[0]),
        }
    }
}
