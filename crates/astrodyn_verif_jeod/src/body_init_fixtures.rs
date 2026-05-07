//! Body initialization fixtures (committed JSON).
//!
//! Each `test_data/body_init/<vehicle>.json` file contains the body-init
//! vectors for one JEOD scenario (`ISS`, `STS_114`, ...) extracted once
//! from the `Modified_data/<vehicle>/*.py` files under
//! `models/dynamics/body_action/verif/SIM_orbinit/`. Tests load these
//! fixtures via the higher-level helpers in
//! [`crate::reference_state`] and [`crate::orbital_init`]; the present
//! module owns the JSON schema, hand-rolled parser, and per-vehicle bundle
//! cache.
//!
//! Regenerate with:
//!
//! ```bash
//! cargo run -p astrodyn_verif_jeod --bin extract_body_init -- \
//!     --jeod-home $JEOD_HOME
//! ```
//!
//! ## Schema
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "vehicle": "ISS",
//!   "source": "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/",
//!   "jeod_version": "5.4",
//!   "note": "Body initialization vectors. Regenerate with: ...",
//!   "reference_inertial": {"position": [...], "velocity": [...]} | null,
//!   "orbital_inits": [
//!     {"name": "trans_Orbit_inertial_body_set01",
//!      "semi_major_axis": ..., "eccentricity": ..., "inclination": ...,
//!      "ascending_node": ..., "arg_periapsis": ...,
//!      "time_periapsis": ... | null,
//!      "mean_anomaly":  ... | null,
//!      "true_anomaly":  ... | null,
//!      "planet_name": "Earth",
//!      "reference_frame": "Earth.inertial"},
//!     ...
//!   ],
//!   "trans_states": [
//!     {"name": "trans_TransState_inertial_body",
//!      "position": [..., ..., ...], "velocity": [..., ..., ...],
//!      "reference_frame": "Earth.inertial"},
//!     ...
//!   ]
//! }
//! ```
//!
//! The parser is hand-rolled (no `serde_json`) to match the project-wide
//! pattern in `planet_geodetic_verif.rs`, `tier3_baseline_diff.rs`, and
//! `tier3_report.rs`. All numeric fields use Rust's `{x:?}` round-trippable
//! `f64` rendering so re-parsing is bit-identical.

use std::sync::OnceLock;

/// Errors raised while parsing JEOD `.py` files (regen path) or the
/// committed JSON fixtures (runtime path).
#[derive(Debug, thiserror::Error)]
pub enum BodyInitFixtureError {
    /// Source data violated the expected layout (missing fields,
    /// non-numeric values, mismatched array lengths).
    #[error("malformed body-init data: {0}")]
    Malformed(String),
}

impl BodyInitFixtureError {
    pub(crate) fn malformed<S: Into<String>>(s: S) -> Self {
        Self::Malformed(s.into())
    }
}

/// One reference-state record (ECI position / velocity).
#[derive(Debug, Clone)]
pub struct ReferenceStateRecord {
    /// Position in metres, expressed in the inertial frame.
    pub position: [f64; 3],
    /// Velocity in m/s, expressed in the inertial frame.
    pub velocity: [f64; 3],
}

/// One orbital-element init record (the JSON form; `reference_state.rs` and
/// `orbital_init.rs` wrap this with their existing `OrbitalInitData` /
/// `ReferenceState` types for compatibility with downstream tests).
#[derive(Debug, Clone)]
pub struct OrbitalInitRecord {
    /// Vehicle name as recorded in the JEOD `Modified_data/*.py` source.
    pub name: String,
    /// Semi-major axis in metres.
    pub semi_major_axis: f64,
    /// Orbital eccentricity (dimensionless).
    pub eccentricity: f64,
    /// Inclination in radians.
    pub inclination: f64,
    /// Right Ascension of the Ascending Node, in radians.
    pub ascending_node: f64,
    /// Argument of periapsis, in radians.
    pub arg_periapsis: f64,
    /// Time-since-periapsis in seconds, when the JEOD source uses it.
    pub time_periapsis: Option<f64>,
    /// Mean anomaly in radians, when the JEOD source uses it.
    pub mean_anomaly: Option<f64>,
    /// True anomaly in radians, when the JEOD source uses it.
    pub true_anomaly: Option<f64>,
    /// JEOD `planet_name` — Earth / Moon / Sun / Mars.
    pub planet_name: String,
    /// JEOD `reference_frame` selector (e.g. `"earth.inertial"`).
    pub reference_frame: String,
}

/// One direct-Cartesian init record.
#[derive(Debug, Clone)]
pub struct TransStateRecord {
    /// Vehicle name as recorded in the JEOD `Modified_data/*.py` source.
    pub name: String,
    /// Initial position in metres.
    pub position: [f64; 3],
    /// Initial velocity in m/s.
    pub velocity: [f64; 3],
    /// JEOD `reference_frame` selector (e.g. `"earth.inertial"`).
    pub reference_frame: String,
}

/// Per-vehicle bundle as parsed from the JSON fixture.
#[derive(Debug, Clone)]
pub struct BodyInitBundle {
    /// Vehicle name (e.g. `"iss"`, `"sts114"`).
    pub vehicle: String,
    /// Reference inertial state, when the source provides one.
    pub reference_inertial: Option<ReferenceStateRecord>,
    /// Orbital-element initialization records.
    pub orbital_inits: Vec<OrbitalInitRecord>,
    /// Direct-Cartesian translational-state records.
    pub trans_states: Vec<TransStateRecord>,
}

/// Load a vehicle's body-init bundle from `test_data/body_init/<vehicle>.json`.
///
/// The result is cached per process via an internal `OnceLock` per vehicle;
/// calls after the first are constant-time and return the same shared
/// `&'static` reference.
///
/// # Panics
/// Panics if the fixture is missing or malformed, or if the parsed bundle's
/// `vehicle` field does not match the requested vehicle. The message names
/// the regen command per CLAUDE.md "Fail Loudly".
pub fn load_vehicle_bundle(vehicle: &str) -> &'static BodyInitBundle {
    let cache = bundle_cache(vehicle);
    cache.get_or_init(|| {
        let filename = format!("body_init/{}.json", vehicle.to_ascii_lowercase());
        let path = crate::tier3_csv::test_data_path(&filename);
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "Cannot read {}: {e}. Regenerate with: cargo run -p astrodyn_verif_jeod \
                 --bin extract_body_init -- --jeod-home $JEOD_HOME",
                path.display(),
            )
        });
        let bundle = parse_bundle_json(&content).unwrap_or_else(|e| {
            panic!(
                "Malformed body-init fixture {}: {e}. Regenerate with: \
                 cargo run -p astrodyn_verif_jeod --bin extract_body_init -- --jeod-home $JEOD_HOME",
                path.display(),
            )
        });
        assert_eq!(
            bundle.vehicle,
            vehicle,
            "body-init fixture {} contains vehicle {:?} but {:?} was requested. \
             Regenerate with: cargo run -p astrodyn_verif_jeod --bin extract_body_init -- \
             --jeod-home $JEOD_HOME",
            path.display(),
            bundle.vehicle,
            vehicle,
        );
        bundle
    })
}

/// Per-vehicle `OnceLock`. Vehicle list is small (ISS, STS_114) so a static
/// match is simpler than a global `Mutex<HashMap>`.
fn bundle_cache(vehicle: &str) -> &'static OnceLock<BodyInitBundle> {
    static ISS: OnceLock<BodyInitBundle> = OnceLock::new();
    static STS_114: OnceLock<BodyInitBundle> = OnceLock::new();

    match vehicle {
        "ISS" => &ISS,
        "STS_114" => &STS_114,
        other => panic!(
            "load_vehicle_bundle: unknown vehicle {other:?}. \
             Add it to extract_body_init.rs SCENARIOS and to body_init_fixtures::bundle_cache."
        ),
    }
}

/// Schema version this code understands. Bumped whenever the on-disk
/// JSON shape changes in a way the parser cannot accept.
const EXPECTED_SCHEMA_VERSION: u64 = 1;

/// Hand-rolled JSON parser for a body-init bundle. Mirrors the
/// no-`serde_json` style of `planet_geodetic_verif.rs`.
pub(crate) fn parse_bundle_json(s: &str) -> Result<BodyInitBundle, String> {
    let schema_version = parse_num_field(s, "schema_version")
        .ok_or_else(|| "missing top-level \"schema_version\" key".to_string())?
        as u64;
    if schema_version != EXPECTED_SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema_version {schema_version}; this build expects \
             {EXPECTED_SCHEMA_VERSION}. Regenerate with: cargo run -p astrodyn_verif_jeod \
             --bin extract_body_init -- --jeod-home $JEOD_HOME"
        ));
    }

    let vehicle = parse_str_field(s, "vehicle")
        .ok_or_else(|| "missing top-level \"vehicle\" key".to_string())?;

    // reference_inertial: object or `null`.
    let reference_inertial = parse_object_field(s, "reference_inertial")?
        .map(|inner| -> Result<ReferenceStateRecord, String> {
            let position = parse_array3_field(inner, "position")
                .ok_or_else(|| "reference_inertial: missing position".to_string())?;
            let velocity = parse_array3_field(inner, "velocity")
                .ok_or_else(|| "reference_inertial: missing velocity".to_string())?;
            Ok(ReferenceStateRecord { position, velocity })
        })
        .transpose()?;

    let orbital_init_entries = parse_array_field_entries(s, "orbital_inits")?;
    let mut orbital_inits = Vec::with_capacity(orbital_init_entries.len());
    for entry in orbital_init_entries {
        orbital_inits.push(parse_orbital_init_entry(entry)?);
    }

    let trans_state_entries = parse_array_field_entries(s, "trans_states")?;
    let mut trans_states = Vec::with_capacity(trans_state_entries.len());
    for entry in trans_state_entries {
        trans_states.push(parse_trans_state_entry(entry)?);
    }

    Ok(BodyInitBundle {
        vehicle,
        reference_inertial,
        orbital_inits,
        trans_states,
    })
}

fn parse_orbital_init_entry(entry: &str) -> Result<OrbitalInitRecord, String> {
    let name = parse_str_field(entry, "name")
        .ok_or_else(|| format!("orbital_inits entry missing \"name\": {entry}"))?;
    let semi_major_axis = parse_num_field(entry, "semi_major_axis")
        .ok_or_else(|| format!("orbital_inits[{name}]: missing semi_major_axis"))?;
    let eccentricity = parse_num_field(entry, "eccentricity")
        .ok_or_else(|| format!("orbital_inits[{name}]: missing eccentricity"))?;
    let inclination = parse_num_field(entry, "inclination")
        .ok_or_else(|| format!("orbital_inits[{name}]: missing inclination"))?;
    let ascending_node = parse_num_field(entry, "ascending_node")
        .ok_or_else(|| format!("orbital_inits[{name}]: missing ascending_node"))?;
    let arg_periapsis = parse_num_field(entry, "arg_periapsis")
        .ok_or_else(|| format!("orbital_inits[{name}]: missing arg_periapsis"))?;
    let time_periapsis = parse_opt_num_field(entry, "time_periapsis");
    let mean_anomaly = parse_opt_num_field(entry, "mean_anomaly");
    let true_anomaly = parse_opt_num_field(entry, "true_anomaly");
    let planet_name = parse_str_field(entry, "planet_name").unwrap_or_default();
    let reference_frame = parse_str_field(entry, "reference_frame").unwrap_or_default();

    Ok(OrbitalInitRecord {
        name,
        semi_major_axis,
        eccentricity,
        inclination,
        ascending_node,
        arg_periapsis,
        time_periapsis,
        mean_anomaly,
        true_anomaly,
        planet_name,
        reference_frame,
    })
}

fn parse_trans_state_entry(entry: &str) -> Result<TransStateRecord, String> {
    let name = parse_str_field(entry, "name")
        .ok_or_else(|| format!("trans_states entry missing \"name\": {entry}"))?;
    let position = parse_array3_field(entry, "position")
        .ok_or_else(|| format!("trans_states[{name}]: missing position"))?;
    let velocity = parse_array3_field(entry, "velocity")
        .ok_or_else(|| format!("trans_states[{name}]: missing velocity"))?;
    let reference_frame = parse_str_field(entry, "reference_frame").unwrap_or_default();
    Ok(TransStateRecord {
        name,
        position,
        velocity,
        reference_frame,
    })
}

// ── tiny hand-rolled JSON helpers ──────────────────────────────────────────

fn find_key_value_start<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let idx = s.find(&needle)?;
    let rest = &s[idx + needle.len()..];
    let colon = rest.find(':')?;
    Some(rest[colon + 1..].trim_start())
}

fn parse_str_field(s: &str, key: &str) -> Option<String> {
    let after = find_key_value_start(s, key)?;
    let bytes = after.as_bytes();
    if bytes.is_empty() || bytes[0] != b'"' {
        return None;
    }
    let end = after[1..].find('"')?;
    Some(after[1..1 + end].to_string())
}

fn parse_num_field(s: &str, key: &str) -> Option<f64> {
    let after = find_key_value_start(s, key)?;
    if after.starts_with("null") {
        return None;
    }
    let end = after
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(after.len());
    after[..end].trim().parse().ok()
}

/// Returns `None` for `null`, `Some(v)` for a literal number, `None` if the
/// key is absent.
fn parse_opt_num_field(s: &str, key: &str) -> Option<f64> {
    let after = find_key_value_start(s, key)?;
    if after.starts_with("null") {
        return None;
    }
    let end = after
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(after.len());
    after[..end].trim().parse().ok()
}

fn parse_array3_field(s: &str, key: &str) -> Option<[f64; 3]> {
    let after = find_key_value_start(s, key)?;
    if !after.starts_with('[') {
        return None;
    }
    let close = after.find(']')?;
    let inner = &after[1..close];
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

/// Parse an object-or-null field. Returns `Ok(None)` for `null` and
/// `Ok(Some(inner))` for an object body, where `inner` is the substring
/// between the matching `{` and `}` (exclusive).
fn parse_object_field<'a>(s: &'a str, key: &str) -> Result<Option<&'a str>, String> {
    let after =
        find_key_value_start(s, key).ok_or_else(|| format!("missing top-level {key:?} key"))?;
    if after.starts_with("null") {
        return Ok(None);
    }
    if !after.starts_with('{') {
        return Err(format!("{key} is not an object or null"));
    }
    let bytes = after.as_bytes();
    let mut depth = 1_i32;
    let mut i = 1;
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
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            _ => {}
        }
        i += 1;
        if depth == 0 {
            return Ok(Some(&after[1..i - 1]));
        }
    }
    Err(format!("unterminated object value for {key}"))
}

/// Parse an array-of-objects field. Returns the substring of each entry
/// (between matching `{` and `}`).
fn parse_array_field_entries<'a>(s: &'a str, key: &str) -> Result<Vec<&'a str>, String> {
    let after =
        find_key_value_start(s, key).ok_or_else(|| format!("missing top-level {key:?} key"))?;
    if !after.starts_with('[') {
        return Err(format!("{key} is not an array"));
    }
    let bytes = after.as_bytes();
    let mut depth = 1_i32; // we have already entered the outer `[`
    let mut i = 1;
    let mut in_string = false;
    let mut entries = Vec::new();
    let mut entry_start: Option<usize> = None;
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
            b'{' => {
                if depth == 1 {
                    entry_start = Some(i + 1);
                }
                depth += 1;
            }
            b'[' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 1 {
                    let start = entry_start.take().ok_or_else(|| {
                        format!("{key}: closing }} without a matching object opener")
                    })?;
                    entries.push(&after[start..i]);
                }
            }
            b']' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return Err(format!("unterminated array for {key}"));
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_BUNDLE: &str = r#"{
  "schema_version": 1,
  "vehicle": "TEST",
  "source": "test/path/",
  "jeod_version": "5.4",
  "note": "Test bundle.",
  "reference_inertial": {"position": [1.0, 2.0, 3.0], "velocity": [4.0, 5.0, 6.0]},
  "orbital_inits": [
    {
      "name": "set_a",
      "semi_major_axis": 6.7e6,
      "eccentricity": 0.001,
      "inclination": 0.9,
      "ascending_node": 0.86,
      "arg_periapsis": 1.75,
      "time_periapsis": 4581.96,
      "mean_anomaly": null,
      "true_anomaly": null,
      "planet_name": "Earth",
      "reference_frame": "Earth.inertial"
    },
    {
      "name": "set_b",
      "semi_major_axis": 6.7e6,
      "eccentricity": 0.001,
      "inclination": 0.9,
      "ascending_node": 0.86,
      "arg_periapsis": 1.75,
      "time_periapsis": null,
      "mean_anomaly": 0.5,
      "true_anomaly": null,
      "planet_name": "Earth",
      "reference_frame": "Earth.inertial"
    }
  ],
  "trans_states": [
    {
      "name": "ts_a",
      "position": [1.0e6, 2.0e6, 3.0e6],
      "velocity": [-1.0, 2.0, -3.0],
      "reference_frame": "Earth.inertial"
    }
  ]
}
"#;

    #[test]
    fn parses_full_bundle() {
        let b = parse_bundle_json(SAMPLE_BUNDLE).unwrap();
        assert_eq!(b.vehicle, "TEST");
        let r = b.reference_inertial.unwrap();
        assert_eq!(r.position, [1.0, 2.0, 3.0]);
        assert_eq!(r.velocity, [4.0, 5.0, 6.0]);

        assert_eq!(b.orbital_inits.len(), 2);
        let a = &b.orbital_inits[0];
        assert_eq!(a.name, "set_a");
        assert_eq!(a.time_periapsis, Some(4581.96));
        assert_eq!(a.mean_anomaly, None);
        let bb = &b.orbital_inits[1];
        assert_eq!(bb.name, "set_b");
        assert_eq!(bb.time_periapsis, None);
        assert_eq!(bb.mean_anomaly, Some(0.5));

        assert_eq!(b.trans_states.len(), 1);
        let t = &b.trans_states[0];
        assert_eq!(t.name, "ts_a");
        assert_eq!(t.position, [1.0e6, 2.0e6, 3.0e6]);
        assert_eq!(t.velocity, [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn parses_null_reference_inertial() {
        let json = r#"{"schema_version": 1, "vehicle": "X", "reference_inertial": null,
"orbital_inits": [], "trans_states": []}"#;
        let b = parse_bundle_json(json).unwrap();
        assert!(b.reference_inertial.is_none());
        assert!(b.orbital_inits.is_empty());
        assert!(b.trans_states.is_empty());
    }

    #[test]
    fn rejects_missing_schema_version() {
        let json = r#"{"vehicle": "X", "reference_inertial": null,
"orbital_inits": [], "trans_states": []}"#;
        let err = parse_bundle_json(json).unwrap_err();
        assert!(err.contains("schema_version"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let json = r#"{"schema_version": 999, "vehicle": "X", "reference_inertial": null,
"orbital_inits": [], "trans_states": []}"#;
        let err = parse_bundle_json(json).unwrap_err();
        assert!(err.contains("unsupported schema_version 999"), "got: {err}");
    }
}
