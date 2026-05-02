//! Orbital-element and direct-Cartesian initialization records.
//!
//! Originally extracted from JEOD `Modified_data/<vehicle>/`
//! `trans_Orbit_*_body_*.py` and `trans_TransState_*_body.py` (in
//! `models/dynamics/body_action/verif/SIM_orbinit/`), the records are now
//! committed to `test_data/body_init/<vehicle>.json` and read back here
//! without touching the JEOD source tree at runtime.
//!
//! Regenerate with:
//!
//! ```bash
//! cargo run -p jeod_test_data --bin extract_body_init -- \
//!     --jeod-home $JEOD_HOME
//! ```
//!
//! The Python parsers still live here (`parse_orbital_init_py`,
//! `parse_trans_state_py`) and are invoked exclusively by the regen
//! binary; runtime test paths never call them.

use regex::Regex;
use std::f64::consts::PI;

use crate::body_init_fixtures::{
    load_vehicle_bundle, BodyInitFixtureError, OrbitalInitRecord, TransStateRecord,
};

/// Orbital element initialization data, originally from a JEOD Trick input file.
///
/// JEOD source files like
/// `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/trans_Orbit_{frame}_body_{init_name}.py`
/// contain Python assignments in two forms:
/// - `key = trick.attach_units("unit", value)` (with unit conversion)
/// - `key = value` (bare numeric)
///
/// Unit conversions applied automatically:
/// - `"degree"` -> radians (multiply by PI/180)
/// - `"km"` -> meters (multiply by 1000)
/// - `"s"` -> seconds (no conversion)
#[derive(Debug, Clone)]
pub struct OrbitalInitData {
    /// Semi-major axis in metres (converted from km in the source).
    pub semi_major_axis: f64,
    /// Eccentricity (dimensionless).
    pub eccentricity: f64,
    /// Inclination in radians (converted from degrees in the source).
    pub inclination: f64,
    /// Right Ascension of the Ascending Node in radians.
    pub ascending_node: f64,
    /// Argument of periapsis in radians.
    pub arg_periapsis: f64,
    /// Time-since-periapsis in seconds, when used.
    pub time_periapsis: Option<f64>,
    /// Mean anomaly in radians, when used.
    pub mean_anomaly: Option<f64>,
    /// True anomaly in radians, when used.
    pub true_anomaly: Option<f64>,
    /// JEOD `planet_name` (e.g. `"Earth"`, `"Mars"`).
    pub planet_name: String,
    /// JEOD `reference_frame` selector (e.g. `"earth.inertial"`).
    pub reference_frame: String,
}

/// Load orbital element initialization data from the committed
/// `test_data/body_init/<vehicle>.json` fixture.
///
/// # Arguments
/// * `vehicle` - Vehicle directory name (e.g. `"ISS"`, `"STS_114"`).
/// * `init_name` - Init file identifier without `.py`
///   (e.g. `"trans_Orbit_inertial_body_set01"`).
///
/// # Panics
/// Panics if the fixture is missing, malformed, or doesn't contain the
/// requested `init_name`. The panic message names the regen command per
/// the CLAUDE.md "Fail Loudly" rule.
pub fn load_orbital_init(vehicle: &str, init_name: &str) -> OrbitalInitData {
    let bundle = load_vehicle_bundle(vehicle);
    let rec = bundle
        .orbital_inits
        .iter()
        .find(|r| r.name == init_name)
        .unwrap_or_else(|| {
            panic!(
                "body_init fixture for {vehicle} is missing orbital_init {init_name:?}. \
                 Add the scenario to extract_body_init.rs SCENARIOS and regenerate with: \
                 cargo run -p jeod_test_data --bin extract_body_init -- --jeod-home $JEOD_HOME"
            )
        });
    OrbitalInitData::from(rec)
}

impl From<&OrbitalInitRecord> for OrbitalInitData {
    fn from(rec: &OrbitalInitRecord) -> Self {
        OrbitalInitData {
            semi_major_axis: rec.semi_major_axis,
            eccentricity: rec.eccentricity,
            inclination: rec.inclination,
            ascending_node: rec.ascending_node,
            arg_periapsis: rec.arg_periapsis,
            time_periapsis: rec.time_periapsis,
            mean_anomaly: rec.mean_anomaly,
            true_anomaly: rec.true_anomaly,
            planet_name: rec.planet_name.clone(),
            reference_frame: rec.reference_frame.clone(),
        }
    }
}

/// Parse the body of a JEOD `trans_Orbit_*_body_*.py` file into an
/// [`OrbitalInitRecord`] (the canonical JSON-serializable form).
///
/// Regen-only path: the runtime [`load_orbital_init`] reads the committed
/// fixture and never invokes this parser.
pub fn parse_orbital_init_py(content: &str) -> Result<OrbitalInitRecord, BodyInitFixtureError> {
    // Match: key = trick.attach_units( "unit", value)
    let units_re =
        Regex::new(r#"\.(\w+)\s*=\s*trick\.attach_units\(\s*"(\w+)"\s*,\s*([-\d.eE+]+)\s*\)"#)
            .unwrap();

    // Match: key = bare_value (no trick.attach_units)
    let bare_re = Regex::new(r"\.(\w+)\s*=\s+([-\d.eE+]+)\s*$").unwrap();

    // Match planet_name
    let planet_re = Regex::new(r#"\.planet_name\s*=\s*"(\w+)""#).unwrap();

    // Match orbit_frame_name
    let frame_re = Regex::new(r#"\.orbit_frame_name\s*=\s*"([^"]+)""#).unwrap();

    let mut semi_major_axis: Option<f64> = None;
    let mut eccentricity: Option<f64> = None;
    let mut inclination: Option<f64> = None;
    let mut ascending_node: Option<f64> = None;
    let mut arg_periapsis: Option<f64> = None;
    let mut time_periapsis: Option<f64> = None;
    let mut mean_anomaly: Option<f64> = None;
    let mut true_anomaly: Option<f64> = None;
    let mut planet_name = String::new();
    let mut reference_frame = String::new();

    for line in content.lines() {
        // Try trick.attach_units pattern
        if let Some(cap) = units_re.captures(line) {
            let key = &cap[1];
            let unit = &cap[2];
            let raw_val: f64 = cap[3]
                .parse()
                .map_err(|e| BodyInitFixtureError::malformed(format!("parse {key}: {e}")))?;

            let val = convert_units(raw_val, unit)?;

            match key {
                "semi_major_axis" => semi_major_axis = Some(val),
                "eccentricity" => eccentricity = Some(val),
                "inclination" => inclination = Some(val),
                "ascending_node" => ascending_node = Some(val),
                "arg_periapsis" => arg_periapsis = Some(val),
                "time_periapsis" => time_periapsis = Some(val),
                "mean_anomaly" => mean_anomaly = Some(val),
                "true_anomaly" => true_anomaly = Some(val),
                _ => {}
            }
            continue;
        }

        // Try bare value pattern
        if let Some(cap) = bare_re.captures(line) {
            let key = &cap[1];
            let val: f64 = cap[2]
                .parse()
                .map_err(|e| BodyInitFixtureError::malformed(format!("parse {key}: {e}")))?;

            match key {
                "semi_major_axis" => semi_major_axis = Some(val * 1000.0), // assume km
                "eccentricity" => eccentricity = Some(val),
                "inclination" => inclination = Some(val * PI / 180.0), // assume degrees
                "ascending_node" => ascending_node = Some(val * PI / 180.0),
                "arg_periapsis" => arg_periapsis = Some(val * PI / 180.0),
                "time_periapsis" => time_periapsis = Some(val),
                "mean_anomaly" => mean_anomaly = Some(val * PI / 180.0),
                "true_anomaly" => true_anomaly = Some(val * PI / 180.0),
                _ => {}
            }
            continue;
        }

        // Try planet_name
        if let Some(cap) = planet_re.captures(line) {
            planet_name = cap[1].to_string();
            continue;
        }

        // Try orbit_frame_name
        if let Some(cap) = frame_re.captures(line) {
            reference_frame = cap[1].to_string();
        }
    }

    Ok(OrbitalInitRecord {
        name: String::new(), // filled in by extract_body_init
        semi_major_axis: semi_major_axis.ok_or_else(|| {
            BodyInitFixtureError::malformed("missing semi_major_axis".to_string())
        })?,
        eccentricity: eccentricity
            .ok_or_else(|| BodyInitFixtureError::malformed("missing eccentricity".to_string()))?,
        inclination: inclination
            .ok_or_else(|| BodyInitFixtureError::malformed("missing inclination".to_string()))?,
        ascending_node: ascending_node
            .ok_or_else(|| BodyInitFixtureError::malformed("missing ascending_node".to_string()))?,
        arg_periapsis: arg_periapsis
            .ok_or_else(|| BodyInitFixtureError::malformed("missing arg_periapsis".to_string()))?,
        time_periapsis,
        mean_anomaly,
        true_anomaly,
        planet_name,
        reference_frame,
    })
}

/// Convert a raw value from the given unit string to SI (meters, radians, seconds).
fn convert_units(val: f64, unit: &str) -> Result<f64, BodyInitFixtureError> {
    match unit {
        "degree" => Ok(val * PI / 180.0),
        "km" => Ok(val * 1000.0),
        "s" => Ok(val),
        "m" => Ok(val),
        "rad" => Ok(val),
        other => Err(BodyInitFixtureError::malformed(format!(
            "unknown unit: {other:?}"
        ))),
    }
}

/// Direct Cartesian translational-state initialization data, originally from a
/// JEOD `trans_TransState_*.py` file.
///
/// JEOD source files like
/// `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/trans_TransState_{frame}_body.py`
/// contain position/velocity vectors with `trick.attach_units("m", [...])` and
/// `trick.attach_units("m/s", [...])` wrappers.
#[derive(Debug, Clone)]
pub struct TransStateData {
    /// Position in metres.
    pub position: [f64; 3],
    /// Velocity in m/s.
    pub velocity: [f64; 3],
    /// JEOD `reference_frame` selector.
    pub reference_frame: String,
}

/// Load a direct Cartesian translational-state init record from the
/// committed `test_data/body_init/<vehicle>.json` fixture.
///
/// # Arguments
/// * `vehicle` - Vehicle directory name (e.g. `"STS_114"`).
/// * `init_name` - Init file identifier without `.py`
///   (e.g. `"trans_TransState_inertial_body"`).
///
/// # Panics
/// Panics if the fixture is missing, malformed, or doesn't contain the
/// requested `init_name`. The panic message names the regen command.
pub fn load_trans_state(vehicle: &str, init_name: &str) -> TransStateData {
    let bundle = load_vehicle_bundle(vehicle);
    let rec = bundle
        .trans_states
        .iter()
        .find(|r| r.name == init_name)
        .unwrap_or_else(|| {
            panic!(
                "body_init fixture for {vehicle} is missing trans_state {init_name:?}. \
                 Add the scenario to extract_body_init.rs SCENARIOS and regenerate with: \
                 cargo run -p jeod_test_data --bin extract_body_init -- --jeod-home $JEOD_HOME"
            )
        });
    TransStateData {
        position: rec.position,
        velocity: rec.velocity,
        reference_frame: rec.reference_frame.clone(),
    }
}

/// Parse the body of a JEOD `trans_TransState_*.py` file into a
/// [`TransStateRecord`].
///
/// Regen-only path: the runtime [`load_trans_state`] reads the committed
/// fixture and never invokes this parser.
pub fn parse_trans_state_py(content: &str) -> Result<TransStateRecord, BodyInitFixtureError> {
    // Match: .position = trick.attach_units( "m", [  x,  y,  z])
    // Match: .velocity = trick.attach_units( "m/s", [  vx,  vy,  vz])
    let vec3_re = Regex::new(
        r#"\.(position|velocity)\s*=\s*trick\.attach_units\(\s*"([^"]+)"\s*,\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]\s*\)"#,
    )
    .unwrap();

    // Match: .reference_ref_frame_name = "Earth.inertial"
    let frame_re = Regex::new(r#"\.reference_ref_frame_name\s*=\s*"([^"]+)""#).unwrap();

    let mut position: Option<[f64; 3]> = None;
    let mut velocity: Option<[f64; 3]> = None;
    let mut reference_frame: Option<String> = None;

    for line in content.lines() {
        if let Some(cap) = vec3_re.captures(line) {
            let field = &cap[1];
            let unit = &cap[2];
            let x: f64 = cap[3]
                .parse()
                .map_err(|e| BodyInitFixtureError::malformed(format!("parse {field}.x: {e}")))?;
            let y: f64 = cap[4]
                .parse()
                .map_err(|e| BodyInitFixtureError::malformed(format!("parse {field}.y: {e}")))?;
            let z: f64 = cap[5]
                .parse()
                .map_err(|e| BodyInitFixtureError::malformed(format!("parse {field}.z: {e}")))?;
            // Validate unit per field — no scale conversion needed for SI.
            match field {
                "position" => {
                    if unit != "m" {
                        return Err(BodyInitFixtureError::malformed(format!(
                            "unexpected unit {unit:?} for position (expected \"m\")"
                        )));
                    }
                    position = Some([x, y, z]);
                }
                "velocity" => {
                    if unit != "m/s" {
                        return Err(BodyInitFixtureError::malformed(format!(
                            "unexpected unit {unit:?} for velocity (expected \"m/s\")"
                        )));
                    }
                    velocity = Some([x, y, z]);
                }
                _ => {}
            }
            continue;
        }
        if let Some(cap) = frame_re.captures(line) {
            reference_frame = Some(cap[1].to_string());
        }
    }

    Ok(TransStateRecord {
        name: String::new(), // filled in by extract_body_init
        position: position
            .ok_or_else(|| BodyInitFixtureError::malformed("missing position".to_string()))?,
        velocity: velocity
            .ok_or_else(|| BodyInitFixtureError::malformed("missing velocity".to_string()))?,
        reference_frame: reference_frame.ok_or_else(|| {
            BodyInitFixtureError::malformed("missing reference_ref_frame_name".to_string())
        })?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_orbital_init_py_attach_units() {
        let py = r#"
vehicle.set01.subject.semi_major_axis = trick.attach_units("km", 6732.90120152)
vehicle.set01.subject.eccentricity   = trick.attach_units("rad", 0.00129073350)
vehicle.set01.subject.inclination    = trick.attach_units("degree", 51.670450765)
vehicle.set01.subject.ascending_node = trick.attach_units("degree", 49.708417385)
vehicle.set01.subject.arg_periapsis  = trick.attach_units("degree", 100.582445989)
vehicle.set01.subject.time_periapsis = trick.attach_units("s", 4581.96167293)
vehicle.set01.subject.planet_name    = "Earth"
vehicle.set01.subject.orbit_frame_name = "Earth.inertial"
"#;
        let rec = parse_orbital_init_py(py).unwrap();
        assert!((rec.semi_major_axis - 6_732_901.201_52).abs() < 1e-6);
        assert!((rec.eccentricity - 0.00129073350).abs() < 1e-12);
        let deg2rad = PI / 180.0;
        assert!((rec.inclination - 51.670450765 * deg2rad).abs() < 1e-12);
        assert_eq!(rec.planet_name, "Earth");
        assert_eq!(rec.reference_frame, "Earth.inertial");
        assert!((rec.time_periapsis.unwrap() - 4581.96167293).abs() < 1e-9);
    }

    #[test]
    fn parse_trans_state_py_attach_units() {
        let py = r#"
sts114.trans.subject.position = trick.attach_units("m", [1.0e6, 2.0e6, 3.0e6])
sts114.trans.subject.velocity = trick.attach_units("m/s", [-1.0, 2.0, -3.0])
sts114.trans.subject.reference_ref_frame_name = "Earth.inertial"
"#;
        let rec = parse_trans_state_py(py).unwrap();
        assert_eq!(rec.position, [1.0e6, 2.0e6, 3.0e6]);
        assert_eq!(rec.velocity, [-1.0, 2.0, -3.0]);
        assert_eq!(rec.reference_frame, "Earth.inertial");
    }

    #[test]
    fn parse_trans_state_py_rejects_wrong_units() {
        let py = r#"sts114.trans.subject.position = trick.attach_units("km", [1.0, 2.0, 3.0])
sts114.trans.subject.velocity = trick.attach_units("m/s", [-1.0, 2.0, -3.0])
sts114.trans.subject.reference_ref_frame_name = "Earth.inertial"
"#;
        let err = parse_trans_state_py(py).unwrap_err();
        assert!(format!("{err}").contains("position"));
    }
}
