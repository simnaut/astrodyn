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
//! cargo run -p astrodyn_verif_jeod --bin extract_body_init -- \
//!     --jeod-home $JEOD_HOME
//! ```
//!
//! The Python parsers still live here (`parse_orbital_init_py`,
//! `parse_trans_state_py`) and are invoked exclusively by the regen
//! binary; runtime test paths never call them.

use regex::Regex;

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
/// - `"degree"` -> radians (via `f64::to_radians`)
/// - `"km"` -> meters (multiply by 1000)
/// - `"s"` -> seconds (no conversion)
#[derive(Debug, Clone)]
pub struct OrbitalInitData {
    /// Semi-major axis in metres (converted from km in the source), when the
    /// JEOD set provides it (sets 01/02/10). `None` for set03, which carries
    /// `semi_latus_rectum` instead.
    pub semi_major_axis: Option<f64>,
    /// Semi-latus rectum in metres (converted from km in the source), when the
    /// JEOD set provides it (set03 `SlrEccIncAscnodeArgperTanom`). `None` for
    /// sma-parameterized sets.
    pub semi_latus_rectum: Option<f64>,
    /// Apoapsis altitude in metres above the planet equatorial radius
    /// (converted from km in the source), when the JEOD set uses the altitude
    /// shape (sets 04/05). `None` otherwise.
    pub alt_apoapsis: Option<f64>,
    /// Periapsis altitude in metres above the planet equatorial radius
    /// (converted from km in the source), when the JEOD set uses the altitude
    /// shape (sets 04/05). `None` otherwise.
    pub alt_periapsis: Option<f64>,
    /// Eccentricity (dimensionless), when the JEOD set provides it directly
    /// (sma/slr sets). `None` for the altitude shape (sets 04/05), where it is
    /// derived from the apo/peri altitudes by the `init_from_altitudes_*`
    /// converters.
    pub eccentricity: Option<f64>,
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
                 cargo run -p astrodyn_verif_jeod --bin extract_body_init -- --jeod-home $JEOD_HOME"
            )
        });
    OrbitalInitData::from(rec)
}

impl From<&OrbitalInitRecord> for OrbitalInitData {
    fn from(rec: &OrbitalInitRecord) -> Self {
        OrbitalInitData {
            semi_major_axis: rec.semi_major_axis,
            semi_latus_rectum: rec.semi_latus_rectum,
            alt_apoapsis: rec.alt_apoapsis,
            alt_periapsis: rec.alt_periapsis,
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
    let mut semi_latus_rectum: Option<f64> = None;
    let mut alt_apoapsis: Option<f64> = None;
    let mut alt_periapsis: Option<f64> = None;
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
                "semi_latus_rectum" => semi_latus_rectum = Some(val),
                "alt_apoapsis" => alt_apoapsis = Some(val),
                "alt_periapsis" => alt_periapsis = Some(val),
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
                "semi_latus_rectum" => semi_latus_rectum = Some(val * 1000.0), // assume km
                "alt_apoapsis" => alt_apoapsis = Some(val * 1000.0),       // assume km
                "alt_periapsis" => alt_periapsis = Some(val * 1000.0),     // assume km
                "eccentricity" => eccentricity = Some(val),
                "inclination" => inclination = Some(val.to_radians()), // assume degrees
                "ascending_node" => ascending_node = Some(val.to_radians()),
                "arg_periapsis" => arg_periapsis = Some(val.to_radians()),
                "time_periapsis" => time_periapsis = Some(val),
                "mean_anomaly" => mean_anomaly = Some(val.to_radians()),
                "true_anomaly" => true_anomaly = Some(val.to_radians()),
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

    // JEOD sets 01/02/10 supply `semi_major_axis`; set03
    // (`SlrEccIncAscnodeArgperTanom`) supplies `semi_latus_rectum`; sets 04/05
    // (`IncAscnodeAltperAltapo…`) supply an apo/peri altitude pair. The shape
    // sources are mutually exclusive per JEOD set — require *exactly* one.
    if alt_apoapsis.is_some() != alt_periapsis.is_some() {
        return Err(BodyInitFixtureError::malformed(
            "the altitude shape requires both alt_apoapsis and alt_periapsis, got only one"
                .to_string(),
        ));
    }
    let n_shapes = u8::from(semi_major_axis.is_some())
        + u8::from(semi_latus_rectum.is_some())
        + u8::from(alt_apoapsis.is_some());
    if n_shapes != 1 {
        return Err(BodyInitFixtureError::malformed(format!(
            "expected exactly one orbit-shape source (semi_major_axis, semi_latus_rectum, or \
             alt_apoapsis+alt_periapsis), found {n_shapes}; they are mutually exclusive per JEOD set"
        )));
    }
    // Eccentricity is supplied directly by sma/slr sets; the altitude shape
    // derives it from the apo/peri altitudes, so require it iff no altitudes.
    if eccentricity.is_none() && alt_apoapsis.is_none() {
        return Err(BodyInitFixtureError::malformed(
            "missing eccentricity (required for the sma/slr shapes)".to_string(),
        ));
    }
    // The altitude shape *derives* eccentricity; an eccentricity supplied
    // alongside the altitude pair is ambiguous (and would mask a malformed or
    // stale deck), so reject it to keep the schema canonical.
    if eccentricity.is_some() && alt_apoapsis.is_some() {
        return Err(BodyInitFixtureError::malformed(
            "eccentricity must not be set alongside the altitude shape (alt_apoapsis/\
             alt_periapsis); the altitude shape derives eccentricity"
                .to_string(),
        ));
    }
    Ok(OrbitalInitRecord {
        name: String::new(), // filled in by extract_body_init
        semi_major_axis,
        semi_latus_rectum,
        alt_apoapsis,
        alt_periapsis,
        eccentricity,
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
        "degree" => Ok(val.to_radians()),
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
                 cargo run -p astrodyn_verif_jeod --bin extract_body_init -- --jeod-home $JEOD_HOME"
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
#[allow(
    clippy::float_cmp,
    reason = "trans-state parser tests assert bit-exact recovery of literal Python init values"
)]
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
        assert!((rec.semi_major_axis.unwrap() - 6_732_901.201_52).abs() < 1e-6);
        assert_eq!(rec.semi_latus_rectum, None);
        assert!((rec.eccentricity.unwrap() - 0.00129073350).abs() < 1e-12);
        assert!((rec.inclination - 51.670450765_f64.to_radians()).abs() < 1e-12);
        assert_eq!(rec.planet_name, "Earth");
        assert_eq!(rec.reference_frame, "Earth.inertial");
        assert!((rec.time_periapsis.unwrap() - 4581.96167293).abs() < 1e-9);
    }

    #[test]
    fn parse_orbital_init_py_set03_slr_true_anomaly() {
        // JEOD set03 (`SlrEccIncAscnodeArgperTanom`): semi_latus_rectum +
        // true_anomaly, no semi_major_axis. Values are the ISS set03 deck.
        let py = r#"
  vehicle.orb_init.arg_periapsis  = trick.attach_units( "degree",100.582445989)
  vehicle.orb_init.eccentricity            =    0.00129073350
  vehicle.orb_init.inclination  = trick.attach_units( "degree",51.670450765)
  vehicle.orb_init.ascending_node  = trick.attach_units( "degree",49.708417385)
  vehicle.orb_init.semi_latus_rectum  = trick.attach_units( "km",6732.88998455)
  vehicle.orb_init.true_anomaly  = trick.attach_units( "degree",299.884499026)
  vehicle.orb_init.planet_name      = "Earth"
  vehicle.orb_init.orbit_frame_name = "Earth.inertial"
"#;
        let rec = parse_orbital_init_py(py).unwrap();
        assert_eq!(rec.semi_major_axis, None);
        assert!((rec.semi_latus_rectum.unwrap() - 6_732_889.984_55).abs() < 1e-6);
        assert!((rec.eccentricity.unwrap() - 0.00129073350).abs() < 1e-12);
        assert!((rec.true_anomaly.unwrap() - 299.884499026_f64.to_radians()).abs() < 1e-12);
        assert_eq!(rec.time_periapsis, None);
        assert_eq!(rec.mean_anomaly, None);
    }

    #[test]
    fn parse_orbital_init_py_rejects_both_sma_and_slr() {
        // `semi_major_axis` and `semi_latus_rectum` are mutually exclusive per
        // JEOD set; a deck carrying both is malformed.
        let py = r#"
  vehicle.orb_init.arg_periapsis  = trick.attach_units( "degree",100.582445989)
  vehicle.orb_init.eccentricity            =    0.00129073350
  vehicle.orb_init.inclination  = trick.attach_units( "degree",51.670450765)
  vehicle.orb_init.ascending_node  = trick.attach_units( "degree",49.708417385)
  vehicle.orb_init.semi_major_axis  = trick.attach_units( "km",6732.90120152)
  vehicle.orb_init.semi_latus_rectum  = trick.attach_units( "km",6732.88998455)
  vehicle.orb_init.true_anomaly  = trick.attach_units( "degree",299.884499026)
  vehicle.orb_init.planet_name      = "Earth"
  vehicle.orb_init.orbit_frame_name = "Earth.inertial"
"#;
        let err = parse_orbital_init_py(py).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("exactly one orbit-shape source"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("mutually exclusive"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn parse_orbital_init_py_set04_altitudes_true_anomaly() {
        // JEOD set04 (`IncAscnodeAltperAltapoArgperTanom`): apo/peri altitudes +
        // true anomaly, no sma/slr, no eccentricity. Values are the ISS set04 deck.
        let py = r#"
  vehicle.orb_init.set              = 4
  vehicle.orb_init.alt_apoapsis  = trick.attach_units( "km",363.45458264)
  vehicle.orb_init.alt_periapsis  = trick.attach_units( "km",346.07382040)
  vehicle.orb_init.arg_periapsis  = trick.attach_units( "degree",100.582445989)
  vehicle.orb_init.inclination  = trick.attach_units( "degree",51.670450765)
  vehicle.orb_init.ascending_node  = trick.attach_units( "degree",49.708417385)
  vehicle.orb_init.true_anomaly  = trick.attach_units( "degree",299.884499026)
  vehicle.orb_init.planet_name      = "Earth"
  vehicle.orb_init.orbit_frame_name = "Earth.inertial"
"#;
        let rec = parse_orbital_init_py(py).unwrap();
        assert_eq!(rec.semi_major_axis, None);
        assert_eq!(rec.semi_latus_rectum, None);
        assert!((rec.alt_apoapsis.unwrap() - 363_454.582_64).abs() < 1e-6);
        assert!((rec.alt_periapsis.unwrap() - 346_073.820_40).abs() < 1e-6);
        assert_eq!(rec.eccentricity, None);
        assert!((rec.true_anomaly.unwrap() - 299.884499026_f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn parse_orbital_init_py_set05_altitudes_time_periapsis() {
        // JEOD set05 (`IncAscnodeAltperAltapoArgperTimeperi`): apo/peri altitudes
        // + time periapsis. Values are the ISS set05 deck.
        let py = r#"
  vehicle.orb_init.set              = 5
  vehicle.orb_init.alt_apoapsis  = trick.attach_units( "km",363.45458264)
  vehicle.orb_init.alt_periapsis  = trick.attach_units( "km",346.07382040)
  vehicle.orb_init.arg_periapsis  = trick.attach_units( "degree",100.582445989)
  vehicle.orb_init.inclination  = trick.attach_units( "degree",51.670450765)
  vehicle.orb_init.ascending_node  = trick.attach_units( "degree",49.708417385)
  vehicle.orb_init.time_periapsis  = trick.attach_units( "s",4581.96167293)
  vehicle.orb_init.planet_name      = "Earth"
  vehicle.orb_init.orbit_frame_name = "Earth.inertial"
"#;
        let rec = parse_orbital_init_py(py).unwrap();
        assert_eq!(rec.eccentricity, None);
        assert!((rec.alt_apoapsis.unwrap() - 363_454.582_64).abs() < 1e-6);
        assert!((rec.alt_periapsis.unwrap() - 346_073.820_40).abs() < 1e-6);
        assert!((rec.time_periapsis.unwrap() - 4581.96167293).abs() < 1e-9);
        assert_eq!(rec.true_anomaly, None);
    }

    #[test]
    fn parse_orbital_init_py_rejects_eccentricity_with_altitude_shape() {
        // The altitude shape derives eccentricity; a deck that also supplies it
        // directly is ambiguous and must be rejected.
        let py = r#"
  vehicle.orb_init.set              = 4
  vehicle.orb_init.alt_apoapsis  = trick.attach_units( "km",363.45458264)
  vehicle.orb_init.alt_periapsis  = trick.attach_units( "km",346.07382040)
  vehicle.orb_init.eccentricity  = 0.001
  vehicle.orb_init.arg_periapsis  = trick.attach_units( "degree",100.582445989)
  vehicle.orb_init.inclination  = trick.attach_units( "degree",51.670450765)
  vehicle.orb_init.ascending_node  = trick.attach_units( "degree",49.708417385)
  vehicle.orb_init.true_anomaly  = trick.attach_units( "degree",299.884499026)
  vehicle.orb_init.planet_name      = "Earth"
  vehicle.orb_init.orbit_frame_name = "Earth.inertial"
"#;
        let err = parse_orbital_init_py(py).unwrap_err();
        assert!(
            format!("{err}").contains("must not be set alongside the altitude shape"),
            "got: {err}"
        );
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
