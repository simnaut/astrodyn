use regex::Regex;
use std::f64::consts::PI;

/// Orbital element initialization data from a JEOD Trick input file.
///
/// Parsed from files like:
/// `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/trans_Orbit_{frame}_body_{init_name}.py`
///
/// These files contain Python assignments in two forms:
/// - `key = trick.attach_units("unit", value)` (with unit conversion)
/// - `key = value` (bare numeric)
///
/// Unit conversions applied automatically:
/// - `"degree"` -> radians (multiply by PI/180)
/// - `"km"` -> meters (multiply by 1000)
/// - `"s"` -> seconds (no conversion)
#[derive(Debug, Clone)]
pub struct OrbitalInitData {
    pub semi_major_axis: f64, // meters (converted from km)
    pub eccentricity: f64,
    pub inclination: f64,            // radians (converted from degrees)
    pub ascending_node: f64,         // radians
    pub arg_periapsis: f64,          // radians
    pub time_periapsis: Option<f64>, // seconds
    pub mean_anomaly: Option<f64>,   // radians
    pub true_anomaly: Option<f64>,   // radians
    pub planet_name: String,
    pub reference_frame: String,
}

/// Load orbital element initialization data from a JEOD Trick input file.
///
/// # Arguments
/// * `jeod_root` - Path to the JEOD source tree root.
/// * `vehicle` - Vehicle directory name (e.g. `"ISS"`).
/// * `init_name` - Init file identifier (e.g. `"trans_Orbit_inertial_body_set01"`).
///
/// # Panics
/// Panics if the file cannot be read, or required fields (semi_major_axis, eccentricity,
/// inclination, ascending_node, arg_periapsis) are missing.
pub fn load_orbital_init(
    jeod_root: &std::path::Path,
    vehicle: &str,
    init_name: &str,
) -> OrbitalInitData {
    let path = jeod_root.join(format!(
        "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/{}.py",
        vehicle, init_name
    ));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

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
            let raw_val: f64 = cap[3].parse().unwrap();

            let val = convert_units(raw_val, unit);

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
            let val: f64 = cap[2].parse().unwrap();

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

    OrbitalInitData {
        semi_major_axis: semi_major_axis
            .unwrap_or_else(|| panic!("Missing semi_major_axis in {}", path.display())),
        eccentricity: eccentricity
            .unwrap_or_else(|| panic!("Missing eccentricity in {}", path.display())),
        inclination: inclination
            .unwrap_or_else(|| panic!("Missing inclination in {}", path.display())),
        ascending_node: ascending_node
            .unwrap_or_else(|| panic!("Missing ascending_node in {}", path.display())),
        arg_periapsis: arg_periapsis
            .unwrap_or_else(|| panic!("Missing arg_periapsis in {}", path.display())),
        time_periapsis,
        mean_anomaly,
        true_anomaly,
        planet_name,
        reference_frame,
    }
}

/// Convert a raw value from the given unit string to SI (meters, radians, seconds).
fn convert_units(val: f64, unit: &str) -> f64 {
    match unit {
        "degree" => val * PI / 180.0,
        "km" => val * 1000.0,
        "s" => val,
        "m" => val,
        "rad" => val,
        other => panic!("Unknown unit: {}", other),
    }
}

/// Direct Cartesian translational-state initialization data from a JEOD
/// `trans_TransState_*.py` file.
///
/// Parsed from files like:
/// `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/trans_TransState_{frame}_body.py`
///
/// Contains position/velocity vectors with `trick.attach_units("m", [...])` and
/// `trick.attach_units("m/s", [...])` wrappers.
#[derive(Debug, Clone)]
pub struct TransStateData {
    pub position: [f64; 3], // meters
    pub velocity: [f64; 3], // m/s
    pub reference_frame: String,
}

/// Load a direct Cartesian translational-state init file.
///
/// # Arguments
/// * `jeod_root` - Path to the JEOD source tree root.
/// * `vehicle` - Vehicle directory name (e.g. `"STS_114"`).
/// * `init_name` - Init file identifier without `.py` (e.g. `"trans_TransState_inertial_body"`).
///
/// # Panics
/// Panics if the file cannot be read or position/velocity cannot be parsed.
pub fn load_trans_state(
    jeod_root: &std::path::Path,
    vehicle: &str,
    init_name: &str,
) -> TransStateData {
    let path = jeod_root.join(format!(
        "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/{}.py",
        vehicle, init_name
    ));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

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
            let x: f64 = cap[3].parse().unwrap();
            let y: f64 = cap[4].parse().unwrap();
            let z: f64 = cap[5].parse().unwrap();
            // "m" for position, "m/s" for velocity — no scale conversion needed.
            assert!(
                unit == "m" || unit == "m/s",
                "Unexpected unit '{}' in {}",
                unit,
                path.display()
            );
            match field {
                "position" => position = Some([x, y, z]),
                "velocity" => velocity = Some([x, y, z]),
                _ => {}
            }
            continue;
        }
        if let Some(cap) = frame_re.captures(line) {
            reference_frame = Some(cap[1].to_string());
        }
    }

    TransStateData {
        position: position.unwrap_or_else(|| panic!("Missing position in {}", path.display())),
        velocity: velocity.unwrap_or_else(|| panic!("Missing velocity in {}", path.display())),
        reference_frame: reference_frame
            .unwrap_or_else(|| panic!("Missing reference_ref_frame_name in {}", path.display())),
    }
}
