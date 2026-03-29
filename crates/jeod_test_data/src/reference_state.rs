use glam::DVec3;
use regex::Regex;

/// ISS reference translational state from JEOD verification data.
///
/// Parsed from files like:
/// `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/reference_{frame}_trans_state.py`
///
/// These files contain Python assignments of the form:
/// ```python
///   vehicle_reference.expected_state.trans.position  = [      1244540.53,   5655938.85,   3425643.22]
///   vehicle_reference.expected_state.trans.velocity  = [    -6003.833051, -1469.496044,  4590.511776]
/// ```
#[derive(Debug, Clone)]
pub struct ReferenceState {
    pub position: DVec3,
    pub velocity: DVec3,
}

/// Load an ISS reference translational state from JEOD's verification data.
///
/// # Arguments
/// * `jeod_root` - Path to the JEOD source tree root.
/// * `vehicle` - Vehicle directory name (e.g. `"ISS"`).
/// * `frame` - Reference frame name used in filename (e.g. `"inertial"`).
///
/// # Panics
/// Panics if the file cannot be read or does not contain at least two 3-element arrays
/// (position and velocity).
pub fn load_reference_state(
    jeod_root: &std::path::Path,
    vehicle: &str,
    frame: &str,
) -> ReferenceState {
    let path = jeod_root.join(format!(
        "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/reference_{}_trans_state.py",
        vehicle, frame
    ));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    let array_re =
        Regex::new(r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]").unwrap();

    let mut arrays: Vec<DVec3> = Vec::new();
    for cap in array_re.captures_iter(&content) {
        let x: f64 = cap[1].parse().unwrap();
        let y: f64 = cap[2].parse().unwrap();
        let z: f64 = cap[3].parse().unwrap();
        arrays.push(DVec3::new(x, y, z));
    }

    assert!(
        arrays.len() >= 2,
        "Expected at least 2 arrays (position, velocity) in {}",
        path.display()
    );

    ReferenceState {
        position: arrays[0],
        velocity: arrays[1],
    }
}
