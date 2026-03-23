use regex::Regex;

/// Mass initialization data from a JEOD Trick mass.py file.
#[derive(Debug, Clone)]
pub struct MassInitData {
    /// Total mass in kg.
    pub mass: f64,
    /// Center of mass position in structural frame (m).
    pub position: [f64; 3],
    /// Inertia tensor in body frame (kg*m^2), row-major.
    pub inertia: [[f64; 3]; 3],
}

/// Load mass initialization data from a JEOD mass.py file.
///
/// File location: `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/mass.py`
///
/// Parses Python assignments of the form:
/// - `properties.mass = 100000.0`
/// - `properties.position = [ -10.201, 0.206, 2.558]`
/// - `properties.inertia[0] = [ 7e12, 0.0, 0.0]`
///
/// # Panics
/// Panics if the file cannot be read or required fields are missing.
pub fn load_mass_data(jeod_root: &std::path::Path, vehicle: &str) -> MassInitData {
    let path = jeod_root.join(format!(
        "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/mass.py",
        vehicle
    ));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    let mass_re = Regex::new(r"\.properties\.mass\s*=\s*([-\d.eE+]+)").unwrap();
    let pos_re =
        Regex::new(r"\.properties\.position\s*=\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]")
            .unwrap();
    let inertia_re =
        Regex::new(r"\.properties\.inertia\[(\d+)\]\s*=\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]")
            .unwrap();

    let mut mass: Option<f64> = None;
    let mut position = [0.0; 3];
    let mut inertia = [[0.0; 3]; 3];
    let mut has_position = false;

    for line in content.lines() {
        if let Some(cap) = mass_re.captures(line) {
            mass = Some(cap[1].parse().unwrap());
            continue;
        }
        if let Some(cap) = pos_re.captures(line) {
            position = [
                cap[1].parse().unwrap(),
                cap[2].parse().unwrap(),
                cap[3].parse().unwrap(),
            ];
            has_position = true;
            continue;
        }
        if let Some(cap) = inertia_re.captures(line) {
            let row: usize = cap[1].parse().unwrap();
            inertia[row] = [
                cap[2].parse().unwrap(),
                cap[3].parse().unwrap(),
                cap[4].parse().unwrap(),
            ];
        }
    }

    assert!(mass.is_some(), "Missing mass in {}", path.display());
    assert!(has_position, "Missing position in {}", path.display());

    MassInitData {
        mass: mass.unwrap(),
        position,
        inertia,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jeod_path;

    #[test]
    fn mass_parser_iss_spot_check() {
        let root = jeod_path();
        if !root.exists() {
            panic!(
                "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
                root.display()
            );
        }
        let data = load_mass_data(&root, "ISS");
        assert_eq!(data.mass, 100000.0, "ISS mass should be 100000 kg");
        assert_eq!(data.position[0], -10.201, "ISS CoM x");
        assert_eq!(data.position[1], 0.206, "ISS CoM y");
        assert_eq!(data.position[2], 2.558, "ISS CoM z");
        assert_eq!(data.inertia[0][0], 7e12, "ISS Ixx");
        assert_eq!(data.inertia[1][1], 12e12, "ISS Iyy");
        assert_eq!(data.inertia[2][2], 10e12, "ISS Izz");
    }
}
