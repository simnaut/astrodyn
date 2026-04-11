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
    let mut inertia_rows_seen = [false; 3];

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
            assert!(
                row < 3,
                "Inertia row index {} out of bounds in {}",
                row,
                path.display()
            );
            inertia[row] = [
                cap[2].parse().unwrap(),
                cap[3].parse().unwrap(),
                cap[4].parse().unwrap(),
            ];
            inertia_rows_seen[row] = true;
        }
    }

    assert!(mass.is_some(), "Missing mass in {}", path.display());
    assert!(has_position, "Missing position in {}", path.display());
    for (i, seen) in inertia_rows_seen.iter().enumerate() {
        assert!(seen, "Missing inertia row {} in {}", i, path.display());
    }

    MassInitData {
        mass: mass.unwrap(),
        position,
        inertia,
    }
}

/// Load mass initialization data from an arbitrary JEOD mass `.py` file.
///
/// If `function_name` is `Some("set_mass_iss")`, only lines within that
/// function definition are parsed (up to the next `def ` or end of file).
/// If `None`, the entire file is parsed (first match wins for each field).
///
/// # Panics
/// Panics if the file cannot be read or required fields (mass, position,
/// 3 inertia rows) are missing.
pub fn load_mass_from_file(path: &std::path::Path, function_name: Option<&str>) -> MassInitData {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    let section = match function_name {
        Some(fname) => extract_function_body(&content, fname, path),
        None => content.clone(),
    };

    parse_mass_content(&section, path)
}

/// Extract lines belonging to a Python function definition.
///
/// Returns all lines from `def <name>(...):` until the next `def ` or EOF.
fn extract_function_body(content: &str, function_name: &str, source: &std::path::Path) -> String {
    let mut lines = Vec::new();
    let mut in_function = false;
    let def_pattern = format!("def {}(", function_name);
    // Also match `def name() :`  with flexible whitespace
    let def_pattern_alt = format!("def {}", function_name);

    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&def_pattern) || trimmed.starts_with(&def_pattern_alt) {
            in_function = true;
            continue;
        }
        if in_function {
            // Next function definition ends our block
            if trimmed.starts_with("def ") {
                break;
            }
            lines.push(line);
        }
    }

    assert!(
        !lines.is_empty(),
        "Function '{}' not found in {}",
        function_name,
        source.display()
    );
    lines.join("\n")
}

/// Parse mass fields from string content.
fn parse_mass_content(content: &str, source: &std::path::Path) -> MassInitData {
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
    let mut inertia_rows_seen = [false; 3];

    for line in content.lines() {
        if let Some(cap) = mass_re.captures(line) {
            if mass.is_none() {
                mass = Some(cap[1].parse().unwrap());
            }
            continue;
        }
        if let Some(cap) = pos_re.captures(line) {
            if !has_position {
                position = [
                    cap[1].parse().unwrap(),
                    cap[2].parse().unwrap(),
                    cap[3].parse().unwrap(),
                ];
                has_position = true;
            }
            continue;
        }
        if let Some(cap) = inertia_re.captures(line) {
            let row: usize = cap[1].parse().unwrap();
            assert!(
                row < 3,
                "Inertia row index {} out of bounds in {}",
                row,
                source.display()
            );
            if !inertia_rows_seen[row] {
                inertia[row] = [
                    cap[2].parse().unwrap(),
                    cap[3].parse().unwrap(),
                    cap[4].parse().unwrap(),
                ];
                inertia_rows_seen[row] = true;
            }
        }
    }

    assert!(mass.is_some(), "Missing mass in {}", source.display());
    assert!(has_position, "Missing position in {}", source.display());
    for (i, seen) in inertia_rows_seen.iter().enumerate() {
        assert!(seen, "Missing inertia row {} in {}", i, source.display());
    }

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

    #[test]
    fn test_load_mass_from_file_with_function() {
        let content = r#"
def set_mass_iss() :
  vehicle.mass_init.properties.mass        = 400000.0
  vehicle.mass_init.properties.position    = [ -3.0, -1.5, 4.0]
  vehicle.mass_init.properties.inertia[0]  = [  1.02e+8,-6.96e+6,-5.48e+6]
  vehicle.mass_init.properties.inertia[1]  = [ -6.96e+6, 0.91e+8, 5.90e+5]
  vehicle.mass_init.properties.inertia[2]  = [ -5.48e+6, 5.90e+5, 1.64e+8]

def set_mass_cylinder() :
  vehicle.mass_init.properties.mass        = 1000.0
  vehicle.mass_init.properties.position    = [ 6.0, 0.0, 0.0]
  vehicle.mass_init.properties.inertia[0]  = [ 500.0,     0.0,     0.0]
  vehicle.mass_init.properties.inertia[1]  = [   0.0, 12250.0,     0.0]
  vehicle.mass_init.properties.inertia[2]  = [   0.0,     0.0, 12250.0]
"#;
        let tmpdir = std::env::temp_dir().join("jeod_test_mass");
        std::fs::create_dir_all(&tmpdir).unwrap();
        let path = tmpdir.join("mass.py");
        std::fs::write(&path, content).unwrap();

        // Load ISS mass
        let iss = load_mass_from_file(&path, Some("set_mass_iss"));
        assert_eq!(iss.mass, 400000.0);
        assert_eq!(iss.position, [-3.0, -1.5, 4.0]);
        assert_eq!(iss.inertia[0][0], 1.02e8);

        // Load cylinder mass
        let cyl = load_mass_from_file(&path, Some("set_mass_cylinder"));
        assert_eq!(cyl.mass, 1000.0);
        assert_eq!(cyl.position, [6.0, 0.0, 0.0]);
        assert_eq!(cyl.inertia[0][0], 500.0);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&tmpdir).ok();
    }
}
