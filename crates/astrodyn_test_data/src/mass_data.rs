//! Parser for JEOD Trick `Modified_data/mass/*.py` mass-initialization
//! Python files (e.g.
//! [`SIM_dyncomp/Modified_data/mass/iss_mass.py`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/verif/SIM_dyncomp/Modified_data/mass/iss_mass.py))
//! into a [`MassInitData`] struct that the test-data pipeline turns
//! into a runtime `astrodyn_dynamics::MassProperties`.
//!
//! Strips Trick's `trick.attach_units("kg", …)` / `trick.attach_units("kg*m2", …)`
//! wrappers per the parsability tier 2 documented in the project
//! [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md#jeod-verification-data).

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

/// Load mass initialization data for a vehicle from the committed
/// `test_data/body_init/<vehicle>_mass.py` fixture.
///
/// Each fixture is a verbatim copy of JEOD's
/// `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{vehicle}/mass.py`,
/// kept in plain Python so reviewers can diff against upstream. Refresh
/// after a JEOD upgrade via
/// `cargo run -p astrodyn_test_data --bin extract_jeod_validation`.
///
/// Parses Python assignments of the form:
/// - `properties.mass = 100000.0`
/// - `properties.position = [ -10.201, 0.206, 2.558]`
/// - `properties.inertia[0] = [ 7e12, 0.0, 0.0]`
///
/// # Panics
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// required fields are absent; the message includes the regen command.
pub fn load_mass_data(vehicle: &str) -> MassInitData {
    let filename = format!("body_init/{}_mass.py", vehicle.to_ascii_lowercase());
    let path = crate::tier3_csv::test_data_path(&filename);
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read {}: {e}. Regenerate with: cargo run -p astrodyn_test_data \
             --bin extract_jeod_validation",
            path.display(),
        )
    });

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
    // Match `def name(` exactly — the char after the name must be `(` or whitespace
    // to avoid prefix matches (e.g., `set_mass_iss` matching `set_mass_iss2`).
    let re = Regex::new(&format!(r"^def\s+{}\s*\(", regex::escape(function_name))).unwrap();

    for line in content.lines() {
        let trimmed = line.trim_start();
        if re.is_match(trimmed) {
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

    #[test]
    fn mass_parser_iss_spot_check() {
        let data = load_mass_data("ISS");
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
        let tmpdir = std::env::temp_dir().join(format!(
            "jeod_test_mass_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
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

        let _ = std::fs::remove_dir_all(&tmpdir);
    }
}
