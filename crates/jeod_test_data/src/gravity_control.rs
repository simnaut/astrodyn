//! Parser for JEOD Trick `input.py` gravity-control assignments.
//!
//! Reads expressions like `earth_grav_control.spherical = False` /
//! `earth_grav_control.degree = 4` from a JEOD verification sim's
//! `SET_test/RUN_*/input.py` and produces a `GravityControlConfig` that
//! the Tier 2 / Tier 3 test harness translates into a runtime
//! `jeod_gravity::GravityControl`. See
//! [`verif/SIM_dyncomp/SET_test/RUN_2/input.py`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/verif/SIM_dyncomp/SET_test/RUN_2/input.py)
//! for an example of the format we parse.

use regex::Regex;
use std::path::Path;

/// Gravity control configuration parsed from a JEOD Trick input file.
///
/// Parses assignments like:
/// - `earth_grav_control.spherical = False`
/// - `earth_grav_control.degree = 4`
/// - `earth_grav_control.order = 4`
/// - `earth_grav_control.gradient = False`
#[derive(Debug, Clone)]
pub struct GravityControlConfig {
    /// Use point-mass-only gravity (`.spherical = True/False`).
    pub spherical: bool,
    /// Spherical-harmonics degree (`.degree = N`).
    pub degree: usize,
    /// Spherical-harmonics order (`.order = N`).
    pub order: usize,
    /// Compute the gravity-gradient tensor (`.gradient = True/False`).
    pub gradient: bool,
}

impl Default for GravityControlConfig {
    /// Default matches JEOD's default: spherical (point mass), degree/order 0, no gradient.
    fn default() -> Self {
        Self {
            spherical: true,
            degree: 0,
            order: 0,
            gradient: false,
        }
    }
}

/// Parse gravity control configuration from a JEOD Trick input file.
///
/// Scans all files for `earth_grav_control.*` assignments. Accepts multiple
/// file paths to handle the common pattern where `common_input.py` exec's
/// `grav_controls.py`, then the RUN input.py overrides specific fields.
///
/// Files are processed in order; later assignments override earlier ones.
///
/// # Panics
/// Panics if any file cannot be read.
pub fn load_gravity_control(py_paths: &[&Path]) -> GravityControlConfig {
    let mut cfg = GravityControlConfig::default();

    let spherical_re = Regex::new(r"earth_grav_control\.spherical\s*=\s*(\w+)").unwrap();
    let degree_re = Regex::new(r"earth_grav_control\.degree\s*=\s*(\d+)").unwrap();
    let order_re = Regex::new(r"earth_grav_control\.order\s*=\s*(\d+)").unwrap();
    let gradient_re = Regex::new(r"earth_grav_control\.gradient\s*=\s*(\w+)").unwrap();

    for py_path in py_paths {
        let content = std::fs::read_to_string(py_path)
            .unwrap_or_else(|e| panic!("Cannot read {}: {}", py_path.display(), e));

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') {
                continue;
            }

            if let Some(cap) = spherical_re.captures(trimmed) {
                cfg.spherical = parse_python_bool(&cap[1]);
            }
            if let Some(cap) = degree_re.captures(trimmed) {
                cfg.degree = cap[1].parse().unwrap();
            }
            if let Some(cap) = order_re.captures(trimmed) {
                cfg.order = cap[1].parse().unwrap();
            }
            if let Some(cap) = gradient_re.captures(trimmed) {
                cfg.gradient = parse_python_bool(&cap[1]);
            }
        }
    }

    cfg
}

fn parse_python_bool(s: &str) -> bool {
    match s {
        "True" | "true" | "1" => true,
        "False" | "false" | "0" => false,
        _ => panic!("Cannot parse '{}' as boolean", s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gravity_control() {
        let grav_controls = r#"
vehicle.earth_grav_control.source_name = "Earth"
vehicle.earth_grav_control.spherical = True
vehicle.earth_grav_control.degree = 20
vehicle.earth_grav_control.order = 20
vehicle.earth_grav_control.gradient = True
"#;
        let run_input = r#"
# Reconfigure gravity to 4x4
vehicle.earth_grav_control.spherical = False
vehicle.earth_grav_control.degree = 4
vehicle.earth_grav_control.order  = 4

# Turn off grav gradient
vehicle.earth_grav_control.gradient  = False
"#;
        let tmpdir = std::env::temp_dir().join(format!(
            "jeod_test_grav_ctrl_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let grav_path = tmpdir.join("grav_controls.py");
        let run_path = tmpdir.join("input.py");
        std::fs::write(&grav_path, grav_controls).unwrap();
        std::fs::write(&run_path, run_input).unwrap();

        let cfg = load_gravity_control(&[grav_path.as_path(), run_path.as_path()]);
        assert!(!cfg.spherical);
        assert_eq!(cfg.degree, 4);
        assert_eq!(cfg.order, 4);
        assert!(!cfg.gradient);

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_default_is_spherical() {
        let cfg = GravityControlConfig::default();
        assert!(cfg.spherical);
        assert_eq!(cfg.degree, 0);
        assert_eq!(cfg.order, 0);
        assert!(!cfg.gradient);
    }
}
