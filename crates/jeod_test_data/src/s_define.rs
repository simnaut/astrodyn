use regex::Regex;
use std::path::Path;

/// Parse the dynamics integration step size from a JEOD S_define file.
///
/// Looks for `#define DYNAMICS <float>` and returns the value in seconds.
///
/// # Panics
/// Panics if the file cannot be read or does not contain a `#define DYNAMICS` line.
pub fn load_dynamics_dt(s_define_path: &Path) -> f64 {
    let content = std::fs::read_to_string(s_define_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", s_define_path.display(), e));

    let re = Regex::new(r"#define\s+DYNAMICS\s+([\d.eE+-]+)").unwrap();

    for line in content.lines() {
        if let Some(cap) = re.captures(line) {
            return cap[1].parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Cannot parse DYNAMICS value '{}' in {}: {}",
                    &cap[1],
                    s_define_path.display(),
                    e
                )
            });
        }
    }

    panic!("No #define DYNAMICS found in {}", s_define_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dynamics_dt() {
        let content = r#"
//=============================================================================
// S_define for SIM_dyncomp
//=============================================================================

#define DYNAMICS 0.03125   // Vehicle and planetary dynamics interval (32Hz)
#define LOW_RATE_ENV 5400.00 // Ephemeris update

#include "sim_objects/default_trick_sys.sm"
"#;
        let tmpdir = std::env::temp_dir().join("jeod_test_s_define");
        std::fs::create_dir_all(&tmpdir).unwrap();
        let path = tmpdir.join("S_define");
        std::fs::write(&path, content).unwrap();

        let dt = load_dynamics_dt(&path);
        assert!((dt - 0.03125).abs() < 1e-15, "Expected 0.03125, got {}", dt);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&tmpdir).ok();
    }

    #[test]
    fn test_parse_dynamics_dt_integer() {
        let content = "#define DYNAMICS 1.0\n";
        let tmpdir = std::env::temp_dir().join("jeod_test_s_define2");
        std::fs::create_dir_all(&tmpdir).unwrap();
        let path = tmpdir.join("S_define");
        std::fs::write(&path, content).unwrap();

        let dt = load_dynamics_dt(&path);
        assert!((dt - 1.0).abs() < 1e-15, "Expected 1.0, got {}", dt);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&tmpdir).ok();
    }
}
