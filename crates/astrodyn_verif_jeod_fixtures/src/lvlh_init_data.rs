//! Parser for JEOD Trick `Modified_data/state.py`-style files that
//! configure a `DynBodyInitLvlhRotState` (LVLH-relative attitude
//! initializer) plus a translational state.
//!
//! The canonical example is
//! [`verif/SIM_dyncomp/Modified_data/state.py`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/verif/SIM_dyncomp/Modified_data/state.py),
//! which defines:
//!
//! - `set_orientation_lvlh()` — populates `vehicle.lvlh_init` with a
//!   Yaw-Pitch-Roll Euler triple and zero LVLH-relative angular
//!   velocity. JEOD's `trick.Orientation.Yaw_Pitch_Roll` maps to a
//!   ZYX Tait-Bryan rotation.
//! - `set_trans_init_typical()` / `set_trans_init_elliptical()` —
//!   populates `vehicle.trans_init.position` / `.velocity` (m, m/s)
//!   in the inertial reference frame.
//!
//! This parser strips Python-side `trick.attach_units("degree", […])`
//! wrappers (parsability tier 2 per
//! [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md#jeod-verification-data))
//! and exposes the underlying numeric arrays so a Tier 3 rig can
//! reproduce the InitLvlhRot inputs without round-tripping through a
//! JEOD CSV.

use regex::Regex;
use std::path::Path;

/// LVLH-relative rotational-init parameters extracted from
/// `set_orientation_lvlh` (or any JEOD `state.py`-style block whose
/// statements all target a common `*.lvlh_init` instance).
///
/// Mirrors the fields populated on a JEOD `DynBodyInitLvlhRotState`
/// at the level of the input.py: a Euler-angle triple plus the
/// `Yaw_Pitch_Roll`-style sequence string, and an LVLH-relative
/// angular-velocity vector. The reference orbit (`reference_position`,
/// `reference_velocity`) lives on the `trans_init` sibling and is
/// loaded via [`load_trans_init_function`].
#[derive(Debug, Clone)]
pub struct LvlhInitData {
    /// Three Euler angles in degrees, in the order indicated by
    /// [`Self::euler_sequence`]. JEOD's `trick.attach_units("degree", …)`
    /// wrapper is stripped at parse time so the values are plain
    /// degrees regardless of how the source file expressed them.
    pub euler_angles_deg: [f64; 3],
    /// Euler-sequence name as written in the JEOD source (e.g.
    /// `"Yaw_Pitch_Roll"`). Callers map this to an
    /// `astrodyn::EulerSequence` (this crate is a pure parser and
    /// does not depend on `astrodyn`, so the link is unlinked).
    pub euler_sequence: String,
    /// Angular velocity of the body wrt the LVLH frame, in rad/s.
    /// JEOD's `ang_velocity` is in rad/s with no `attach_units`
    /// wrapper at the SIM_dyncomp call sites, so the raw triple is
    /// passed through unchanged.
    pub ang_velocity: [f64; 3],
}

/// Translational-state inputs extracted from a JEOD `state.py`
/// translational-init block (e.g. `set_trans_init_typical`).
///
/// The raw `position` / `velocity` triples are in meters and m/s
/// respectively (JEOD's SIM_dyncomp helpers do not wrap them in
/// `attach_units`).
#[derive(Debug, Clone)]
pub struct TransInitData {
    /// Inertial-frame position in meters.
    pub position: [f64; 3],
    /// Inertial-frame velocity in m/s.
    pub velocity: [f64; 3],
}

/// Load a `set_orientation_lvlh`-style function from a JEOD
/// `state.py` and parse the LVLH-rot-init parameters it sets.
///
/// `function_name` selects which Python function body to scan; the
/// body extends from `def <name>(…):` up to the next `def ` or end
/// of file. This keeps multi-function `state.py` files (which is the
/// SIM_dyncomp pattern) parseable without the helper functions
/// colliding with each other.
///
/// # Panics
/// Panics with a fail-loudly diagnostic if any of the four required
/// assignments (`orientation.euler_sequence`, `orientation.euler_angles`,
/// `ang_velocity`) is missing inside the named function body. The
/// message names the source path and the missing field per the
/// CLAUDE.md "Fail Loudly" rule.
pub fn load_lvlh_init_function(path: &Path, function_name: &str) -> LvlhInitData {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", path.display()));
    let body = extract_function_body(&content, function_name, path);
    parse_lvlh_init_content(&body, path, function_name)
}

/// Load a `set_trans_init_typical`-style function from a JEOD
/// `state.py` and parse the inertial-frame position / velocity
/// triples it sets on the `trans_init` instance.
///
/// `function_name` selects which Python function body to scan, as
/// documented on [`load_lvlh_init_function`].
///
/// # Panics
/// Panics with a fail-loudly diagnostic if either `position` or
/// `velocity` is missing inside the named function body.
pub fn load_trans_init_function(path: &Path, function_name: &str) -> TransInitData {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", path.display()));
    let body = extract_function_body(&content, function_name, path);
    parse_trans_init_content(&body, path, function_name)
}

// ── private helpers ────────────────────────────────────────────────

/// Extract a Python function body, mirroring the helper shipped
/// alongside the [`mass_data`](crate::mass_data) parser. The match
/// requires the next character after the function name to be `(` or
/// whitespace so a `def set_foo` lookup cannot accidentally match
/// `def set_foo_helper`.
fn extract_function_body(content: &str, function_name: &str, source: &Path) -> String {
    let mut lines = Vec::new();
    let mut in_function = false;
    let re = Regex::new(&format!(r"^def\s+{}\s*\(", regex::escape(function_name))).unwrap();
    for line in content.lines() {
        let trimmed = line.trim_start();
        if re.is_match(trimmed) {
            in_function = true;
            continue;
        }
        if in_function {
            if trimmed.starts_with("def ") {
                break;
            }
            lines.push(line);
        }
    }
    assert!(
        !lines.is_empty(),
        "Function '{function_name}' not found in {}",
        source.display()
    );
    lines.join("\n")
}

fn parse_lvlh_init_content(content: &str, source: &Path, function_name: &str) -> LvlhInitData {
    // `orientation.euler_sequence = trick.Orientation.Yaw_Pitch_Roll`
    let seq_re =
        Regex::new(r"orientation\.euler_sequence\s*=\s*trick\.Orientation\.([A-Za-z_]+)").unwrap();
    // `orientation.euler_angles = trick.attach_units("degree", [a, b, c])`
    //
    // The inner array regex is shared with `ang_velocity`; the outer
    // wrapping `attach_units("degree", …)` is allowed but not required
    // (JEOD's set_orientation_lvlh always wraps it; the parser
    // tolerates both forms).
    let angles_re = Regex::new(
        r#"orientation\.euler_angles\s*=\s*(?:trick\.attach_units\(\s*"degree"\s*,\s*)?\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]"#,
    )
    .unwrap();
    let ang_vel_re = Regex::new(
        r"ang_velocity\s*=\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
    )
    .unwrap();

    let mut euler_sequence: Option<String> = None;
    let mut euler_angles_deg: Option<[f64; 3]> = None;
    let mut ang_velocity: Option<[f64; 3]> = None;

    for line in content.lines() {
        if let Some(cap) = seq_re.captures(line) {
            if euler_sequence.is_none() {
                euler_sequence = Some(cap[1].to_string());
            }
            continue;
        }
        if let Some(cap) = angles_re.captures(line) {
            if euler_angles_deg.is_none() {
                euler_angles_deg = Some([
                    cap[1].parse().unwrap(),
                    cap[2].parse().unwrap(),
                    cap[3].parse().unwrap(),
                ]);
            }
            continue;
        }
        if let Some(cap) = ang_vel_re.captures(line) {
            if ang_velocity.is_none() {
                ang_velocity = Some([
                    cap[1].parse().unwrap(),
                    cap[2].parse().unwrap(),
                    cap[3].parse().unwrap(),
                ]);
            }
        }
    }

    LvlhInitData {
        euler_sequence: euler_sequence.unwrap_or_else(|| {
            panic!(
                "Missing `orientation.euler_sequence` in `{function_name}` of {}: \
                 expected `<inst>.orientation.euler_sequence = trick.Orientation.<name>`",
                source.display()
            )
        }),
        euler_angles_deg: euler_angles_deg.unwrap_or_else(|| {
            panic!(
                "Missing `orientation.euler_angles` in `{function_name}` of {}: \
                 expected `<inst>.orientation.euler_angles = trick.attach_units(\"degree\", [a, b, c])` \
                 or a bare `[a, b, c]` literal",
                source.display()
            )
        }),
        ang_velocity: ang_velocity.unwrap_or_else(|| {
            panic!(
                "Missing `ang_velocity` in `{function_name}` of {}: \
                 expected `<inst>.ang_velocity = [wx, wy, wz]`",
                source.display()
            )
        }),
    }
}

fn parse_trans_init_content(content: &str, source: &Path, function_name: &str) -> TransInitData {
    let pos_re = Regex::new(
        r"\.position\s*=\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
    )
    .unwrap();
    let vel_re = Regex::new(
        r"\.velocity\s*=\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
    )
    .unwrap();

    let mut position: Option<[f64; 3]> = None;
    let mut velocity: Option<[f64; 3]> = None;
    for line in content.lines() {
        if let Some(cap) = pos_re.captures(line) {
            if position.is_none() {
                position = Some([
                    cap[1].parse().unwrap(),
                    cap[2].parse().unwrap(),
                    cap[3].parse().unwrap(),
                ]);
            }
            continue;
        }
        if let Some(cap) = vel_re.captures(line) {
            if velocity.is_none() {
                velocity = Some([
                    cap[1].parse().unwrap(),
                    cap[2].parse().unwrap(),
                    cap[3].parse().unwrap(),
                ]);
            }
        }
    }

    TransInitData {
        position: position.unwrap_or_else(|| {
            panic!(
                "Missing `.position` in `{function_name}` of {}: \
                 expected `<inst>.position = [x, y, z]`",
                source.display()
            )
        }),
        velocity: velocity.unwrap_or_else(|| {
            panic!(
                "Missing `.velocity` in `{function_name}` of {}: \
                 expected `<inst>.velocity = [vx, vy, vz]`",
                source.display()
            )
        }),
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "parser tests assert bit-exact recovery of literal Python init values"
)]
mod tests {
    use super::*;

    const SAMPLE_STATE_PY: &str = r#"
def set_trans_init_typical() :
  vehicle.trans_init.position  = [ -4292653.41, 955168.47, 5139356.57]
  vehicle.trans_init.velocity  = [ 109.649663, -7527.726490, 1484.521489]


def set_trans_init_elliptical() :
  vehicle.trans_init.position  = [ -4315967.74, 960356.20, 5167269.53]
  vehicle.trans_init.velocity  = [ 129.091037, -7491.513855, 1452.515654]


def set_orientation_lvlh():
  vehicle.lvlh_init.set_subject_body( vehicle.dyn_body )
  vehicle.lvlh_init.planet_name                = "Earth"
  vehicle.lvlh_init.body_frame_id              = "composite_body"
  vehicle.lvlh_init.orientation.data_source    = trick.Orientation.InputEulerRotation
  vehicle.lvlh_init.orientation.euler_sequence = trick.Orientation.Yaw_Pitch_Roll
  vehicle.lvlh_init.orientation.euler_angles   = trick.attach_units( "degree",[ 0.0, -11.6, 0.0])
  vehicle.lvlh_init.ang_velocity               = [ 0.0, 0.0, 0.0]

  dynamics.dyn_manager.add_body_action(vehicle.lvlh_init)
"#;

    #[test]
    fn parse_lvlh_init_extracts_typical_dyncomp_inputs() {
        let path = std::path::Path::new("test_state.py");
        let body = extract_function_body(SAMPLE_STATE_PY, "set_orientation_lvlh", path);
        let data = parse_lvlh_init_content(&body, path, "set_orientation_lvlh");
        assert_eq!(data.euler_sequence, "Yaw_Pitch_Roll");
        assert_eq!(data.euler_angles_deg, [0.0, -11.6, 0.0]);
        assert_eq!(data.ang_velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn parse_trans_init_typical_extracts_position_velocity() {
        let path = std::path::Path::new("test_state.py");
        let body = extract_function_body(SAMPLE_STATE_PY, "set_trans_init_typical", path);
        let data = parse_trans_init_content(&body, path, "set_trans_init_typical");
        assert_eq!(data.position, [-4292653.41, 955168.47, 5139356.57]);
        assert_eq!(data.velocity, [109.649663, -7527.726490, 1484.521489]);
    }

    #[test]
    fn parse_trans_init_elliptical_extracts_position_velocity() {
        let path = std::path::Path::new("test_state.py");
        let body = extract_function_body(SAMPLE_STATE_PY, "set_trans_init_elliptical", path);
        let data = parse_trans_init_content(&body, path, "set_trans_init_elliptical");
        assert_eq!(data.position, [-4315967.74, 960356.20, 5167269.53]);
        assert_eq!(data.velocity, [129.091037, -7491.513855, 1452.515654]);
    }

    #[test]
    #[should_panic(expected = "Missing `orientation.euler_angles`")]
    fn parse_lvlh_init_panics_on_missing_angles() {
        let py = "def f():\n  v.lvlh_init.orientation.euler_sequence = trick.Orientation.Yaw_Pitch_Roll\n  v.lvlh_init.ang_velocity = [0.0, 0.0, 0.0]\n";
        let path = std::path::Path::new("missing.py");
        let body = extract_function_body(py, "f", path);
        let _ = parse_lvlh_init_content(&body, path, "f");
    }
}
