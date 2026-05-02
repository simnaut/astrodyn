//! ISS / STS-114 reference translational-state vectors.
//!
//! Originally extracted from JEOD `Modified_data/<vehicle>/`
//! `reference_inertial_trans_state.py` (e.g.
//! `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/reference_inertial_trans_state.py`),
//! the reference state is now committed to
//! `test_data/body_init/<vehicle>.json` and read back here without
//! touching the JEOD source tree at runtime.
//!
//! Regenerate with:
//!
//! ```bash
//! cargo run -p jeod_test_data --bin extract_body_init -- \
//!     --jeod-home $JEOD_HOME
//! ```
//!
//! The Python parser still lives here (`parse_reference_state_py`) and is
//! invoked exclusively by the regen binary; runtime test paths never
//! call it.

use glam::DVec3;
use jeod_quantities::prelude::*;
use regex::Regex;

use crate::body_init_fixtures::{load_vehicle_bundle, BodyInitFixtureError, ReferenceStateRecord};

/// ISS / STS-114 reference translational state from JEOD verification data.
///
/// Frame: inertial (ICRF) — JEOD's `reference_inertial_trans_state.py` files
/// specify the state in the inertial frame. Use [`ReferenceState::position_typed`]
/// and [`ReferenceState::velocity_typed`] to obtain frame-tagged typed views.
#[derive(Debug, Clone)]
pub struct ReferenceState {
    pub position: DVec3,
    pub velocity: DVec3,
}

impl ReferenceState {
    /// Frame-tagged typed position in the inertial (ICRF) frame.
    ///
    /// This wraps the raw SI-unit `DVec3` (meters) without conversion.
    #[inline]
    pub fn position_typed(&self) -> Position<RootInertial> {
        self.position.m_at::<RootInertial>()
    }

    /// Frame-tagged typed velocity in the inertial (ICRF) frame.
    ///
    /// This wraps the raw SI-unit `DVec3` (m/s) without conversion.
    #[inline]
    pub fn velocity_typed(&self) -> Velocity<RootInertial> {
        self.velocity.m_per_s_at::<RootInertial>()
    }
}

/// Load a vehicle's reference translational state from the committed
/// `test_data/body_init/<vehicle>.json` fixture.
///
/// # Arguments
/// * `vehicle` - Vehicle directory name (e.g. `"ISS"`, `"STS_114"`).
/// * `frame` - Reference frame name. Only `"inertial"` is currently
///   committed; other values will panic with a clear regen-command
///   message.
///
/// # Panics
/// Panics if the fixture is missing, malformed, or doesn't include the
/// requested frame. The panic message names the regen command per the
/// CLAUDE.md "Fail Loudly" rule.
pub fn load_reference_state(vehicle: &str, frame: &str) -> ReferenceState {
    let bundle = load_vehicle_bundle(vehicle);
    match frame {
        "inertial" => match &bundle.reference_inertial {
            Some(state) => ReferenceState {
                position: DVec3::new(state.position[0], state.position[1], state.position[2]),
                velocity: DVec3::new(state.velocity[0], state.velocity[1], state.velocity[2]),
            },
            None => panic!(
                "body_init fixture for {vehicle} is missing `reference_inertial`. \
                 Regenerate with: cargo run -p jeod_test_data --bin extract_body_init \
                 -- --jeod-home $JEOD_HOME"
            ),
        },
        other => panic!(
            "load_reference_state: only \"inertial\" frame is committed (got {other:?} \
             for vehicle {vehicle}). Add the new frame to extract_body_init.rs and \
             regenerate the fixture."
        ),
    }
}

/// Parse `reference_inertial_trans_state.py` content into a
/// [`ReferenceStateRecord`] suitable for JSON serialization.
///
/// This is the regen-only path: the runtime [`load_reference_state`] reads
/// the committed fixture and never invokes this parser.
///
/// JEOD's `reference_*_trans_state.py` files contain Python assignments of
/// the form:
/// ```python
///   vehicle_reference.expected_state.trans.position  = [      1244540.53,   5655938.85,   3425643.22]
///   vehicle_reference.expected_state.trans.velocity  = [    -6003.833051, -1469.496044,  4590.511776]
/// ```
///
/// Returns the first two 3-element arrays found, interpreted as `(position, velocity)`.
pub fn parse_reference_state_py(
    content: &str,
) -> Result<ReferenceStateRecord, BodyInitFixtureError> {
    let array_re =
        Regex::new(r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]").unwrap();

    let mut arrays: Vec<[f64; 3]> = Vec::new();
    for cap in array_re.captures_iter(content) {
        let x: f64 = cap[1]
            .parse()
            .map_err(|e| BodyInitFixtureError::malformed(format!("parse array component: {e}")))?;
        let y: f64 = cap[2]
            .parse()
            .map_err(|e| BodyInitFixtureError::malformed(format!("parse array component: {e}")))?;
        let z: f64 = cap[3]
            .parse()
            .map_err(|e| BodyInitFixtureError::malformed(format!("parse array component: {e}")))?;
        arrays.push([x, y, z]);
    }

    if arrays.len() < 2 {
        return Err(BodyInitFixtureError::malformed(format!(
            "expected at least 2 arrays (position, velocity), got {}",
            arrays.len()
        )));
    }

    Ok(ReferenceStateRecord {
        position: arrays[0],
        velocity: arrays[1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uom::si::{length::meter, velocity::meter_per_second};

    #[test]
    fn position_typed_matches_raw_components_bit_identically() {
        let pos = DVec3::new(1244540.53, 5655938.85, 3425643.22);
        let vel = DVec3::new(-6003.833051, -1469.496044, 4590.511776);
        let state = ReferenceState {
            position: pos,
            velocity: vel,
        };

        let typed = state.position_typed();
        // Bit-identical per-component check.
        assert_eq!(typed.x.get::<meter>().to_bits(), pos.x.to_bits());
        assert_eq!(typed.y.get::<meter>().to_bits(), pos.y.to_bits());
        assert_eq!(typed.z.get::<meter>().to_bits(), pos.z.to_bits());
        // Raw-SI round-trip also bit-identical.
        assert_eq!(typed.raw_si(), pos);
    }

    #[test]
    fn velocity_typed_matches_raw_components_bit_identically() {
        let pos = DVec3::new(1244540.53, 5655938.85, 3425643.22);
        let vel = DVec3::new(-6003.833051, -1469.496044, 4590.511776);
        let state = ReferenceState {
            position: pos,
            velocity: vel,
        };

        let typed = state.velocity_typed();
        assert_eq!(typed.x.get::<meter_per_second>().to_bits(), vel.x.to_bits());
        assert_eq!(typed.y.get::<meter_per_second>().to_bits(), vel.y.to_bits());
        assert_eq!(typed.z.get::<meter_per_second>().to_bits(), vel.z.to_bits());
        assert_eq!(typed.raw_si(), vel);
    }

    #[test]
    fn typed_accessors_preserve_zero_and_negative_components() {
        let state = ReferenceState {
            position: DVec3::new(0.0, -1.5e7, 9.87654),
            velocity: DVec3::new(-7123.456, 0.0, 42.0),
        };
        let p = state.position_typed();
        let v = state.velocity_typed();
        assert_eq!(p.raw_si(), state.position);
        assert_eq!(v.raw_si(), state.velocity);
    }

    #[test]
    fn parse_reference_state_py_picks_first_two_arrays() {
        let py = "vehicle_reference.expected_state.trans.position  = [1.0, 2.0, 3.0]\n\
                  vehicle_reference.expected_state.trans.velocity  = [4.0, 5.0, 6.0]\n";
        let rec = parse_reference_state_py(py).unwrap();
        assert_eq!(rec.position, [1.0, 2.0, 3.0]);
        assert_eq!(rec.velocity, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn parse_reference_state_py_rejects_too_few_arrays() {
        let py = "x = [1.0, 2.0, 3.0]";
        let err = parse_reference_state_py(py).unwrap_err();
        assert!(format!("{err}").contains("at least 2 arrays"));
    }
}
