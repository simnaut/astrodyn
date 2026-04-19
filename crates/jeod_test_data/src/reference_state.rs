use glam::DVec3;
use jeod_quantities::prelude::*;
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
    pub fn position_typed(&self) -> Position<Inertial> {
        self.position.m_at::<Inertial>()
    }

    /// Frame-tagged typed velocity in the inertial (ICRF) frame.
    ///
    /// This wraps the raw SI-unit `DVec3` (m/s) without conversion.
    #[inline]
    pub fn velocity_typed(&self) -> Velocity<Inertial> {
        self.velocity.m_per_s_at::<Inertial>()
    }
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
}
