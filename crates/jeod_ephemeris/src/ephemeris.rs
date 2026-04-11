use std::path::Path;

use anise::constants::celestial_objects::*;
use anise::constants::frames::*;
use anise::constants::orientations::J2000;
use anise::prelude::*;
use glam::DVec3;

use crate::bodies::EphemerisBody;

/// Planetary ephemeris reader backed by ANISE (pure Rust SPICE).
///
/// Reads standard JPL .bsp (SPK) files such as DE421 or DE440.
/// Returns positions in meters and velocities in m/s in J2000 ICRF.
pub struct Ephemeris {
    almanac: Almanac,
}

impl Ephemeris {
    /// Load an ephemeris from a .bsp file.
    pub fn from_bsp(path: &Path) -> Result<Self, EphemerisError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EphemerisError::LoadError("Path contains invalid UTF-8".to_string()))?;
        let spk = SPK::load(path_str).map_err(|e| EphemerisError::LoadError(e.to_string()))?;
        Ok(Self {
            almanac: Almanac::from_spk(spk),
        })
    }

    /// Load a Binary PCK (orientation) kernel alongside existing data.
    ///
    /// Call after `from_bsp()` to add Moon libration or other body orientation
    /// data. The BPC is merged into the almanac so `get_body_rotation()` works.
    pub fn load_bpc(&mut self, path: &Path) -> Result<(), EphemerisError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EphemerisError::LoadError("Path contains invalid UTF-8".to_string()))?;
        let bpc = BPC::load(path_str).map_err(|e| EphemerisError::LoadError(e.to_string()))?;
        let almanac = std::mem::take(&mut self.almanac);
        self.almanac = almanac.with_bpc(bpc);
        Ok(())
    }

    /// Print a summary of loaded data (SPK, BPC) for debugging.
    pub fn describe_loaded(&self) {
        self.almanac
            .describe(Some(true), Some(true), None, None, None, None, None, None);
    }

    /// Get state of `target` relative to `observer` at a given TDB Julian Date.
    ///
    /// Returns `(position_m, velocity_m_per_s)` in J2000 ICRF.
    pub fn get_state(
        &self,
        target: EphemerisBody,
        observer: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<(DVec3, DVec3), EphemerisError> {
        // Convert JD to seconds since J2000.0 TDB: (jd - 2451545.0) * 86400.0
        let tdb_s_since_j2000 = (tdb_jd - 2_451_545.0) * 86_400.0;
        let epoch = Epoch::from_tdb_seconds(tdb_s_since_j2000);
        let target_frame = body_to_frame(target);
        let observer_frame = body_to_frame(observer);

        let state = self
            .almanac
            .translate(target_frame, observer_frame, epoch, None)
            .map_err(|e| EphemerisError::QueryError(e.to_string()))?;

        // Convert km → m, km/s → m/s
        let pos_m = DVec3::new(
            state.radius_km.x * 1000.0,
            state.radius_km.y * 1000.0,
            state.radius_km.z * 1000.0,
        );
        let vel_m_s = DVec3::new(
            state.velocity_km_s.x * 1000.0,
            state.velocity_km_s.y * 1000.0,
            state.velocity_km_s.z * 1000.0,
        );

        Ok((pos_m, vel_m_s))
    }

    /// Get Earth-centered state of `target` at a given TDB Julian Date.
    ///
    /// Returns `(position_m, velocity_m_per_s)` relative to Earth center in J2000 ICRF.
    pub fn get_earth_centered_state(
        &self,
        target: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<(DVec3, DVec3), EphemerisError> {
        self.get_state(target, EphemerisBody::Earth, tdb_jd)
    }

    /// Get the rotation matrix from J2000 inertial to a body's body-fixed frame.
    ///
    /// For Moon: uses the DE421 Principal Axes (PA) frame from a BPC kernel,
    /// which must already be loaded via `load_bpc()`.
    /// For Mars: uses IAU_MARS built-in constants.
    pub fn get_body_rotation(
        &self,
        body: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<glam::DMat3, EphemerisError> {
        let tdb_s_since_j2000 = (tdb_jd - 2_451_545.0) * 86_400.0;
        let epoch = Epoch::from_tdb_seconds(tdb_s_since_j2000);

        // Use DE421 PA frame for Moon (high fidelity libration from BPC),
        // IAU built-in for other bodies.
        let orient = match body {
            EphemerisBody::Moon => 31006, // Moon PA from de421.bpc
            EphemerisBody::Mars => 499,   // IAU_MARS
            _ => {
                return Err(EphemerisError::QueryError(format!(
                    "No IAU orientation model for {body:?}"
                )));
            }
        };

        let from_frame = Frame::new(body_to_naif(body), J2000);
        let to_frame = Frame::new(body_to_naif(body), orient);

        let dcm = self
            .almanac
            .rotate(from_frame, to_frame, epoch)
            .map_err(|e| EphemerisError::QueryError(format!("Rotation query failed: {e}")))?;

        // Convert ANISE Matrix3 to glam DMat3
        // ANISE's rot_mat is column-major (same as glam)
        let m = dcm.rot_mat;
        Ok(glam::DMat3::from_cols_array(&[
            m[(0, 0)],
            m[(1, 0)],
            m[(2, 0)],
            m[(0, 1)],
            m[(1, 1)],
            m[(2, 1)],
            m[(0, 2)],
            m[(1, 2)],
            m[(2, 2)],
        ]))
    }
}

/// Map `EphemerisBody` to NAIF ID for orientation lookups.
fn body_to_naif(body: EphemerisBody) -> i32 {
    match body {
        EphemerisBody::SolarSystemBarycenter => 0,
        EphemerisBody::Mercury => MERCURY,
        EphemerisBody::Venus => VENUS,
        EphemerisBody::EarthMoonBarycenter => EARTH_MOON_BARYCENTER,
        EphemerisBody::Earth => EARTH,
        EphemerisBody::Mars => 499, // Mars body (not barycenter) — orientation frame IAU_MARS is body-fixed
        EphemerisBody::Jupiter => JUPITER_BARYCENTER,
        EphemerisBody::Saturn => SATURN_BARYCENTER,
        EphemerisBody::Uranus => URANUS_BARYCENTER,
        EphemerisBody::Neptune => NEPTUNE_BARYCENTER,
        EphemerisBody::Pluto => PLUTO_BARYCENTER,
        EphemerisBody::Moon => MOON,
        EphemerisBody::Sun => SUN,
    }
}

/// Map `EphemerisBody` to anise `Frame` constants.
fn body_to_frame(body: EphemerisBody) -> Frame {
    match body {
        EphemerisBody::SolarSystemBarycenter => SSB_J2000,
        EphemerisBody::Mercury => MERCURY_J2000,
        EphemerisBody::Venus => VENUS_J2000,
        EphemerisBody::EarthMoonBarycenter => EARTH_MOON_BARYCENTER_J2000,
        EphemerisBody::Earth => EARTH_J2000,
        EphemerisBody::Mars => MARS_BARYCENTER_J2000,
        EphemerisBody::Jupiter => JUPITER_BARYCENTER_J2000,
        EphemerisBody::Saturn => SATURN_BARYCENTER_J2000,
        EphemerisBody::Uranus => URANUS_BARYCENTER_J2000,
        EphemerisBody::Neptune => NEPTUNE_BARYCENTER_J2000,
        EphemerisBody::Pluto => Frame::new(PLUTO_BARYCENTER, J2000),
        EphemerisBody::Moon => MOON_J2000,
        EphemerisBody::Sun => SUN_J2000,
    }
}

/// Ephemeris errors.
#[derive(Debug, thiserror::Error)]
pub enum EphemerisError {
    #[error("Failed to load ephemeris file: {0}")]
    LoadError(String),
    #[error("Ephemeris query failed: {0}")]
    QueryError(String),
}
