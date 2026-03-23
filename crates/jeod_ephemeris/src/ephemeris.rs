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
        let path_str = path.to_str().ok_or_else(|| {
            EphemerisError::LoadError("Path contains invalid UTF-8".to_string())
        })?;
        let spk = SPK::load(path_str).map_err(|e| EphemerisError::LoadError(e.to_string()))?;
        Ok(Self {
            almanac: Almanac::from_spk(spk),
        })
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
}

/// Map `EphemerisBody` to anise `Frame` constants.
fn body_to_frame(body: EphemerisBody) -> Frame {
    match body {
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
