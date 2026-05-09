//! [`Ephemeris`] reader and the [`EphemerisError`] failure type.
//!
//! Ports the kernel-loader / state-query surface of
//! [`models/environment/ephemerides/de4xx_ephem/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/ephemerides/de4xx_ephem/)
//! from JEOD v5.4.0. JEOD links a hand-rolled binary loader to JPL DE405 /
//! DE421 kernels; this crate delegates the file format and Chebyshev
//! evaluation to the `anise` crate (a Rust SPICE/NAIF reimplementation) and
//! exposes a thin frame-tagged wrapper.
//!
//! Outputs are wrapped as [`Position<Inertial>`] / [`Velocity<Inertial>`]
//! from [`astrodyn_quantities`] in the J2000 ICRF (meters and m/s).

use std::path::Path;

use anise::constants::celestial_objects::*;
use anise::constants::frames::*;
use anise::constants::orientations::J2000;
use anise::prelude::*;
use astrodyn_quantities::prelude::{Position, RootInertial, Vec3Ext, Velocity};
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

    /// Load an ephemeris from raw .bsp bytes (e.g. an `include_bytes!` blob).
    ///
    /// Equivalent to [`from_bsp`](Self::from_bsp) but skips the filesystem
    /// lookup. Use this with [`crate::data::DE421_BSP`] to load the
    /// embedded DE421 kernel without any path resolution.
    pub fn from_bsp_bytes(bytes: &[u8]) -> Result<Self, EphemerisError> {
        let spk = SPK::parse(bytes).map_err(|e| EphemerisError::LoadError(e.to_string()))?;
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

    /// Load a Binary PCK (orientation) kernel from raw bytes
    /// (e.g. an `include_bytes!` blob).
    ///
    /// Equivalent to [`load_bpc`](Self::load_bpc) but skips the filesystem
    /// lookup. Use this with [`crate::data::MOON_PA_BPC`] to merge the
    /// embedded Moon principal-axes orientation kernel.
    pub fn load_bpc_bytes(&mut self, bytes: &[u8]) -> Result<(), EphemerisError> {
        let bpc = BPC::parse(bytes).map_err(|e| EphemerisError::LoadError(e.to_string()))?;
        let almanac = std::mem::take(&mut self.almanac);
        self.almanac = almanac.with_bpc(bpc);
        Ok(())
    }

    /// Print a summary of loaded data (SPK, BPC) for debugging.
    pub fn describe_loaded(&self) {
        self.almanac
            .describe(Some(true), Some(true), None, None, None, None, None, None);
    }

    /// Get state of `target` relative to `observer` at a given TDB Julian Date,
    /// returning frame-tagged, dimensioned quantities in the J2000 (ICRF-aligned)
    /// inertial frame.
    ///
    /// Internal kernel below extracts SI base units from ANISE; this typed
    /// entry point wraps as `Position<RootInertial>` / `Velocity<RootInertial>`. The
    /// pre-Phase-10 bare-`f64` `get_state` was removed; use `.raw_si()` on
    /// the returned values when an unwrapped `DVec3` is needed.
    pub fn get_state_typed(
        &self,
        target: EphemerisBody,
        observer: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<(Position<RootInertial>, Velocity<RootInertial>), EphemerisError> {
        // Convert JD to seconds since J2000.0 TDB: (jd - 2451545.0) * 86400.0
        let tdb_s_since_j2000 = (tdb_jd - 2_451_545.0) * 86_400.0;
        let epoch = Epoch::from_tdb_seconds(tdb_s_since_j2000);
        let target_frame = body_to_frame(target);
        let observer_frame = body_to_frame(observer);

        // JEOD_INV: EP.14 — query epoch must lie within the loaded SPK segment range.
        // JEOD_INV: EP.17 — body ephemeris must be available for the requested body.
        // ANISE surfaces both as a translate error which we map into QueryError.
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

        Ok((
            pos_m.m_at::<RootInertial>(),
            vel_m_s.m_per_s_at::<RootInertial>(),
        ))
    }

    /// Earth-centered variant of [`Self::get_state_typed`].
    ///
    /// Returns `(Position<RootInertial>, Velocity<RootInertial>)` relative to Earth
    /// center in J2000 ICRF (meters, m/s).
    pub fn get_earth_centered_state_typed(
        &self,
        target: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<(Position<RootInertial>, Velocity<RootInertial>), EphemerisError> {
        self.get_state_typed(target, EphemerisBody::Earth, tdb_jd)
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
// JEOD_INV: EP.25 — ephemeris errors surface through two variants: LoadError for kernel
// load failures (EP.11-13 aggregated), QueryError for out-of-range / missing-body / rotation
// failures (EP.14, EP.17 aggregated). JEOD uses distinct message codes; we aggregate.
#[derive(Debug, thiserror::Error)]
pub enum EphemerisError {
    /// SPK / BPC kernel could not be loaded — bad path, wrong format,
    /// or unreadable bytes. Aggregates JEOD's `EP.11`-`EP.13` load-time
    /// failure codes.
    #[error("Failed to load ephemeris file: {0}")]
    LoadError(String),
    /// Translation or rotation query failed — unsupported body, epoch
    /// out of segment range, or missing orientation kernel.
    /// Aggregates JEOD's `EP.14` (epoch range) and `EP.17`
    /// (body availability) failure codes.
    #[error("Ephemeris query failed: {0}")]
    QueryError(String),
}

#[cfg(test)]
mod typed_accessor_tests {
    //! Smoke tests for the typed accessors using the committed `de421.bsp`
    //! kernel in `test_data/`; the test will panic with a descriptive
    //! message if the kernel is missing (no graceful skip — see project
    //! policy).
    use super::*;

    const J2000_TDB_JD: f64 = 2_451_545.0;

    fn load_de421() -> Ephemeris {
        let path = crate::assets::de421_path();
        assert!(
            path.exists(),
            "DE421.bsp not found at {}. Download with: curl -Lo crates/astrodyn_ephemeris/assets/de421.bsp \
             https://public-data.nyxspace.com/anise/de421.bsp",
            path.display(),
        );
        Ephemeris::from_bsp(&path).expect("load DE421.bsp")
    }

    #[test]
    fn get_state_typed_smoke_test_moon_at_j2000() {
        let ephem = load_de421();
        let (pos_t, vel_t) = ephem
            .get_state_typed(EphemerisBody::Moon, EphemerisBody::Earth, J2000_TDB_JD)
            .expect("query Moon state (typed)");
        // Earth-Moon distance at J2000 ≈ 4.024e8 m; speed ≈ 1.02 km/s.
        let r_km = pos_t.raw_si().length() / 1000.0;
        let v_km_s = vel_t.raw_si().length() / 1000.0;
        assert!(
            (r_km - 402_449.0).abs() < 1.0,
            "Earth-Moon distance: {r_km:.1} km",
        );
        assert!(
            (v_km_s - 1.02).abs() < 0.1,
            "Moon orbital speed: {v_km_s:.4} km/s",
        );
    }

    #[test]
    fn get_earth_centered_state_typed_smoke_test_sun_at_j2000() {
        let ephem = load_de421();
        let (pos_t, _vel_t) = ephem
            .get_earth_centered_state_typed(EphemerisBody::Sun, J2000_TDB_JD)
            .expect("query Sun state (typed)");
        // Earth-Sun distance at J2000 ≈ 0.9833 AU (perihelion-adjacent).
        let r_au = pos_t.raw_si().length() / 1.496e11;
        assert!(
            (r_au - 0.9833).abs() < 0.01,
            "Earth-Sun distance: {r_au} AU"
        );
    }
}
