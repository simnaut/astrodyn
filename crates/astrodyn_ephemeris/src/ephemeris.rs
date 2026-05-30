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
use astrodyn_quantities::prelude::{
    FrameTransform, Planet, PlanetFixed, Position, RootInertial, Vec3Ext, Velocity,
};
use glam::DVec3;

use crate::bodies::EphemerisBody;

/// Selects which body-fixed realization a rotation query targets.
///
/// A single body can have several body-fixed frames that differ by tens of
/// arcseconds — enough to mislocate a cartographic product (DEM, gazetteer)
/// by hundreds of metres on the surface. Rather than hard-code one choice per
/// body, [`Ephemeris::get_body_rotation_to`] takes this selector so the caller
/// picks the realization its data is referenced to.
///
/// `#[non_exhaustive]`: more body-fixed realizations will be added over time,
/// so downstream `match`es must include a wildcard arm. Adding a variant is
/// therefore a non-breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BodyFixedFrame {
    /// The body's IAU body-fixed frame (IAU 2015 rotation elements, via ANISE's
    /// built-in `IAU_<body>` orientation). Used for Mars today; extended to the
    /// other major bodies separately.
    Iau,
    /// The Moon DE421 Principal-Axes frame (NAIF 31006), from the
    /// `moon_pa_de421` BPC kernel. This is the frame the gravity field and
    /// mass concentrations are referenced to — *not* cartographic lat/lon.
    MoonPaDe421,
    /// The Moon DE421 Mean-Earth/mean-rotation frame (NAIF 31007). This is the
    /// frame lunar cartography (LOLA, SLDEM2015 DEMs, all map lat/lon) is
    /// referenced to. Reaching it requires the lunar frame kernel (PA→ME
    /// offset) to be loaded in addition to the PA BPC; without it the query
    /// fails loudly with a [`EphemerisError::QueryError`].
    MoonMeDe421,
}

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

    /// Load an ephemeris from raw .bsp bytes (e.g. a `Vec<u8>` returned
    /// by [`crate::data::load`]).
    ///
    /// Equivalent to [`from_bsp`](Self::from_bsp) but skips the filesystem
    /// lookup. Pair with `data::load(&data::DE421)` (or `DE440`) to keep
    /// the kernel-resolution policy in one place.
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

    /// Load a Binary PCK (orientation) kernel from raw bytes (e.g. a
    /// `Vec<u8>` returned by [`crate::data::load`]).
    ///
    /// Equivalent to [`load_bpc`](Self::load_bpc) but skips the filesystem
    /// lookup. Pair with `data::load(&data::MOON_PA)` to merge the Moon
    /// principal-axes orientation kernel.
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

    /// Build the ANISE [`Epoch`] for a TDB Julian Date.
    ///
    /// Hoisting this construction out of the per-query path lets callers
    /// that issue multiple queries at the same instant (e.g. the per-step
    /// ephemeris update which fetches Earth, Sun, and Moon-libration in
    /// one go) pay the `Epoch::from_tdb_seconds` +
    /// `hifitime::Epoch::to_time_scale` cost once instead of once per
    /// query. The byte-identity guarantee follows from `Epoch` being a
    /// plain value with no internal allocations: constructing once vs.
    /// three times yields the same bits.
    #[inline]
    pub fn tdb_jd_to_epoch(tdb_jd: f64) -> Epoch {
        // Convert JD to seconds since J2000.0 TDB: (jd - 2451545.0) * 86400.0
        let tdb_s_since_j2000 = (tdb_jd - 2_451_545.0) * 86_400.0;
        Epoch::from_tdb_seconds(tdb_s_since_j2000)
    }

    /// Get state of `target` relative to `observer` at a given TDB Julian Date,
    /// returning frame-tagged, dimensioned quantities in the J2000 (ICRF-aligned)
    /// inertial frame.
    ///
    /// Internal kernel below extracts SI base units from ANISE; this typed
    /// entry point wraps as `Position<RootInertial>` / `Velocity<RootInertial>`. The
    /// pre-Phase-10 bare-`f64` `get_state` was removed; use `.raw_si()` on
    /// the returned values when an unwrapped `DVec3` is needed.
    ///
    /// When issuing multiple queries at the same instant, prefer
    /// [`Self::get_state_typed_epoch`] paired with [`Self::tdb_jd_to_epoch`]
    /// to avoid rebuilding the [`Epoch`] (which internally calls
    /// `hifitime::Epoch::to_time_scale`) on every call.
    pub fn get_state_typed(
        &self,
        target: EphemerisBody,
        observer: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<(Position<RootInertial>, Velocity<RootInertial>), EphemerisError> {
        self.get_state_typed_epoch(target, observer, Self::tdb_jd_to_epoch(tdb_jd))
    }

    /// [`Self::get_state_typed`] variant that takes a pre-built [`Epoch`].
    ///
    /// Use this when a single step issues multiple ephemeris queries at
    /// the same instant: build the [`Epoch`] once with
    /// [`Self::tdb_jd_to_epoch`] and pass it to each query, amortising
    /// the `hifitime::Epoch::to_time_scale` cost across all queries.
    pub fn get_state_typed_epoch(
        &self,
        target: EphemerisBody,
        observer: EphemerisBody,
        epoch: Epoch,
    ) -> Result<(Position<RootInertial>, Velocity<RootInertial>), EphemerisError> {
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
    ///
    /// When issuing multiple queries at the same instant, prefer
    /// [`Self::get_body_rotation_epoch`] paired with
    /// [`Self::tdb_jd_to_epoch`] to avoid rebuilding the [`Epoch`].
    pub fn get_body_rotation(
        &self,
        body: EphemerisBody,
        tdb_jd: f64,
    ) -> Result<glam::DMat3, EphemerisError> {
        self.get_body_rotation_epoch(body, Self::tdb_jd_to_epoch(tdb_jd))
    }

    /// [`Self::get_body_rotation`] variant that takes a pre-built [`Epoch`].
    ///
    /// Use this when a single step issues multiple ephemeris queries at
    /// the same instant (see [`Self::get_state_typed_epoch`] for the
    /// amortisation rationale).
    pub fn get_body_rotation_epoch(
        &self,
        body: EphemerisBody,
        epoch: Epoch,
    ) -> Result<glam::DMat3, EphemerisError> {
        // Use DE421 PA frame for Moon (high fidelity libration from BPC),
        // IAU built-in for other bodies.
        // JEOD_INV: EP.17 — orientation-ephemeris analog of the body-not-in-file
        // check: a request for a body with no IAU/PA orientation model registered
        // here is refused with a QueryError rather than silently falling through
        // to ANISE with an unknown frame ID. JEOD's `de4xx_file_update.cc`
        // `item_not_in_file` covers translation queries; this arm is the same
        // intent applied to body rotations.
        let orient = match body {
            EphemerisBody::Moon => 31006, // Moon PA from de421.bpc
            EphemerisBody::Mars => 499,   // IAU_MARS
            _ => {
                return Err(EphemerisError::QueryError(format!(
                    "No IAU orientation model for {body:?}"
                )));
            }
        };

        self.body_rotation_matrix(body, orient, epoch)
    }

    /// Typed, frame-tagged sibling of [`Self::get_body_rotation_epoch`].
    ///
    /// Returns the inertial→body-fixed rotation as a
    /// [`FrameTransform<RootInertial, PlanetFixed<P>>`] — composing directly
    /// with the rest of the typed frame math instead of a bare `DMat3` the
    /// caller must re-wrap. The [`BodyFixedFrame`] selector picks *which*
    /// body-fixed realization (e.g. Moon PA vs. ME) the result targets, so the
    /// caller's cartographic data and the rotation agree on a frame.
    ///
    /// The planet tag `P` is supplied by the caller and is a pure label: this
    /// method does **not** verify that `P` corresponds to `body` (the phantom
    /// carries no runtime identity). Callers already cross this `RootInertial →
    /// PlanetFixed<P>` boundary today when they relabel ephemeris output; pass
    /// the `P` that matches `body`.
    pub fn get_body_rotation_to<P: Planet>(
        &self,
        body: EphemerisBody,
        frame: BodyFixedFrame,
        epoch: Epoch,
    ) -> Result<FrameTransform<RootInertial, PlanetFixed<P>>, EphemerisError> {
        // JEOD_INV: EP.17 — refuse (body, frame) pairs with no orientation model
        // registered here rather than handing ANISE an unknown frame ID.
        let orient = match (body, frame) {
            (EphemerisBody::Moon, BodyFixedFrame::MoonPaDe421) => 31006,
            (EphemerisBody::Moon, BodyFixedFrame::MoonMeDe421) => 31007,
            (EphemerisBody::Mars, BodyFixedFrame::Iau) => 499,
            (body, frame) => {
                return Err(EphemerisError::QueryError(format!(
                    "No orientation model for {body:?} in body-fixed frame {frame:?}"
                )));
            }
        };

        let matrix = self.body_rotation_matrix(body, orient, epoch)?;
        Ok(FrameTransform::from_matrix(matrix))
    }

    /// TDB-Julian-Date convenience wrapper over [`Self::get_body_rotation_to`].
    ///
    /// When issuing multiple queries at the same instant, prefer
    /// [`Self::get_body_rotation_to`] paired with [`Self::tdb_jd_to_epoch`] to
    /// amortise the [`Epoch`] construction.
    pub fn get_body_rotation_to_jd<P: Planet>(
        &self,
        body: EphemerisBody,
        frame: BodyFixedFrame,
        tdb_jd: f64,
    ) -> Result<FrameTransform<RootInertial, PlanetFixed<P>>, EphemerisError> {
        self.get_body_rotation_to::<P>(body, frame, Self::tdb_jd_to_epoch(tdb_jd))
    }

    /// Shared core for the rotation accessors: query ANISE for the
    /// J2000→`orient` rotation of `body` and convert it to a `glam::DMat3`.
    ///
    /// Both the bare [`Self::get_body_rotation_epoch`] and the typed
    /// [`Self::get_body_rotation_to`] route through here, so they are
    /// bit-for-bit identical for the same `orient` — the only difference is
    /// whether the caller receives a raw matrix or a wrapped
    /// [`FrameTransform`].
    fn body_rotation_matrix(
        &self,
        body: EphemerisBody,
        orient: i32,
        epoch: Epoch,
    ) -> Result<glam::DMat3, EphemerisError> {
        let from_frame = Frame::new(body_to_naif(body), J2000);
        let to_frame = Frame::new(body_to_naif(body), orient);

        // JEOD_INV: EP.17 — ANISE raises a segment-not-found or no-orientation-data
        // error when the requested body's orientation data is not loaded (e.g.
        // Moon PA without a BPC kernel); we map it through QueryError so the
        // adapter layer can escalate to a loud panic.
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

    // Querying outside the loaded SPK segment's valid epoch range surfaces
    // an `EphemerisError::QueryError` carrying ANISE's "valid from … but not
    // at requested …" message. The `unwrap()` panics with the Debug-formatted
    // error, pinning the substring proves the range check actually fires
    // rather than silently extrapolating.
    #[test]
    #[should_panic(expected = "but not at requested")]
    fn ep_14_panics_on_far_future_epoch_into_get_state_typed() {
        // JEOD_INV: EP.14 — query epoch must lie within the loaded segment's
        // valid range; far-future epochs must fail loudly, not extrapolate.
        let ephem = load_de421();
        // DE421's translation segment for Earth ends in 2053 (per ANISE).
        // 2_500_000.0 TDB JD ≈ 2132 AD — well past the loaded segment.
        let far_future_tdb_jd = 2_500_000.0_f64;
        let _ = ephem
            .get_state_typed(EphemerisBody::Moon, EphemerisBody::Earth, far_future_tdb_jd)
            .unwrap();
    }

    // Symmetric coverage for the lower bound of the valid segment: DE421's
    // translation segment begins in 1899; a TDB JD far before that must
    // surface the same `QueryError` rather than silently extrapolating
    // polynomial coefficients into invalid epochs.
    #[test]
    #[should_panic(expected = "but not at requested")]
    fn ep_14_panics_on_far_past_epoch_into_get_state_typed() {
        // JEOD_INV: EP.14 — query epoch must lie within the loaded segment's
        // valid range; far-past epochs must fail loudly, not extrapolate.
        let ephem = load_de421();
        // 2_000_000.0 TDB JD ≈ 763 BC — well before the loaded segment.
        let far_past_tdb_jd = 2_000_000.0_f64;
        let _ = ephem
            .get_state_typed(EphemerisBody::Moon, EphemerisBody::Earth, far_past_tdb_jd)
            .unwrap();
    }

    // Orientation-ephemeris availability check fires for any body that has
    // no IAU/PA orientation model wired through `get_body_rotation_epoch`'s
    // match (currently only Moon and Mars are supported). Driving the Sun
    // branch confirms the explicit QueryError-returning arm refuses to
    // silently fall through to ANISE with an unknown orientation frame ID.
    #[test]
    #[should_panic(expected = "No IAU orientation model")]
    fn ep_17_panics_on_unsupported_body_for_get_body_rotation() {
        // JEOD_INV: EP.17 — orientation-ephemeris availability for the
        // requested body must be enforced rather than silently extrapolated.
        let ephem = load_de421();
        let _ = ephem
            .get_body_rotation(EphemerisBody::Sun, J2000_TDB_JD)
            .unwrap();
    }

    // When a body *is* in the supported set (Moon → DE421 PA) but the
    // corresponding orientation kernel was never loaded, ANISE's rotation
    // engine raises a "no orientation data loaded" error which maps through
    // the rotation `map_err` site to a `QueryError`. The kernel under test
    // loads only the .bsp (no .bpc), so the Moon rotation path is forced
    // down the missing-orientation branch.
    #[test]
    #[should_panic(expected = "no orientation data loaded")]
    fn ep_17_panics_on_moon_rotation_without_bpc_loaded() {
        // JEOD_INV: EP.17 — body-orientation kernel must be loaded before
        // a rotation query for that body; missing data must surface loudly.
        let ephem = load_de421();
        let _ = ephem
            .get_body_rotation(EphemerisBody::Moon, J2000_TDB_JD)
            .unwrap();
    }

    /// Load DE421 plus the committed Moon principal-axes BPC, so Moon-PA
    /// rotation queries resolve.
    fn load_de421_with_moon_pa() -> Ephemeris {
        let mut ephem = load_de421();
        let bpc = crate::assets::moon_pa_path();
        assert!(bpc.exists(), "moon_pa BPC not found at {}", bpc.display(),);
        ephem.load_bpc(&bpc).expect("load moon_pa BPC");
        ephem
    }

    // The typed `get_body_rotation_to` and the bare `get_body_rotation_epoch`
    // route through the same private `body_rotation_matrix` core, so for the
    // same body-fixed realization (Moon PA) they must agree bit-for-bit. This
    // is what keeps the `bevy_parity_*` suite green after call sites migrate
    // from `from_matrix(get_body_rotation_epoch(..))` to the typed accessor.
    #[test]
    fn get_body_rotation_to_matches_bare_accessor_bit_for_bit_moon_pa() {
        use astrodyn_quantities::prelude::Moon;

        let ephem = load_de421_with_moon_pa();
        let epoch = Ephemeris::tdb_jd_to_epoch(J2000_TDB_JD);

        let bare = ephem
            .get_body_rotation_epoch(EphemerisBody::Moon, epoch)
            .expect("bare Moon PA rotation");
        let typed = ephem
            .get_body_rotation_to::<Moon>(EphemerisBody::Moon, BodyFixedFrame::MoonPaDe421, epoch)
            .expect("typed Moon PA rotation");

        // Compare bit patterns: the typed path must not perturb a single bit.
        let bare_bits = bare.to_cols_array().map(f64::to_bits);
        let typed_bits = typed.matrix().to_cols_array().map(f64::to_bits);
        assert_eq!(
            bare_bits, typed_bits,
            "typed accessor must reproduce the bare DMat3 bit-for-bit",
        );
    }

    // The typed accessor refuses (body, frame) pairs with no registered
    // orientation model — the EP.17 fail-loud arm for the new selector API.
    #[test]
    #[should_panic(expected = "No orientation model")]
    fn get_body_rotation_to_panics_on_unsupported_pair() {
        use astrodyn_quantities::prelude::Sun;

        let ephem = load_de421();
        let _ = ephem
            .get_body_rotation_to_jd::<Sun>(EphemerisBody::Sun, BodyFixedFrame::Iau, J2000_TDB_JD)
            .unwrap();
    }

    // Requesting the Moon ME frame without the lunar frame kernel loaded must
    // fail loudly (the PA→ME offset is not available), not silently return PA.
    #[test]
    #[should_panic(expected = "Rotation query failed")]
    fn get_body_rotation_to_moon_me_panics_without_frame_kernel() {
        use astrodyn_quantities::prelude::Moon;

        // PA BPC loaded, but no PA→ME frame kernel: ME (31007) is unreachable.
        let ephem = load_de421_with_moon_pa();
        let _ = ephem
            .get_body_rotation_to_jd::<Moon>(
                EphemerisBody::Moon,
                BodyFixedFrame::MoonMeDe421,
                J2000_TDB_JD,
            )
            .unwrap();
    }
}
