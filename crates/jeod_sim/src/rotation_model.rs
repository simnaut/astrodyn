/// Rotation model for a gravity source's planet-fixed frame.
///
/// Determines how `t_inertial_pfix` is updated each step. Each planet has its
/// own rotation model; point-mass sources use `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RotationModel {
    /// No rotation — point-mass source or body without a planet-fixed frame.
    #[default]
    None,
    /// Earth rotation via IAU 2000A precession-nutation + GAST + optional polar
    /// motion. Uses the simulation's `gmst_seconds`, `tt_tjt`, and `polar_motion`.
    EarthRNP,
    /// Mars rotation via IAU pole orientation + spin + nutation Fourier series.
    /// Uses the simulation's TT seconds since J2000 (matching JEOD's RNPMars).
    MarsIAU,
    /// Moon rotation via IAU 2009 pole + prime meridian model.
    /// Uses the simulation's TDB seconds.
    MoonIAU,
    /// Moon rotation from DE421 BPC libration data (high-fidelity).
    /// Requires the simulation's `ephemeris` field to be set with BPC loaded.
    MoonDE421,
}
