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

impl RotationModel {
    /// Planet angular velocity about the spin axis (rad/s), if the model has
    /// a constant rate.
    ///
    /// JEOD sets `ang_vel_this = [0, 0, planet_omega]` on the pfix frame's
    /// rotational state. Earth and Mars have constant spin rates from JEOD
    /// data files (`data_rnp_j2000.cc`, `data_rnp_mars.cc`). Moon models
    /// have time-varying angular velocity due to libration; they return
    /// `None` here — callers should compute angular velocity from the
    /// rotation derivative when needed.
    pub fn planet_omega(&self) -> Option<f64> {
        match self {
            // JEOD: RNPJ2000_ptr->planet_omega = 7.292115146706388e-5
            Self::EarthRNP => Some(7.292_115_146_706_388e-5),
            // JEOD: RNPMars_ptr->planet_omega = 350.891985303 * deg/day → rad/s
            // 350.891985303 * (π/180) / 86400
            Self::MarsIAU => Some(350.891_985_303 * std::f64::consts::PI / 180.0 / 86400.0),
            // Moon libration — angular velocity is not constant.
            Self::MoonIAU | Self::MoonDE421 => None,
            Self::None => None,
        }
    }
}
