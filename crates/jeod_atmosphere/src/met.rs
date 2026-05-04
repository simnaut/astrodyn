//! MET (Marshall Engineering Thermosphere) atmosphere model.
//!
//! Faithful port of JEOD's MET atmosphere implementation, which is based on:
//! - Jacchia, L.G., "New Static Models of the Thermosphere and Exosphere with
//!   Empirical Temperature Profiles", SAO Special Report No. 313, 1970.
//! - Jacchia, L.G., "Revised Static Models of the Thermosphere and Exosphere
//!   with Empirical Temperature Profiles", SAO Report No. 332, 1971.
//!
//! JEOD uses primarily the 1970 paper's constants/formulations, with selective
//! use of the 1971 paper for seasonal-latitude density variations.

use crate::AtmosphereState;
use jeod_quantities::frame::SelfPlanet;

// ---------------------------------------------------------------------------
// Gauss quadrature tables (from JEOD utils/math/gauss_quadrature.cc)
// Index 0 is unused; index N gives the N-point quadrature.
// ---------------------------------------------------------------------------

const GAUSS_WEIGHTS: [[f64; 8]; 9] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5555556, 0.8888889, 0.5555556, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        0.3478548, 0.6521452, 0.6521452, 0.3478548, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269, 0.0, 0.0, 0.0,
    ],
    [
        0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245, 0.0, 0.0,
    ],
    [
        0.1294850, 0.2797054, 0.3818301, 0.4179592, 0.3818301, 0.2797054, 0.1294850, 0.0,
    ],
    [
        0.1012285, 0.2223810, 0.3137067, 0.3626838, 0.3626838, 0.3137067, 0.2223810, 0.1012285,
    ],
];

const GAUSS_XVALUES: [[f64; 8]; 9] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.5773503, 0.5773503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.7745967, 0.0, 0.7745967, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        -0.8611363, -0.3399810, 0.3399810, 0.8611363, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        -0.9061798, -0.5384693, 0.0, 0.5384693, 0.9061798, 0.0, 0.0, 0.0,
    ],
    [
        -0.9324695, -0.6612094, -0.2386192, 0.2386192, 0.6612094, 0.9324695, 0.0, 0.0,
    ],
    [
        -0.9491079, -0.7415312, -0.4058452, 0.0, 0.4058452, 0.7415312, 0.9491079, 0.0,
    ],
    [
        -0.9602899, -0.7966665, -0.5255324, -0.1834346, 0.1834346, 0.5255324, 0.7966665, 0.9602899,
    ],
];

// ---------------------------------------------------------------------------
// Physical constants (from JEOD, matching Jacchia's values -- NOT modern CODATA)
// ---------------------------------------------------------------------------

/// Gas constant used by Jacchia. See Jacchia(1971) p9 eq 5.
/// Matches JEOD `METAtmosphere::R_gas_constant`.
const R_GAS_CONSTANT: f64 = 8.31432;
/// Days per year (Jacchia's value).
const DAYS_PER_YEAR: f64 = 365.2422;
/// Avogadro's number as used by Jacchia. See Jacchia(1971) p7 eq 4.
const AVOGADRO: f64 = 6.02257e23;
/// 2 * pi (Jacchia's value, truncated).
// Jacchia's truncated 2π (intentionally not std::f64::consts::TAU for JEOD fidelity).
#[allow(clippy::approx_constant)]
const TWO_PI: f64 = 6.28318531;
/// 3/2 * pi (Jacchia's value, truncated).
const THREE_PI_TWO: f64 = 4.71238898;
/// Degree to radian conversion (Jacchia's value).
const DEG_TO_RAD: f64 = 0.017453293;
/// Minutes per day.
const MINUTES_PER_DAY: f64 = 1440.0;

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

/// Mean molar mass at the barometric ceiling and higher. See Jacchia(1970) eq 1.
const MOL_WEIGHT_BAROMETRIC_CEILING: f64 = 27.72594278125;

/// Altitude at which to start fairing between no He seasonal-latitude variation
/// and full variation (which starts at 500 km).
const BASE_FAIRING_HEIGHT: f64 = 440.0;

/// Barometric equation ceiling (105 km per 1970 paper).
const BAROMETRIC_EQUATION_CEILING: f64 = 105.0;

/// Molecular weight polynomial coefficients from Jacchia(1970) p4 eq 1.
const MOL_WT_COEFFS: [f64; 7] = [
    28.15204, -0.085586, 1.284e-4, -1.0056e-5, -1.021e-5, 1.5044e-6, 9.9826e-8,
];

/// Altitude bin boundaries for Gauss quadrature integration.
const GAUSS_ALTITUDES: [f64; 9] = [
    90.0, 105.0, 125.0, 160.0, 200.0, 300.0, 500.0, 1500.0, 2500.0,
];

/// Gauss quadrature order for each altitude bin.
const GAUSS_N: [usize; 8] = [4, 5, 6, 6, 6, 6, 6, 6];

/// Number of chemical species tracked.
const NUM_SPECIES: usize = 6;

/// Volume fractions at sea level. From Jacchia(1971) p5.
/// Order: N2, O2, O, Ar, He, H
const SPECIES_FRAC: [f64; NUM_SPECIES] = [
    0.78110,   // N2
    0.20955,   // O2
    0.0,       // O (atomic oxygen)
    0.0093432, // Ar
    1.289e-05, // He (1970 value)
    0.0,       // H
];

/// Molecular weights (molar masses) in g/mol for each species.
const SPECIES_MOL_WEIGHT: [f64; NUM_SPECIES] = [
    28.0134, // N2
    31.9988, // O2
    15.9994, // O
    39.948,  // Ar
    4.0026,  // He
    1.00797, // H
];

/// Nominal molecular weight of the atmosphere at sea level.
const NOMINAL_MOL_WEIGHT: f64 = 28.96;

// ---------------------------------------------------------------------------
// Temperature model constants
// ---------------------------------------------------------------------------

/// k_1: parameter for first coefficient of temperature polynomial;
/// also the temperature gradient at 125 km.
const THERMAL_K1: f64 = 0.054285714;
/// k_3: parameter for third coefficient of temperature polynomial.
const THERMAL_K3: f64 = -3.96501457725948e-5;
/// k_4: parameter for fourth coefficient of temperature polynomial.
const THERMAL_K4: f64 = -5.3311120366514e-7;
/// Temperature at 90 km reference point.
const T_90: f64 = 183.0;

// ---------------------------------------------------------------------------
// Fairing factor
// ---------------------------------------------------------------------------

/// pi/2 / (500 - 440) = pi/2 / 60
const FAIRING_K: f64 = std::f64::consts::FRAC_PI_2 / (500.0 - BASE_FAIRING_HEIGHT);

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Type of geomagnetic index provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeoIndexType {
    /// Ap formulation for geomagnetic variation in temperature.
    Ap,
    /// Kp formulation for geomagnetic variation in temperature.
    Kp,
}

/// Solar activity preset: solar minimum conditions.
pub const SOLAR_MIN: MetAtmosphere = MetAtmosphere {
    f10: 70.0,
    f10b: 70.0,
    geo_index: 0.0,
    geo_index_type: GeoIndexType::Ap,
};

/// Solar activity preset: solar mean conditions.
pub const SOLAR_MEAN: MetAtmosphere = MetAtmosphere {
    f10: 128.8,
    f10b: 128.8,
    geo_index: 15.7,
    geo_index_type: GeoIndexType::Ap,
};

/// Solar activity preset: solar maximum conditions.
pub const SOLAR_MAX: MetAtmosphere = MetAtmosphere {
    f10: 250.0,
    f10b: 250.0,
    geo_index: 25.0,
    geo_index_type: GeoIndexType::Ap,
};

/// MET (Marshall Engineering Thermosphere) atmosphere model configuration.
///
/// Implements the Jacchia 1970/1971 thermospheric density model as ported from
/// NASA JEOD. Provides atmospheric density, temperature, and pressure as a
/// function of altitude, latitude, longitude, and time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetAtmosphere {
    /// 10.7 cm solar radio flux in Solar Flux Units (SFU).
    pub f10: f64,
    /// 81-day (or 90-day) average of f10.
    pub f10b: f64,
    /// Geomagnetic activity index (Ap or Kp, depending on `geo_index_type`).
    pub geo_index: f64,
    /// Which geomagnetic index formulation to use.
    pub geo_index_type: GeoIndexType,
}

/// Extended MET atmosphere output, including species number densities.
#[derive(Debug, Clone, Copy)]
pub struct MetAtmosphereStateVars {
    /// Base atmospheric state (density, temperature, pressure, wind).
    pub base: AtmosphereState<SelfPlanet>,
    /// Exospheric temperature in K.
    pub exo_temp: f64,
    /// Log10 of total density.
    pub log10_dens: f64,
    /// Mean molecular weight in g/mol.
    pub mol_weight: f64,
    /// N2 number density in 1/m^3.
    pub n2: f64,
    /// O2 number density in 1/m^3.
    /// Matches JEOD `METAtmosphereStateVars::Ox2`.
    pub ox2: f64,
    /// Atomic oxygen number density in 1/m^3.
    /// Matches JEOD `METAtmosphereStateVars::Ox`.
    pub ox: f64,
    /// Argon number density in 1/m^3.
    /// Matches JEOD `METAtmosphereStateVars::A`.
    pub a: f64,
    /// Helium number density in 1/m^3.
    pub he: f64,
    /// Atomic hydrogen number density in 1/m^3.
    /// Matches JEOD `METAtmosphereStateVars::Hyd`.
    pub hyd: f64,
}

impl MetAtmosphere {
    /// Compute atmospheric state at the given position and time.
    ///
    /// # Arguments
    /// * `altitude_km` - Geodetic altitude in kilometres.
    /// * `latitude_rad` - Geodetic latitude in radians, range [-pi/2, pi/2].
    /// * `longitude_rad` - Geodetic longitude in radians.
    /// * `truncated_julian_day` - Truncated Julian Time (TJT), defined as
    ///   `MJD - 40000` (Modified Julian Date minus 40000).
    ///
    /// # Returns
    /// An [`AtmosphereState`] with density (kg/m^3), temperature (K), pressure (Pa),
    /// and zero wind.
    pub fn density(
        &self,
        altitude_km: f64,
        latitude_rad: f64,
        longitude_rad: f64,
        truncated_julian_day: f64,
    ) -> AtmosphereState<SelfPlanet> {
        self.density_full(
            altitude_km,
            latitude_rad,
            longitude_rad,
            truncated_julian_day,
        )
        .base
    }

    /// Convenience wrapper that accepts altitude in **metres** (SI).
    ///
    /// Internally the MET model works in kilometres (the 1970s-era
    /// Jacchia convention). This method performs the m→km conversion
    /// so callers that use SI throughout do not need to.
    pub fn density_si(
        &self,
        altitude_m: f64,
        latitude_rad: f64,
        longitude_rad: f64,
        truncated_julian_day: f64,
    ) -> AtmosphereState<SelfPlanet> {
        self.density(
            altitude_m / 1000.0,
            latitude_rad,
            longitude_rad,
            truncated_julian_day,
        )
    }

    /// Compute extended atmospheric state including species number densities.
    pub fn density_full(
        &self,
        altitude_km: f64,
        latitude_rad: f64,
        longitude_rad: f64,
        truncated_julian_day: f64,
    ) -> MetAtmosphereStateVars {
        let mut ctx = ComputeContext::new(self, altitude_km, latitude_rad, longitude_rad);
        ctx.compute_solar_angles(truncated_julian_day);
        ctx.compute_exospheric_temperature(self);
        ctx.jacchia();
        ctx.modify_densities();

        // Finalize
        let log10_dens = ctx.density.log10();
        // Pressure: p = (rho * 1000 / M) * R * T
        // The 1000 converts g/mol to kg/mol to cancel with density in kg/m^3.
        let pressure = (ctx.density * 1000.0 / ctx.mol_weight) * R_GAS_CONSTANT * ctx.temperature;

        MetAtmosphereStateVars {
            base: AtmosphereState::<SelfPlanet>::from_raw(
                ctx.density,
                ctx.temperature,
                pressure,
                glam::DVec3::ZERO,
            ),
            exo_temp: ctx.exo_temp,
            log10_dens,
            mol_weight: ctx.mol_weight,
            n2: ctx.num_density[0],
            ox2: ctx.num_density[1],
            ox: ctx.num_density[2],
            a: ctx.num_density[3],
            he: ctx.num_density[4],
            hyd: ctx.num_density[5],
        }
    }
}

// ===========================================================================
// Internal computation context (replaces mutable state on the C++ class)
// ===========================================================================

struct ComputeContext {
    altitude_km: f64,
    latitude: f64,
    longitude: f64,

    // Time-derived quantities
    fraction_of_year: f64,
    solar_declination_angle: f64,
    solar_hour_angle: f64,

    // State variables
    exo_temp: f64,
    temperature: f64,
    density: f64,
    mol_weight: f64,

    // Species number densities [N2, O2, O, Ar, He, H]
    num_density: [f64; NUM_SPECIES],
}

impl ComputeContext {
    fn new(_atmos: &MetAtmosphere, altitude_km: f64, latitude: f64, longitude: f64) -> Self {
        Self {
            altitude_km,
            latitude,
            longitude,
            fraction_of_year: 0.0,
            solar_declination_angle: 0.0,
            solar_hour_angle: 0.0,
            exo_temp: 0.0,
            temperature: 0.0,
            density: 0.0,
            mol_weight: 0.0,
            num_density: [0.0; NUM_SPECIES],
        }
    }

    // -----------------------------------------------------------------------
    // Temperature model (METAtmosphereThermal)
    // -----------------------------------------------------------------------

    /// Compute T_125 from the exospheric temperature. See eq 9 in Jacchia(1970).
    fn compute_t125(&self) -> f64 {
        // 1970 formulation:
        444.3807 + 0.02385 * self.exo_temp - 392.8292 * (-0.0021357 * self.exo_temp).exp()
    }

    /// Compute temperature at a given altitude. Eq 10 (below 125 km) and eq 13 (above).
    fn compute_temperature_at(&self, alt_km: f64, t_125: f64) -> f64 {
        let dz = alt_km - 125.0;
        let dz_2 = dz * dz;
        let dt = t_125 - T_90;

        if dz <= 0.0 {
            // Polynomial for 90-125 km range (eq 10)
            let dz_3 = dz * dz_2;
            let dz_4 = dz_2 * dz_2;
            t_125 + dt * (THERMAL_K1 * dz + THERMAL_K3 * dz_3 + THERMAL_K4 * dz_4)
        } else {
            // Asymptotic approach to exospheric temperature (eq 13)
            let dz_2_5 = dz_2 * dz.sqrt();
            let coeff_a = 2.0 * (self.exo_temp - t_125) / std::f64::consts::PI;
            // Magic number 4.5E-6 is empirical "B" in eq 13.
            t_125 + coeff_a * (THERMAL_K1 * dt * dz * (1.0 + 4.5e-6 * dz_2_5)).atan2(coeff_a)
        }
    }

    // -----------------------------------------------------------------------
    // Solar angles computation (atmos_MET_TME)
    // -----------------------------------------------------------------------

    fn compute_solar_angles(&mut self, trunc_julian_time: f64) {
        // ---- PART A: time representations ----
        let tjt_prev_midnight = trunc_julian_time.floor();
        let fraction_of_day = trunc_julian_time - tjt_prev_midnight;

        // Compute year, day-of-year from TJT.
        // TJT = MJD - 40000; 2000-01-01 00:00 has TJT = 11544.0.
        // We replicate JEOD's iterative year-tracking logic.
        let tjt_year_start: f64 = 11544.0; // TJT of 2000-01-01 00:00
        let mut year: i32 = 2000;
        let mut max_days_this_year: i32 = 366; // 2000 is a leap year

        let mut day_of_year = (tjt_prev_midnight - tjt_year_start + 1.0) as i32;

        while day_of_year > max_days_this_year {
            day_of_year -= max_days_this_year;
            year += 1;
            max_days_this_year = if year % 4 == 0 { 366 } else { 365 };
        }
        while day_of_year < 0 {
            year -= 1;
            max_days_this_year = if year % 4 == 0 { 366 } else { 365 };
            day_of_year += max_days_this_year;
        }

        self.fraction_of_year = day_of_year as f64 / DAYS_PER_YEAR;

        // Days since 1900-01-01. TJT epoch offset: 24980 days from 1900-01-01 to TJT epoch.
        let century_days = tjt_prev_midnight + 24980.0;
        let century_frac = (century_days + 0.5) / 36525.0;

        // FMJD: days since 1956-12-31 00:00:00 (TJT of that date is -4162).
        let fmjd = 4162.0 + tjt_prev_midnight + fraction_of_day;

        // ---- PART B: solar declination angle ----
        let b1: f64 = 0.0172028;
        let b2: f64 = 0.0335;
        let b3: f64 = 1.407;

        let celestial_longitude =
            (b1 * fmjd + b2 * (0.017202 * (fmjd - 3.0)).sin() - b3).rem_euclid(TWO_PI);

        let dec_angle_const = (23.4523 - 0.013 * century_frac) * DEG_TO_RAD;
        self.solar_declination_angle = (celestial_longitude.sin() * dec_angle_const.sin()).asin();

        // ---- PART C: solar hour angle ----
        // C-1: right ascension of sun
        let scratch1 = self.solar_declination_angle.tan();
        let scratch2 = dec_angle_const.tan();

        let mut solar_right_ascension = std::f64::consts::FRAC_PI_2;
        if scratch1.abs() < scratch2.abs() {
            solar_right_ascension = (scratch1 / scratch2).asin().abs();
        }

        // Put RA in same quadrant as celestial longitude
        if celestial_longitude > THREE_PI_TWO {
            solar_right_ascension = TWO_PI - solar_right_ascension;
        } else if celestial_longitude > std::f64::consts::PI {
            solar_right_ascension += std::f64::consts::PI;
        } else if celestial_longitude > std::f64::consts::FRAC_PI_2 {
            solar_right_ascension = std::f64::consts::PI - solar_right_ascension;
        }

        // C-2: compute solar hour angle
        let minutes_of_day = fraction_of_day * MINUTES_PER_DAY;
        let a1: f64 = 99.6909833;
        let a2: f64 = 36000.76892;
        let a3: f64 = 0.00038708;
        let a4: f64 = 0.250684477;

        let greenwich_mean_position =
            (a1 + a2 * century_frac + a3 * century_frac * century_frac + a4 * minutes_of_day)
                .rem_euclid(360.0);

        let right_ascension_point = greenwich_mean_position * DEG_TO_RAD + self.longitude;
        self.solar_hour_angle = right_ascension_point - solar_right_ascension;

        while self.solar_hour_angle > std::f64::consts::PI {
            self.solar_hour_angle -= TWO_PI;
        }
        while self.solar_hour_angle < -std::f64::consts::PI {
            self.solar_hour_angle += TWO_PI;
        }
    }

    // -----------------------------------------------------------------------
    // Exospheric temperature (atmos_MET_TINF)
    // -----------------------------------------------------------------------

    fn compute_exospheric_temperature(&mut self, atmos: &MetAtmosphere) {
        // PART A: solar-activity variation (eq 14, 1970 constants)
        let c1: f64 = 383.0;
        let c2: f64 = 3.32;
        let c3: f64 = 1.80;
        let solar_activity_variation = c1 + c2 * atmos.f10b + c3 * (atmos.f10 - atmos.f10b);

        // PART B: diurnal variation (Jacchia 1971 p28)
        let beta: f64 = -0.6457718;
        let gamma: f64 = 0.7504916;
        let p: f64 = 0.1047198;
        let re: f64 = 0.31;

        let theta = 0.5 * (self.latitude + self.solar_declination_angle).abs();
        let eta = 0.5 * (self.latitude - self.solar_declination_angle).abs();
        let mut tau = self.solar_hour_angle + beta + p * (self.solar_hour_angle + gamma).sin();

        if tau > std::f64::consts::PI {
            tau -= TWO_PI;
        } else if tau < -std::f64::consts::PI {
            tau += TWO_PI;
        }

        let sin_theta = theta.sin();
        let cos_eta = eta.cos();
        let cos_tau_2 = (tau / 2.0).cos();

        // 1970: power of sin_theta and cos_eta is 2.5
        let a1 = sin_theta * sin_theta * sin_theta.sqrt();
        let a2 = cos_eta * cos_eta * cos_eta.sqrt();
        let a3 = cos_tau_2 * cos_tau_2 * cos_tau_2;

        let diurnal_variation = 1.0 + re * (a1 + a3 * (a2 - a1));

        // PART C: geomagnetic variation (1970 formulation)
        let geomagnetic_variation = match atmos.geo_index_type {
            GeoIndexType::Kp => 28.0 * atmos.geo_index + 0.03 * atmos.geo_index.exp(),
            GeoIndexType::Ap => {
                1.0 * atmos.geo_index + 100.0 * (1.0 - (-0.08 * atmos.geo_index).exp())
            }
        };

        // PART D: semiannual variation (eq 23, 1970 formulation)
        let e1: f64 = 2.41;
        let e2: f64 = 0.349;
        let e3: f64 = 0.206;
        let e4: f64 = 3.9531708; // 226.5 degrees in radians
        let e5: f64 = 4.3214352; // 247.6 degrees in radians
        let e6: f64 = 0.1145;
        let e7: f64 = 5.974262; // 342.3 degrees in radians
        let e8: f64 = 2.16;

        let tau0 = ((1.0 + (std::f64::consts::PI * 2.0 * self.fraction_of_year + e7).sin()) / 2.0)
            .powf(e8);
        let tau1 = self.fraction_of_year + e6 * (tau0 - 0.5);

        let sav_a = e2 + e3 * (2.0 * std::f64::consts::PI * tau1 + e4).sin();
        let sav_b = (4.0 * std::f64::consts::PI * tau1 + e5).sin();
        let semiannual_variation = e1 + atmos.f10b * sav_a * sav_b;

        self.exo_temp = solar_activity_variation * diurnal_variation
            + geomagnetic_variation
            + semiannual_variation;
    }

    // -----------------------------------------------------------------------
    // Main Jacchia routine (atmos_MET_JAC)
    // -----------------------------------------------------------------------

    fn jacchia(&mut self) {
        // Compute temperatures
        let t_125 = self.compute_t125();
        self.temperature = self.compute_temperature_at(self.altitude_km, t_125);

        // ---- PART A: Barometric equation (below barometric ceiling) ----
        let mut integration_ceiling_a = self.altitude_km;
        let mut temperature_ceiling_a = self.temperature;

        if self.altitude_km < BAROMETRIC_EQUATION_CEILING {
            self.mol_weight = self.compute_mol_wt(self.altitude_km);
        } else {
            integration_ceiling_a = BAROMETRIC_EQUATION_CEILING;
            self.mol_weight = MOL_WEIGHT_BAROMETRIC_CEILING;
            temperature_ceiling_a = self.compute_temperature_at(BAROMETRIC_EQUATION_CEILING, t_125);
        }

        let integral_mg_rt =
            self.apply_gauss_quadrature(0, integration_ceiling_a, t_125) / R_GAS_CONSTANT;

        // Magic number 2.1926E-5: (rho_0 * T_0 / M_0) at 90 km, converted to kg/m^3.
        self.density =
            2.1926e-5 * (self.mol_weight / temperature_ceiling_a) * (-integral_mg_rt).exp();

        // Compute particle densities
        let scratch = AVOGADRO * self.density * 1000.0;
        let nominal_particle_density = scratch / NOMINAL_MOL_WEIGHT;
        let actual_particle_density = scratch / self.mol_weight;

        // N2
        self.num_density[0] = SPECIES_FRAC[0] * nominal_particle_density;
        // O2
        self.num_density[1] = actual_particle_density
            * ((self.mol_weight / NOMINAL_MOL_WEIGHT) * (1.0 + SPECIES_FRAC[1]) - 1.0);
        // O (atomic oxygen)
        self.num_density[2] =
            2.0 * actual_particle_density * (1.0 - self.mol_weight / NOMINAL_MOL_WEIGHT);
        // Ar
        self.num_density[3] = SPECIES_FRAC[3] * nominal_particle_density;
        // He
        self.num_density[4] = SPECIES_FRAC[4] * nominal_particle_density;
        // H (placeholder, not valid below 500 km)
        self.num_density[5] = 1.0;

        // ---- PART B: Diffusion equation (above barometric ceiling) ----
        if self.altitude_km > BAROMETRIC_EQUATION_CEILING {
            let integral_g_rt =
                self.apply_gauss_quadrature(1, self.altitude_km, t_125) / R_GAS_CONSTANT;
            let temp_ratio = temperature_ceiling_a / self.temperature;

            // Adjust non-Hydrogen species
            for (ii, &mol_wt) in SPECIES_MOL_WEIGHT.iter().enumerate().take(5) {
                self.num_density[ii] *= temp_ratio * (-mol_wt * integral_g_rt).exp();
            }
            // He gets an extra alpha = -0.38 term
            self.num_density[4] *= temp_ratio.powf(-0.38);
        }

        // ---- PART C: Hydrogen above 500 km ----
        if self.altitude_km > 500.0 {
            let temperature_500 = self.compute_temperature_at(500.0, t_125);
            let log_temperature_500 = temperature_500.log10();

            let integral_g_rt =
                self.apply_gauss_quadrature(6, self.altitude_km, t_125) / R_GAS_CONSTANT;

            // Magic numbers from Kockarts and Nicolet (1962, 1963).
            // 79.13 = 73.13 + 6 to go from cm^-3 to m^-3.
            self.num_density[5] = 10.0_f64.powf(
                79.13 - 39.4 * log_temperature_500
                    + 5.5 * log_temperature_500 * log_temperature_500,
            ) * (temperature_500 / self.temperature)
                * (-SPECIES_MOL_WEIGHT[5] * integral_g_rt).exp();
        }

        // Clamp number densities to minimum of 1.0
        for nd in &mut self.num_density {
            if *nd < 1.0 {
                *nd = 1.0;
            }
        }

        // ---- PART D: Wrap up for altitudes above barometric ceiling ----
        if self.altitude_km > BAROMETRIC_EQUATION_CEILING {
            let mut weighted_num_density = 0.0;
            let mut total_num_density = 0.0;
            for (ii, &mol_wt) in SPECIES_MOL_WEIGHT.iter().enumerate().take(NUM_SPECIES) {
                weighted_num_density += mol_wt * self.num_density[ii];
                total_num_density += self.num_density[ii];
            }
            self.mol_weight = weighted_num_density / total_num_density;
            // Convert density to kg/m^3
            self.density = weighted_num_density / (1000.0 * AVOGADRO);
        }
    }

    // -----------------------------------------------------------------------
    // Density modifications
    // -----------------------------------------------------------------------

    fn modify_densities(&mut self) {
        if self.altitude_km <= 170.0 {
            self.compute_seasonal_latitude_variation();
        } else if self.altitude_km >= 500.0 {
            self.compute_seasonal_lat_variation_he();
        } else if self.altitude_km > BASE_FAIRING_HEIGHT {
            // Between 440 km and 500 km: fair helium
            self.atmos_met_fair5();
        }
    }

    /// Seasonal-latitude density variation in the lower thermosphere (90-170 km).
    /// From Jacchia(1971) eq 24.
    fn compute_seasonal_latitude_variation(&mut self) {
        let z = self.altitude_km - 90.0;
        let exp_arg = -0.0013 * z * z;
        let p = (2.0 * std::f64::consts::PI * self.fraction_of_year + 1.72).sin();
        let sin_lat = self.latitude.sin();
        let s_lat_2 = sin_lat * sin_lat;
        let s = 0.014 * z * exp_arg.exp();

        let mut d_log_rho = s * p * s_lat_2;

        if self.latitude < 0.0 {
            d_log_rho = -d_log_rho;
        }

        self.density *= 10.0_f64.powf(d_log_rho);
    }

    /// Seasonal-latitude variation of helium number density.
    /// From Jacchia(1971) eq 25. Applicable above ~500 km.
    fn compute_seasonal_lat_variation_he(&mut self) {
        // 0.4091 radians = 23.44 degrees (obliquity of the equator)
        let a = (0.65 * (self.solar_declination_angle / 0.4091)).abs();
        let mut b = 0.5 * self.latitude;
        if self.solar_declination_angle < 0.0 {
            b = -b;
        }
        let sin_x = (std::f64::consts::FRAC_PI_4 - b).sin();
        let sin_x_3 = sin_x * sin_x * sin_x;
        // 0.35355 = sin^3(pi/4)
        let delta_log_he_density = a * (sin_x_3 - 0.35355);

        let delta_he_num_density =
            self.num_density[4] * (10.0_f64.powf(delta_log_he_density) - 1.0);
        self.num_density[4] += delta_he_num_density;

        // 6.646E-27 = (mol_weight_He / Avogadro) * (1 kg / 1000 g)
        self.density += 6.646e-27 * delta_he_num_density;
    }

    /// Fair helium variation between base_fairing_height (440 km) and 500 km.
    fn atmos_met_fair5(&mut self) {
        let he_num_density_pre = self.num_density[4];
        let mass_density_pre = self.density;

        self.compute_seasonal_lat_variation_he();

        let a = (FAIRING_K * (self.altitude_km - BASE_FAIRING_HEIGHT)).cos();
        let czi = a * a;

        self.density *= (mass_density_pre / self.density).powf(czi);
        self.num_density[4] *= (he_num_density_pre / self.num_density[4]).powf(czi);
    }

    // -----------------------------------------------------------------------
    // Molecular weight polynomial (atmos_MET_MOL_WT)
    // -----------------------------------------------------------------------

    fn compute_mol_wt(&self, alt_km: f64) -> f64 {
        let alt_km_clamped = if alt_km < 90.0 { 90.0 } else { alt_km };

        if alt_km_clamped > BAROMETRIC_EQUATION_CEILING {
            return MOL_WEIGHT_BAROMETRIC_CEILING;
        }

        // Reference altitude for polynomial: 100 km (1970 paper)
        let reference_altitude = 100.0;
        let offset = (alt_km_clamped - reference_altitude).max(90.0 - reference_altitude);

        let mut mol_weight = MOL_WT_COEFFS[0];
        let mut offset_to_n = 1.0;
        for coeff in &MOL_WT_COEFFS[1..7] {
            offset_to_n *= offset;
            mol_weight += coeff * offset_to_n;
        }
        mol_weight
    }

    // -----------------------------------------------------------------------
    // Gauss quadrature integration (atmos_MET_GAUSS)
    // -----------------------------------------------------------------------

    fn apply_gauss_quadrature(
        &self,
        altitude_index_start: usize,
        altitude_end: f64,
        t_125: f64,
    ) -> f64 {
        let include_mu = altitude_index_start == 0;
        let mut integral = 0.0;

        let mut ii = altitude_index_start;
        while ii < 8 && GAUSS_ALTITUDES[ii] < altitude_end {
            let gauss_order = GAUSS_N[ii];
            let alt_lo = GAUSS_ALTITUDES[ii];
            let alt_hi = altitude_end.min(GAUSS_ALTITUDES[ii + 1]);
            let half_cell_height = 0.5 * (alt_hi - alt_lo);

            let mut integral_cell = 0.0;
            for kk in 0..gauss_order {
                let alt_eval_point =
                    alt_lo + half_cell_height * (1.0 + GAUSS_XVALUES[gauss_order][kk]);

                // g = g_surface / r_G^2 where r_G = 1 + alt/R_earth
                // Magic numbers: 9.80665 m/s^2 surface gravity, 6356.766 km earth radius
                let rad_eval_point = 1.0 + alt_eval_point / 6356.766e0;
                let grav = 9.80665 / (rad_eval_point * rad_eval_point);

                let temp_at_point = self.compute_temperature_at(alt_eval_point, t_125);
                let mut value = grav / temp_at_point;

                if include_mu {
                    value *= self.compute_mol_wt(alt_eval_point);
                }

                integral_cell += GAUSS_WEIGHTS[gauss_order][kk] * value;
            }

            integral_cell *= half_cell_height;
            integral += integral_cell;

            ii += 1;
        }

        integral
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative TJT for testing: 2020-06-15 12:00 UTC.
    /// TJT epoch is 1957-05-24. Days from 1957-05-24 to 2020-06-15 = 23034 days approx.
    /// More precisely: 2020-01-01 has TJT = 22867.0, June 15 = day 167, noon = +0.5
    const TJT_2020_JUN_15_NOON: f64 = 22867.0 + 167.0 + 0.5;

    /// ISS-like latitude (51.6 degrees) in radians.
    const ISS_LAT: f64 = 51.6 * std::f64::consts::PI / 180.0;

    /// ISS-like longitude (arbitrary).
    const ISS_LON: f64 = 0.0;

    #[test]
    fn density_decreases_with_altitude() {
        let atmos = SOLAR_MEAN;
        let d200 = atmos
            .density(200.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d400 = atmos
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d600 = atmos
            .density(600.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;

        assert!(
            d200 > d400,
            "Density at 200 km ({d200:e}) should exceed density at 400 km ({d400:e})"
        );
        assert!(
            d400 > d600,
            "Density at 400 km ({d400:e}) should exceed density at 600 km ({d600:e})"
        );
    }

    #[test]
    fn higher_f10_gives_higher_density() {
        let d_min = SOLAR_MIN
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d_mean = SOLAR_MEAN
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d_max = SOLAR_MAX
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;

        assert!(
            d_min < d_mean,
            "Solar min density ({d_min:e}) should be less than solar mean ({d_mean:e})"
        );
        assert!(
            d_mean < d_max,
            "Solar mean density ({d_mean:e}) should be less than solar max ({d_max:e})"
        );
    }

    #[test]
    fn density_at_iss_altitude_solar_min() {
        let d = SOLAR_MIN
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;

        assert!(
            d > 1e-14 && d < 1e-10,
            "Solar min density at 400 km = {d:e}, expected O(1e-13 to 1e-11)"
        );
    }

    #[test]
    fn density_at_iss_altitude_solar_mean() {
        let d = SOLAR_MEAN
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;

        assert!(
            d > 1e-13 && d < 1e-9,
            "Solar mean density at 400 km = {d:e}, expected O(1e-12 to 1e-10)"
        );
    }

    #[test]
    fn density_at_iss_altitude_solar_max() {
        let d = SOLAR_MAX
            .density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;

        assert!(
            d > 1e-12 && d < 1e-9,
            "Solar max density at 400 km = {d:e}, expected O(1e-11 to 1e-10)"
        );
    }

    #[test]
    fn temperature_increases_with_solar_activity() {
        let t_min = SOLAR_MIN
            .density_full(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .base
            .temperature;
        let t_max = SOLAR_MAX
            .density_full(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .base
            .temperature;

        assert!(
            t_max > t_min,
            "Solar max temperature ({t_max} K) should exceed solar min ({t_min} K)"
        );
        // Exospheric temperature should be in hundreds to low-thousands of K
        assert!(
            t_min > 500.0,
            "Solar min temp at 400 km = {t_min} K, too low"
        );
        assert!(
            t_max < 3000.0,
            "Solar max temp at 400 km = {t_max} K, too high"
        );
    }

    #[test]
    fn pressure_is_positive() {
        let state = SOLAR_MEAN.density(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        assert!(
            state.pressure > 0.0,
            "Pressure should be positive, got {}",
            state.pressure
        );
    }

    #[test]
    fn species_densities_are_positive() {
        let state = SOLAR_MEAN.density_full(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        assert!(state.n2 >= 1.0);
        assert!(state.ox2 >= 1.0);
        assert!(state.ox >= 1.0);
        assert!(state.a >= 1.0);
        assert!(state.he >= 1.0);
    }

    #[test]
    fn hydrogen_above_500km() {
        let state = SOLAR_MEAN.density_full(600.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        assert!(
            state.hyd > 1.0,
            "Hydrogen number density above 500 km should be computed, got {}",
            state.hyd
        );
    }

    #[test]
    fn hydrogen_below_500km_is_placeholder() {
        let state = SOLAR_MEAN.density_full(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        assert!(
            (state.hyd - 1.0).abs() < 1e-10,
            "Hydrogen below 500 km should be 1.0, got {}",
            state.hyd
        );
    }

    #[test]
    fn exospheric_temperature_range() {
        // Exospheric temperature should be in a physically reasonable range
        let state_min = SOLAR_MIN.density_full(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        let state_max = SOLAR_MAX.density_full(400.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);

        assert!(
            state_min.exo_temp > 500.0 && state_min.exo_temp < 1500.0,
            "Solar min exo temp = {} K",
            state_min.exo_temp
        );
        assert!(
            state_max.exo_temp > 1000.0 && state_max.exo_temp < 3000.0,
            "Solar max exo temp = {} K",
            state_max.exo_temp
        );
    }

    #[test]
    fn mol_weight_at_90km() {
        // At 90 km, offset = 90 - 100 = -10. The mol_wt_coeffs polynomial should give
        // a value near 28-29 g/mol.
        let ctx = ComputeContext::new(&SOLAR_MEAN, 90.0, 0.0, 0.0);
        let mw = ctx.compute_mol_wt(90.0);
        assert!(
            mw > 27.0 && mw < 30.0,
            "Molecular weight at 90 km = {} g/mol",
            mw
        );
    }

    #[test]
    fn temperature_at_90km_is_183k() {
        // At 90 km, temperature should be T_90 = 183 K regardless of exospheric temp.
        let mut ctx = ComputeContext::new(&SOLAR_MEAN, 400.0, 0.0, 0.0);
        ctx.exo_temp = 1000.0;
        let t_125 = ctx.compute_t125();
        let t_at_90 = ctx.compute_temperature_at(90.0, t_125);
        assert!(
            (t_at_90 - 183.0).abs() < 0.5,
            "Temperature at 90 km should be ~183 K, got {} K",
            t_at_90
        );
    }

    #[test]
    fn different_times_give_different_densities() {
        // Diurnal variation should produce different densities at different times of day
        let tjt_midnight = 22867.0 + 167.0; // midnight
        let tjt_noon = tjt_midnight + 0.5; // noon

        let d_midnight = SOLAR_MEAN
            .density(400.0, ISS_LAT, ISS_LON, tjt_midnight)
            .density;
        let d_noon = SOLAR_MEAN
            .density(400.0, ISS_LAT, ISS_LON, tjt_noon)
            .density;

        // They should be different (diurnal variation)
        assert!(
            (d_midnight - d_noon).abs() / d_midnight > 0.01,
            "Densities at midnight ({d_midnight:e}) and noon ({d_noon:e}) should differ significantly"
        );
    }

    #[test]
    fn density_at_various_altitudes_monotonic() {
        let atmos = SOLAR_MEAN;
        let alts = [100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0];
        let densities: Vec<f64> = alts
            .iter()
            .map(|&alt| {
                atmos
                    .density(alt, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
                    .density
            })
            .collect();

        for i in 1..densities.len() {
            assert!(
                densities[i] < densities[i - 1],
                "Density should decrease monotonically: at {} km = {:e}, at {} km = {:e}",
                alts[i - 1],
                densities[i - 1],
                alts[i],
                densities[i]
            );
        }
    }

    #[test]
    fn fairing_region_smooth() {
        // Density should transition smoothly through the fairing region (440-500 km)
        let atmos = SOLAR_MEAN;
        let d430 = atmos
            .density(430.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d460 = atmos
            .density(460.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d500 = atmos
            .density(500.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;
        let d540 = atmos
            .density(540.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON)
            .density;

        assert!(d430 > d460, "430 > 460");
        assert!(d460 > d500, "460 > 500");
        assert!(d500 > d540, "500 > 540");
    }

    #[test]
    fn equatorial_vs_polar_density() {
        // At low altitudes, density should vary with latitude due to seasonal effects
        let atmos = SOLAR_MEAN;
        let d_equator = atmos.density(120.0, 0.0, 0.0, TJT_2020_JUN_15_NOON).density;
        let d_polar = atmos.density(120.0, 1.2, 0.0, TJT_2020_JUN_15_NOON).density;

        // They should differ (seasonal-latitude variation)
        assert!(
            (d_equator - d_polar).abs() / d_equator > 1e-4,
            "Equatorial ({d_equator:e}) and polar ({d_polar:e}) densities should differ"
        );
    }

    #[test]
    fn density_si_matches_density_with_conversion() {
        let atmos = SOLAR_MEAN;
        let alt_m = 400_000.0; // 400 km in metres
        let result_si = atmos.density_si(alt_m, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        let result_km = atmos.density(alt_m / 1000.0, ISS_LAT, ISS_LON, TJT_2020_JUN_15_NOON);
        assert_eq!(result_si.density, result_km.density);
        assert_eq!(result_si.temperature, result_km.temperature);
        assert_eq!(result_si.pressure, result_km.pressure);
    }
}
