//! The `F64Ext` facade and its vector/array companions.
//!
//! Makes mission-crate code read like physics:
//!
//! ```
//! use jeod_quantities::prelude::*;
//! let altitude = 400.0.km();
//! let inclination = 51.6.deg();
//! let mass = 420_000.0.kg();
//! ```

use core::marker::PhantomData;

use glam::DVec3;
use uom::si::{
    acceleration::{meter_per_second_squared, standard_gravity},
    angle::{degree, minute as arcminute, radian, second as arcsecond},
    angular_acceleration::radian_per_second_squared,
    angular_velocity::{degree_per_second, radian_per_second, revolution_per_minute},
    area::{square_kilometer, square_meter},
    f64::{
        Acceleration, Angle, AngularAcceleration, AngularVelocity, Area, Force, Frequency, Length,
        Mass, Ratio, Time, Torque, Velocity,
    },
    force::{kilonewton, newton},
    frequency::{hertz, kilohertz},
    length::{centimeter, foot, kilometer, meter, mile, millimeter, nautical_mile},
    mass::{gram, kilogram, pound, ton},
    ratio::{percent, ratio},
    time::{day, hour, microsecond, millisecond, minute, second, year},
    torque::{newton_meter, pound_force_foot},
    velocity::{kilometer_per_hour, kilometer_per_second, meter_per_second, mile_per_hour},
};

use crate::dims::{GravParam, MassFlowRate, SpecificAngMom, SpecificEnergy};
use crate::frame::{Frame, Planet};
use crate::qty3::Qty3;

/// Unit-construction extension on `f64`.
///
/// Every method returns a `uom` scalar quantity; dimensional analysis then
/// takes over at the call site.
///
/// # Example
///
/// ```
/// use jeod_quantities::prelude::*;
///
/// let altitude = 400.0.km();          // Length
/// let inclination = 51.6.deg();        // Angle
/// let mass = 420_000.0.kg();           // Mass
/// let dt = 60.0.s();                   // Time
/// let g = 9.81.m_per_s2();             // Acceleration
/// let mu = 3.986_004_418e14.m3_per_s2_for::<Earth>(); // GravParam<Earth>
/// ```
pub trait F64Ext: Copy {
    // --- Length ---
    /// Length in meters.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let r = 6_378_000.0.m();
    /// ```
    fn m(self) -> Length;
    /// Length in kilometers.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let altitude = 400.0.km();
    /// ```
    fn km(self) -> Length;
    /// Length in centimeters.
    fn cm(self) -> Length;
    /// Length in millimeters.
    fn mm(self) -> Length;
    /// Length in feet.
    fn ft(self) -> Length;
    /// Length in statute miles.
    fn mi(self) -> Length;
    /// Length in nautical miles.
    fn nmi(self) -> Length;

    // --- Velocity ---
    /// Velocity in meters per second.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let v_circular = 7_660.0.m_per_s();
    /// ```
    fn m_per_s(self) -> Velocity;
    /// Velocity in kilometers per second.
    fn km_per_s(self) -> Velocity;
    /// Velocity in kilometers per hour.
    fn km_per_h(self) -> Velocity;
    /// Velocity in miles per hour.
    fn mph(self) -> Velocity;

    // --- Acceleration ---
    /// Acceleration in m/s².
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let g = 9.81.m_per_s2();
    /// ```
    fn m_per_s2(self) -> Acceleration;
    /// Acceleration in km/s² (scaled to m/s² internally).
    fn km_per_s2(self) -> Acceleration;
    /// Acceleration in standard gravities (1 g ≈ 9.80665 m/s²).
    fn g(self) -> Acceleration;

    // --- Mass ---
    /// Mass in kilograms.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let iss_mass = 420_000.0.kg();
    /// ```
    fn kg(self) -> Mass;
    /// Mass in grams. Suffixed `_mass` to disambiguate from `g` (acceleration).
    fn g_mass(self) -> Mass;
    /// Mass in metric tons (1000 kg).
    fn tonnes(self) -> Mass;
    /// Mass in pounds.
    fn lb(self) -> Mass;

    // --- Angle ---
    /// Angle in radians.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let half_pi = std::f64::consts::FRAC_PI_2.rad();
    /// ```
    fn rad(self) -> Angle;
    /// Angle in degrees.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let inclination = 51.6.deg();
    /// ```
    fn deg(self) -> Angle;
    /// Angle in arcminutes (1/60 degree).
    fn arcmin(self) -> Angle;
    /// Angle in arcseconds (1/3600 degree).
    fn arcsec(self) -> Angle;

    // --- Time ---
    /// Time in seconds.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let dt = 60.0.s();
    /// ```
    fn s(self) -> Time;
    /// Time in milliseconds.
    fn ms(self) -> Time;
    /// Time in microseconds.
    fn us(self) -> Time;
    /// Time in minutes (60 s).
    fn min(self) -> Time;
    /// Time in hours.
    fn hours(self) -> Time;
    /// Time in days (86_400 s).
    fn days(self) -> Time;
    /// Time in weeks (7 days).
    fn weeks(self) -> Time;
    /// Time in Julian years (365.25 days, per `uom`).
    fn years(self) -> Time;

    // --- Force ---
    /// Force in newtons.
    fn n(self) -> Force;
    /// kilonewtons (N × 1000). Spelled `kn` per the public facade list.
    fn kn(self) -> Force;
    /// Torque in newton-meters.
    fn n_m(self) -> Torque;
    /// Torque in pound-force-feet.
    fn ft_lb(self) -> Torque;

    // --- Angular velocity / acceleration ---
    /// Angular velocity in radians per second.
    fn rad_per_s(self) -> AngularVelocity;
    /// Angular velocity in degrees per second.
    fn deg_per_s(self) -> AngularVelocity;
    /// Angular velocity in revolutions per minute.
    fn rpm(self) -> AngularVelocity;
    /// Angular acceleration in radians per second².
    fn rad_per_s2(self) -> AngularAcceleration;

    // --- Gravitational parameter ---
    /// Gravitational parameter μ in m³/s², tagged with the planet
    /// phantom inferred from the surrounding type (defaults to
    /// [`crate::frame::SelfPlanet`] when no context is available).
    ///
    /// Prefer [`Self::m3_per_s2_for`] in mission code where the planet
    /// identity is known at the call site — the explicit turbofish
    /// makes the planet-pinning load-bearing and lets the compiler
    /// catch a μ that targets the wrong frame.
    ///
    /// # Example
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// // Planet-pinned via type ascription:
    /// let mu_earth: GravParam<Earth> = 3.986_004_418e14.m3_per_s2();
    /// ```
    fn m3_per_s2<P: Planet>(self) -> GravParam<P>;
    /// Gravitational parameter μ in km³/s² (scaled to m³/s² internally).
    /// Same planet-tagging shape as [`Self::m3_per_s2`].
    fn km3_per_s2<P: Planet>(self) -> GravParam<P>;
    /// Gravitational parameter μ in m³/s², explicitly pinned to
    /// planet `P`. Equivalent to [`Self::m3_per_s2`] with an explicit
    /// turbofish — preferred in mission code where the planet
    /// identity is load-bearing.
    ///
    /// ```
    /// # use jeod_quantities::prelude::*;
    /// let mu_earth = 3.986_004_418e14.m3_per_s2_for::<Earth>();
    /// // mu_earth: GravParam<Earth>
    /// ```
    fn m3_per_s2_for<P: Planet>(self) -> GravParam<P>;
    /// Planet-pinned sibling of [`Self::km3_per_s2`].
    fn km3_per_s2_for<P: Planet>(self) -> GravParam<P>;

    // --- Specific angular momentum / energy ---
    /// Specific angular momentum in m²/s.
    fn m2_per_s(self) -> SpecificAngMom;
    /// Specific angular momentum in km²/s (scaled to m²/s internally).
    fn km2_per_s(self) -> SpecificAngMom;
    /// Specific energy in m²/s².
    fn m2_per_s2(self) -> SpecificEnergy;
    /// Specific energy in km²/s² (scaled to m²/s² internally).
    fn km2_per_s2(self) -> SpecificEnergy;

    // --- Mass flow rate ---
    /// Mass flow rate in kg/s.
    fn kg_per_s(self) -> MassFlowRate;

    // --- Frequency ---
    /// Frequency in hertz.
    fn hz(self) -> Frequency;
    /// Frequency in kilohertz.
    fn khz(self) -> Frequency;

    // --- Area ---
    /// Area in square meters.
    fn m2(self) -> Area;
    /// Area in square kilometers.
    fn km2(self) -> Area;

    // --- Unitless ratios ---
    /// Dimensionless ratio. Use for unit-cancelled quantities like
    /// reflectivity coefficients.
    fn unitless(self) -> Ratio;
    /// Percentage; the value is divided by 100 to produce the ratio.
    fn percent(self) -> Ratio;
}

impl F64Ext for f64 {
    #[inline]
    fn m(self) -> Length {
        Length::new::<meter>(self)
    }
    #[inline]
    fn km(self) -> Length {
        Length::new::<kilometer>(self)
    }
    #[inline]
    fn cm(self) -> Length {
        Length::new::<centimeter>(self)
    }
    #[inline]
    fn mm(self) -> Length {
        Length::new::<millimeter>(self)
    }
    #[inline]
    fn ft(self) -> Length {
        Length::new::<foot>(self)
    }
    #[inline]
    fn mi(self) -> Length {
        Length::new::<mile>(self)
    }
    #[inline]
    fn nmi(self) -> Length {
        Length::new::<nautical_mile>(self)
    }

    #[inline]
    fn m_per_s(self) -> Velocity {
        Velocity::new::<meter_per_second>(self)
    }
    #[inline]
    fn km_per_s(self) -> Velocity {
        Velocity::new::<kilometer_per_second>(self)
    }
    #[inline]
    fn km_per_h(self) -> Velocity {
        Velocity::new::<kilometer_per_hour>(self)
    }
    #[inline]
    fn mph(self) -> Velocity {
        Velocity::new::<mile_per_hour>(self)
    }

    #[inline]
    fn m_per_s2(self) -> Acceleration {
        Acceleration::new::<meter_per_second_squared>(self)
    }
    #[inline]
    fn km_per_s2(self) -> Acceleration {
        Acceleration::new::<meter_per_second_squared>(self * 1000.0)
    }
    #[inline]
    fn g(self) -> Acceleration {
        Acceleration::new::<standard_gravity>(self)
    }

    #[inline]
    fn kg(self) -> Mass {
        Mass::new::<kilogram>(self)
    }
    #[inline]
    fn g_mass(self) -> Mass {
        Mass::new::<gram>(self)
    }
    #[inline]
    fn tonnes(self) -> Mass {
        Mass::new::<ton>(self)
    }
    #[inline]
    fn lb(self) -> Mass {
        Mass::new::<pound>(self)
    }

    #[inline]
    fn rad(self) -> Angle {
        Angle::new::<radian>(self)
    }
    #[inline]
    fn deg(self) -> Angle {
        Angle::new::<degree>(self)
    }
    #[inline]
    fn arcmin(self) -> Angle {
        Angle::new::<arcminute>(self)
    }
    #[inline]
    fn arcsec(self) -> Angle {
        Angle::new::<arcsecond>(self)
    }

    #[inline]
    fn s(self) -> Time {
        Time::new::<second>(self)
    }
    #[inline]
    fn ms(self) -> Time {
        Time::new::<millisecond>(self)
    }
    #[inline]
    fn us(self) -> Time {
        Time::new::<microsecond>(self)
    }
    #[inline]
    fn min(self) -> Time {
        Time::new::<minute>(self)
    }
    #[inline]
    fn hours(self) -> Time {
        Time::new::<hour>(self)
    }
    #[inline]
    fn days(self) -> Time {
        Time::new::<day>(self)
    }
    #[inline]
    fn weeks(self) -> Time {
        Time::new::<day>(self * 7.0)
    }
    #[inline]
    fn years(self) -> Time {
        Time::new::<year>(self)
    }

    #[inline]
    fn n(self) -> Force {
        Force::new::<newton>(self)
    }
    #[inline]
    fn kn(self) -> Force {
        Force::new::<kilonewton>(self)
    }
    #[inline]
    fn n_m(self) -> Torque {
        Torque::new::<newton_meter>(self)
    }
    #[inline]
    fn ft_lb(self) -> Torque {
        Torque::new::<pound_force_foot>(self)
    }

    #[inline]
    fn rad_per_s(self) -> AngularVelocity {
        AngularVelocity::new::<radian_per_second>(self)
    }
    #[inline]
    fn deg_per_s(self) -> AngularVelocity {
        AngularVelocity::new::<degree_per_second>(self)
    }
    #[inline]
    fn rpm(self) -> AngularVelocity {
        AngularVelocity::new::<revolution_per_minute>(self)
    }
    #[inline]
    fn rad_per_s2(self) -> AngularAcceleration {
        AngularAcceleration::new::<radian_per_second_squared>(self)
    }

    #[inline]
    fn m3_per_s2<P: Planet>(self) -> GravParam<P> {
        GravParam::<P>::from_si(self)
    }
    #[inline]
    fn km3_per_s2<P: Planet>(self) -> GravParam<P> {
        // (10³ m)³ / s² = 10⁹ m³/s²
        GravParam::<P>::from_si(self * 1.0e9)
    }
    #[inline]
    fn m3_per_s2_for<P: Planet>(self) -> GravParam<P> {
        GravParam::<P>::from_si(self)
    }
    #[inline]
    fn km3_per_s2_for<P: Planet>(self) -> GravParam<P> {
        GravParam::<P>::from_si(self * 1.0e9)
    }

    #[inline]
    fn m2_per_s(self) -> SpecificAngMom {
        SpecificAngMom {
            dimension: PhantomData,
            units: PhantomData,
            value: self,
        }
    }
    #[inline]
    fn km2_per_s(self) -> SpecificAngMom {
        SpecificAngMom {
            dimension: PhantomData,
            units: PhantomData,
            value: self * 1.0e6, // (10³ m)² / s = 10⁶ m²/s
        }
    }
    #[inline]
    fn m2_per_s2(self) -> SpecificEnergy {
        SpecificEnergy {
            dimension: PhantomData,
            units: PhantomData,
            value: self,
        }
    }
    #[inline]
    fn km2_per_s2(self) -> SpecificEnergy {
        SpecificEnergy {
            dimension: PhantomData,
            units: PhantomData,
            value: self * 1.0e6,
        }
    }

    #[inline]
    fn kg_per_s(self) -> MassFlowRate {
        MassFlowRate {
            dimension: PhantomData,
            units: PhantomData,
            value: self,
        }
    }

    #[inline]
    fn hz(self) -> Frequency {
        Frequency::new::<hertz>(self)
    }
    #[inline]
    fn khz(self) -> Frequency {
        Frequency::new::<kilohertz>(self)
    }

    #[inline]
    fn m2(self) -> Area {
        Area::new::<square_meter>(self)
    }
    #[inline]
    fn km2(self) -> Area {
        Area::new::<square_kilometer>(self)
    }

    #[inline]
    fn unitless(self) -> Ratio {
        Ratio::new::<ratio>(self)
    }
    #[inline]
    fn percent(self) -> Ratio {
        Ratio::new::<percent>(self)
    }
}

/// Extension trait on `glam::DVec3` producing frame-tagged 3-vectors.
///
/// # Example
///
/// ```
/// use glam::DVec3;
/// use jeod_quantities::prelude::*;
///
/// let r: Position<RootInertial> = DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>();
/// let v: Velocity<RootInertial> = DVec3::new(0.0, 7_546.0, 0.0).m_per_s_at::<RootInertial>();
/// ```
pub trait Vec3Ext: Copy {
    /// Interpret as meters (position) in frame `F`.
    fn m_at<F: Frame>(self) -> crate::aliases::Position<F>;
    /// Interpret as kilometers (position) in frame `F`.
    fn km_at<F: Frame>(self) -> crate::aliases::Position<F>;
    /// Interpret as m/s (velocity) in frame `F`.
    fn m_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F>;
    /// Interpret as km/s (velocity) in frame `F`.
    fn km_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F>;
    /// Interpret as m/s² (acceleration) in frame `F`.
    fn m_per_s2_at<F: Frame>(self) -> crate::aliases::Acceleration<F>;
    /// Interpret as newtons (force) in frame `F`.
    fn n_at<F: Frame>(self) -> crate::aliases::Force<F>;
    /// Interpret as rad/s (angular velocity) in frame `F`.
    fn rad_per_s_at<F: Frame>(self) -> crate::aliases::AngularVelocity<F>;
}

impl Vec3Ext for DVec3 {
    #[inline]
    fn m_at<F: Frame>(self) -> crate::aliases::Position<F> {
        Qty3::from_raw_si(self)
    }
    #[inline]
    fn km_at<F: Frame>(self) -> crate::aliases::Position<F> {
        Qty3::from_raw_si(self * 1000.0)
    }
    #[inline]
    fn m_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F> {
        Qty3::from_raw_si(self)
    }
    #[inline]
    fn km_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F> {
        Qty3::from_raw_si(self * 1000.0)
    }
    #[inline]
    fn m_per_s2_at<F: Frame>(self) -> crate::aliases::Acceleration<F> {
        Qty3::from_raw_si(self)
    }
    #[inline]
    fn n_at<F: Frame>(self) -> crate::aliases::Force<F> {
        Qty3::from_raw_si(self)
    }
    #[inline]
    fn rad_per_s_at<F: Frame>(self) -> crate::aliases::AngularVelocity<F> {
        Qty3::from_raw_si(self)
    }
}

/// Extension trait on `[f64; 3]` (raw component arrays from CSVs or JEOD
/// initial conditions) producing frame-tagged 3-vectors.
///
/// # Example
///
/// ```
/// use jeod_quantities::prelude::*;
///
/// // Loaded from a JEOD CSV row, [m, m, m].
/// let row: [f64; 3] = [7_000_000.0, 0.0, 0.0];
/// let r: Position<RootInertial> = row.m_at::<RootInertial>();
/// ```
pub trait Array3Ext: Copy {
    /// Interpret `self` as a position in metres tagged with frame `F`.
    fn m_at<F: Frame>(self) -> crate::aliases::Position<F>;
    /// Interpret `self` as a position in kilometres tagged with frame `F`.
    fn km_at<F: Frame>(self) -> crate::aliases::Position<F>;
    /// Interpret `self` as a velocity in m/s tagged with frame `F`.
    fn m_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F>;
    /// Interpret `self` as a velocity in km/s tagged with frame `F`.
    fn km_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F>;
}

impl Array3Ext for [f64; 3] {
    #[inline]
    fn m_at<F: Frame>(self) -> crate::aliases::Position<F> {
        DVec3::from_array(self).m_at::<F>()
    }
    #[inline]
    fn km_at<F: Frame>(self) -> crate::aliases::Position<F> {
        DVec3::from_array(self).km_at::<F>()
    }
    #[inline]
    fn m_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F> {
        DVec3::from_array(self).m_per_s_at::<F>()
    }
    #[inline]
    fn km_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F> {
        DVec3::from_array(self).km_per_s_at::<F>()
    }
}
