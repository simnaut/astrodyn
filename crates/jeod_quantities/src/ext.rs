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
    angular_velocity::{
        degree_per_second, radian_per_second, revolution_per_minute as rpm_per_second,
    },
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
use crate::frame::Frame;
use crate::qty3::Qty3;

/// Unit-construction extension on `f64`.
///
/// Every method returns a `uom` scalar quantity; dimensional analysis then
/// takes over at the call site.
pub trait F64Ext: Copy {
    // --- Length ---
    fn m(self) -> Length;
    fn km(self) -> Length;
    fn cm(self) -> Length;
    fn mm(self) -> Length;
    fn ft(self) -> Length;
    fn mi(self) -> Length;
    fn nmi(self) -> Length;

    // --- Velocity ---
    fn m_per_s(self) -> Velocity;
    fn km_per_s(self) -> Velocity;
    fn km_per_h(self) -> Velocity;
    fn mph(self) -> Velocity;

    // --- Acceleration ---
    fn m_per_s2(self) -> Acceleration;
    fn km_per_s2(self) -> Acceleration;
    fn g(self) -> Acceleration;

    // --- Mass ---
    fn kg(self) -> Mass;
    fn g_mass(self) -> Mass;
    fn tonnes(self) -> Mass;
    fn lb(self) -> Mass;

    // --- Angle ---
    fn rad(self) -> Angle;
    fn deg(self) -> Angle;
    fn arcmin(self) -> Angle;
    fn arcsec(self) -> Angle;

    // --- Time ---
    fn s(self) -> Time;
    fn ms(self) -> Time;
    fn us(self) -> Time;
    fn min(self) -> Time;
    fn hours(self) -> Time;
    fn days(self) -> Time;
    fn weeks(self) -> Time;
    fn years(self) -> Time;

    // --- Force ---
    fn n(self) -> Force;
    /// kilonewtons (N × 1000). Spelled `kn` per the public facade list.
    fn kn(self) -> Force;
    fn n_m(self) -> Torque;
    fn ft_lb(self) -> Torque;

    // --- Angular velocity / acceleration ---
    fn rad_per_s(self) -> AngularVelocity;
    fn deg_per_s(self) -> AngularVelocity;
    fn rpm(self) -> AngularVelocity;
    fn rad_per_s2(self) -> AngularAcceleration;

    // --- Gravitational parameter ---
    fn m3_per_s2(self) -> GravParam;
    fn km3_per_s2(self) -> GravParam;

    // --- Specific angular momentum / energy ---
    fn m2_per_s(self) -> SpecificAngMom;
    fn km2_per_s(self) -> SpecificAngMom;
    fn m2_per_s2(self) -> SpecificEnergy;
    fn km2_per_s2(self) -> SpecificEnergy;

    // --- Mass flow rate ---
    fn kg_per_s(self) -> MassFlowRate;

    // --- Frequency ---
    fn hz(self) -> Frequency;
    fn khz(self) -> Frequency;

    // --- Area ---
    fn m2(self) -> Area;
    fn km2(self) -> Area;

    // --- Unitless ratios ---
    fn unitless(self) -> Ratio;
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
        AngularVelocity::new::<rpm_per_second>(self)
    }
    #[inline]
    fn rad_per_s2(self) -> AngularAcceleration {
        AngularAcceleration::new::<radian_per_second_squared>(self)
    }

    #[inline]
    fn m3_per_s2(self) -> GravParam {
        GravParam {
            dimension: PhantomData,
            units: PhantomData,
            value: self,
        }
    }
    #[inline]
    fn km3_per_s2(self) -> GravParam {
        GravParam {
            dimension: PhantomData,
            units: PhantomData,
            value: self * 1.0e9, // (10³ m)³ / s² = 10⁹ m³/s²
        }
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
pub trait Array3Ext: Copy {
    fn m_at<F: Frame>(self) -> crate::aliases::Position<F>;
    fn km_at<F: Frame>(self) -> crate::aliases::Position<F>;
    fn m_per_s_at<F: Frame>(self) -> crate::aliases::Velocity<F>;
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
