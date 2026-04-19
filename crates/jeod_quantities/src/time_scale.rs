//! Time-scale phantom markers + typed seconds.
//!
//! JEOD's `TimeManager` stores every scale as an `f64` field and dispatches by
//! `TimeScaleId` enum. We lift the scale distinction to the type system:
//! `SecondsSince<TAI>` and `SecondsSince<TT>` are distinct types; converting
//! requires an explicit `TimeConverter<From, To>`.

use core::marker::PhantomData;

use uom::si::f64::Time;
use uom::si::time::second;

use crate::sealed::Sealed;

/// Compile-time time-scale tag. Sealed.
pub trait TimeScale: Sealed + 'static {
    /// Human-readable name (e.g. "TAI", "TT").
    const NAME: &'static str;
}

macro_rules! time_scale_marker {
    ($name:ident, $human:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl Sealed for $name {}
        impl TimeScale for $name {
            const NAME: &'static str = $human;
        }
    };
}

time_scale_marker!(TAI, "TAI", "International Atomic Time.");
time_scale_marker!(
    UTC,
    "UTC",
    "Coordinated Universal Time (leap-second stepped)."
);
time_scale_marker!(UT1, "UT1", "Universal Time 1 (Earth-rotation based).");
time_scale_marker!(TT, "TT", "Terrestrial Time (TAI + 32.184 s).");
time_scale_marker!(
    TDB,
    "TDB",
    "Barycentric Dynamical Time (relativistic correction to TT)."
);
time_scale_marker!(GPS, "GPS", "GPS Time (TAI − 19 s).");
time_scale_marker!(
    GMST,
    "GMST",
    "Greenwich Mean Sidereal Time (angle, but seconds-ized by convention)."
);

/// Seconds elapsed in the named time scale, measured from that scale's epoch.
///
/// The wrapped `Time` is `uom::si::f64::Time` so unit arithmetic works
/// naturally (e.g. `SecondsSince<TAI>::new(1.0.hours())`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SecondsSince<S: TimeScale> {
    /// Seconds in scale `S`. Stored as a `uom` `Time` so dimensionally-safe
    /// multiplications (velocity × time → position) compose out of the box.
    pub value: Time,
    _s: PhantomData<S>,
}

impl<S: TimeScale> SecondsSince<S> {
    /// Construct from a `uom` `Time`.
    #[inline]
    pub fn new(t: Time) -> Self {
        Self {
            value: t,
            _s: PhantomData,
        }
    }

    /// Construct from raw SI seconds.
    #[inline]
    pub fn from_seconds(s: f64) -> Self {
        Self {
            value: Time::new::<second>(s),
            _s: PhantomData,
        }
    }

    /// Raw SI seconds.
    #[inline]
    pub fn as_seconds(self) -> f64 {
        self.value.get::<second>()
    }
}

/// Explicit converter between two time scales.
///
/// The converters exposed by this crate model the *constant* offsets only
/// (TAI↔TT is exactly +32.184 s; GPS↔TAI is exactly −19 s). Leap-second-aware
/// (UTC) and Earth-orientation-aware (UT1) conversions live in `jeod_time`
/// and are not part of Phase 0.
#[derive(Debug, Clone, Copy)]
pub struct TimeConverter<From: TimeScale, To: TimeScale> {
    /// Offset to add to a `From`-scale reading to get the `To`-scale reading,
    /// in SI seconds.
    offset_seconds: f64,
    _f: PhantomData<From>,
    _t: PhantomData<To>,
}

impl<From: TimeScale, To: TimeScale> TimeConverter<From, To> {
    /// Offset in SI seconds: `to = from + offset`.
    #[inline]
    pub const fn offset_seconds(self) -> f64 {
        self.offset_seconds
    }

    /// Apply the converter.
    #[inline]
    pub fn apply(self, t: SecondsSince<From>) -> SecondsSince<To> {
        SecondsSince::from_seconds(t.as_seconds() + self.offset_seconds)
    }

    /// Reverse the converter.
    #[inline]
    pub fn inverse(self) -> TimeConverter<To, From> {
        TimeConverter {
            offset_seconds: -self.offset_seconds,
            _f: PhantomData,
            _t: PhantomData,
        }
    }
}

// --- Concrete converters for constant offsets --------------------------------

/// TT − TAI offset in SI seconds (exactly +32.184 s per IAU 1991 resolution).
pub const TAI_TO_TT_OFFSET_S: f64 = 32.184;

/// GPS − TAI offset in SI seconds (exactly −19 s, fixed since 1980-01-06).
pub const GPS_TO_TAI_OFFSET_S: f64 = 19.0;

impl TimeConverter<TAI, TT> {
    /// TAI → TT: add 32.184 s.
    pub const TAI_TO_TT: Self = Self {
        offset_seconds: TAI_TO_TT_OFFSET_S,
        _f: PhantomData,
        _t: PhantomData,
    };
}

impl TimeConverter<TT, TAI> {
    /// TT → TAI: subtract 32.184 s.
    pub const TT_TO_TAI: Self = Self {
        offset_seconds: -TAI_TO_TT_OFFSET_S,
        _f: PhantomData,
        _t: PhantomData,
    };
}

impl TimeConverter<GPS, TAI> {
    /// GPS → TAI: add 19 s.
    pub const GPS_TO_TAI: Self = Self {
        offset_seconds: GPS_TO_TAI_OFFSET_S,
        _f: PhantomData,
        _t: PhantomData,
    };
}

impl TimeConverter<TAI, GPS> {
    /// TAI → GPS: subtract 19 s.
    pub const TAI_TO_GPS: Self = Self {
        offset_seconds: -GPS_TO_TAI_OFFSET_S,
        _f: PhantomData,
        _t: PhantomData,
    };
}
