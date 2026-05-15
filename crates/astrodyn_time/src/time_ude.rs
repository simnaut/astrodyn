//! UDE — User-Defined Epoch time scale.
//!
//! Ported from JEOD `time_ude.cc`.
//!
//! A UDE counts seconds from an arbitrary user-defined epoch within
//! a configurable parent time scale. `UDE = parent_time - epoch_in_parent`.

use crate::epoch::SECONDS_PER_DAY;

/// User-Defined Epoch time state.
///
/// Matches the core functionality of JEOD `TimeUDE`: a time scale
/// that counts elapsed seconds from an epoch defined in some parent
/// time scale.
#[derive(Debug, Clone)]
pub struct UserDefinedEpoch {
    /// The epoch expressed in parent-time seconds-since-epoch.
    /// `UDE = parent_seconds - epoch_in_parent`.
    pub epoch_in_parent: f64,
    /// Current UDE value in seconds.
    pub seconds: f64,
    /// Current UDE value in days.
    pub days: f64,
    /// Clock decomposition: whole days.
    pub clock_day: i32,
    /// Clock decomposition: hours (0-23).
    pub clock_hour: i32,
    /// Clock decomposition: minutes (0-59).
    pub clock_minute: i32,
    /// Clock decomposition: seconds (0-60).
    pub clock_second: f64,
}

impl UserDefinedEpoch {
    // JEOD_INV: TM.25 — UDE is parameterised by a single parent-scale epoch value.
    // JEOD allows a cascade (UDE-updates-from-UDE-with-epoch-in-UDE) and spends many
    // invariants guarding that; we forbid it by API shape — one parent scale per UDE.
    /// Create a new UDE with epoch at the given parent time value.
    pub fn new(epoch_in_parent: f64) -> Self {
        Self {
            epoch_in_parent,
            seconds: 0.0,
            days: 0.0,
            clock_day: 0,
            clock_hour: 0,
            clock_minute: 0,
            clock_second: 0.0,
        }
    }

    /// Update UDE from the parent time scale's current value.
    pub fn update(&mut self, parent_seconds: f64) {
        self.seconds = parent_seconds - self.epoch_in_parent;
        self.days = self.seconds / SECONDS_PER_DAY;
        self.clock_update();
    }

    /// Convert seconds to clock representation.
    ///
    /// Ported from JEOD `TimeUDE::clock_update()`.
    //
    // After `rem_euclid(SECONDS_PER_DAY)` / `div_euclid(3600.0)` /
    // `div_euclid(60.0)` the operands are < 24 / < 60 / < 60
    // respectively, all comfortably within `i32`. `clock_day` is bounded
    // by the simulation's epoch span (well within `i32::MAX` days ~
    // 5.8 million years).
    #[allow(
        clippy::cast_possible_truncation,
        reason = "clock fields bounded by div_euclid moduli (24h/60m/60s)"
    )]
    fn clock_update(&mut self) {
        let mut scratch = self.seconds.rem_euclid(SECONDS_PER_DAY);
        self.clock_day = self.seconds.div_euclid(SECONDS_PER_DAY) as i32;
        self.clock_hour = scratch.div_euclid(3600.0) as i32;
        scratch = scratch.rem_euclid(3600.0);
        self.clock_minute = scratch.div_euclid(60.0) as i32;
        self.clock_second = scratch.rem_euclid(60.0);

        // JEOD_INV: TM.38 — clock decomposition must carry correctly near 60s/60min/24h
        // boundaries; JEOD's default clock_resolution = 1e-6 rounds up sub-microsecond residuals.
        let clock_resolution = 1e-6;
        if self.clock_second > 60.0 - clock_resolution {
            self.clock_second = 0.0;
            self.clock_minute += 1;
            if self.clock_minute == 60 {
                self.clock_minute = 0;
                self.clock_hour += 1;
                if self.clock_hour == 24 {
                    self.clock_hour = 0;
                    self.clock_day += 1;
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "UDE reset and clock-decomposition tests assert bit-exact zero / literal values"
)]
mod tests {
    use super::*;

    #[test]
    fn ude_basic() {
        let mut ude = UserDefinedEpoch::new(1000.0);
        assert_eq!(ude.seconds, 0.0);

        ude.update(1100.0);
        assert!((ude.seconds - 100.0).abs() < 1e-15);
        assert!((ude.days - 100.0 / 86400.0).abs() < 1e-15);
    }

    #[test]
    fn ude_clock_decomposition() {
        let mut ude = UserDefinedEpoch::new(0.0);
        // 1 day, 2 hours, 3 minutes, 4.5 seconds
        let secs = 86400.0 + 7200.0 + 180.0 + 4.5;
        ude.update(secs);

        assert_eq!(ude.clock_day, 1);
        assert_eq!(ude.clock_hour, 2);
        assert_eq!(ude.clock_minute, 3);
        assert!((ude.clock_second - 4.5).abs() < 1e-10);
    }

    #[test]
    fn ude_zero_epoch() {
        let mut ude = UserDefinedEpoch::new(0.0);
        ude.update(3600.0);
        assert!((ude.seconds - 3600.0).abs() < 1e-15);
        assert_eq!(ude.clock_hour, 1);
        assert_eq!(ude.clock_minute, 0);
    }

    #[test]
    fn ude_negative_time() {
        // UDE before its epoch
        let mut ude = UserDefinedEpoch::new(1000.0);
        ude.update(500.0);
        assert!((ude.seconds - (-500.0)).abs() < 1e-15);
    }
}
