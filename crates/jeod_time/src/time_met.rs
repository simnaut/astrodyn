//! MET — Mission Elapsed Time.
//!
//! Ported from JEOD `time_met.cc`.
//!
//! MET is a UDE (User-Defined Epoch) time that counts seconds since a
//! configurable mission epoch, with support for deliberate holds (pauses).
//! MET = parent_time - epoch_offset.

/// Mission Elapsed Time state.
///
/// Tracks elapsed time from a user-defined epoch with optional hold capability.
/// Matches JEOD `TimeMET` (which extends `TimeUDE`).
#[derive(Debug, Clone)]
pub struct MissionElapsedTime {
    /// Offset from parent time scale to MET zero-point, in seconds.
    /// `MET = parent_seconds - epoch_offset`.
    pub epoch_offset: f64,
    /// Current MET in seconds.
    pub seconds: f64,
    /// Whether MET is held (paused).
    pub hold: bool,
    /// Whether we were in hold state on the previous update.
    was_held: bool,
    /// Parent time value at the moment hold was activated, used to
    /// resume correctly after hold release.
    held_at_parent: f64,
}

impl MissionElapsedTime {
    /// Create a new MET starting at zero, with epoch at the given parent time.
    ///
    /// # Arguments
    /// * `parent_seconds_at_epoch` - Parent time scale value at MET=0.
    pub fn new(parent_seconds_at_epoch: f64) -> Self {
        Self {
            epoch_offset: parent_seconds_at_epoch,
            seconds: 0.0,
            hold: false,
            was_held: false,
            held_at_parent: 0.0,
        }
    }

    /// Update MET from the parent time scale.
    ///
    /// When `hold` is true, MET freezes at its current value.
    /// When hold is released, MET resumes from where it was held,
    /// adjusting the epoch offset to account for the hold duration.
    ///
    /// Matches JEOD `TimeMET::update()` + converter reset logic.
    pub fn update(&mut self, parent_seconds: f64) {
        if self.hold {
            if !self.was_held {
                // Transition into hold: record current parent time
                self.was_held = true;
                self.held_at_parent = parent_seconds;
            }
            // MET stays frozen at its current value
        } else if self.was_held {
            // Transition out of hold: adjust epoch offset to skip held duration
            let hold_duration = parent_seconds - self.held_at_parent;
            self.epoch_offset += hold_duration;
            self.was_held = false;
            self.seconds = parent_seconds - self.epoch_offset;
        } else {
            // Normal operation
            self.seconds = parent_seconds - self.epoch_offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn met_basic() {
        let mut met = MissionElapsedTime::new(1000.0);
        assert_eq!(met.seconds, 0.0);

        met.update(1100.0);
        assert!((met.seconds - 100.0).abs() < 1e-15);

        met.update(2000.0);
        assert!((met.seconds - 1000.0).abs() < 1e-15);
    }

    #[test]
    fn met_hold_and_release() {
        let mut met = MissionElapsedTime::new(0.0);
        met.update(100.0);
        assert!((met.seconds - 100.0).abs() < 1e-15);

        // Activate hold
        met.hold = true;
        met.update(100.0); // Freeze at t=100
        let frozen = met.seconds;
        met.update(200.0); // Parent advances but MET stays frozen
        assert_eq!(met.seconds, frozen);

        // Release hold
        met.hold = false;
        met.update(200.0);
        // MET should resume from where it was held (100s), skipping the 100s hold
        assert!(
            (met.seconds - 100.0).abs() < 1e-15,
            "After hold release, MET should be 100, got {}",
            met.seconds
        );

        // Continue advancing
        met.update(300.0);
        assert!(
            (met.seconds - 200.0).abs() < 1e-15,
            "After 100 more parent seconds, MET should be 200, got {}",
            met.seconds
        );
    }

    #[test]
    fn met_from_nonzero_epoch() {
        let mut met = MissionElapsedTime::new(5000.0);
        met.update(5000.0);
        assert!(met.seconds.abs() < 1e-15);

        met.update(5060.0);
        assert!((met.seconds - 60.0).abs() < 1e-15);
    }
}
