//! DYN — Dynamic Time scale.
//!
//! Ported from JEOD `time_dyn.cc`.
//!
//! Dynamic time is the integration clock. It advances at a configurable
//! rate relative to simulation time: `DYN = simtime * scale_factor + offset`.
//! When `scale_factor = 1.0` (default), DYN = simtime. Changing the scale
//! factor mid-simulation adjusts the offset to maintain continuity.

/// Dynamic time state.
///
/// Matches JEOD `TimeDyn`: the root of the time update tree. All other time
/// scales ultimately derive from DYN (which derives from simtime).
#[derive(Debug, Clone)]
pub struct DynamicTime {
    /// Multiplicative factor between simtime and dynamic time.
    /// 1.0 = real-time, >1.0 = fast-forward, <0 = time reversal.
    /// Matches JEOD `TimeDyn::scale_factor`.
    pub scale_factor: f64,
    /// Private copy for detecting changes.
    ref_scale: f64,
    /// Offset to maintain continuity when scale_factor changes:
    /// `seconds = simtime * ref_scale + offset`.
    offset: f64,
    /// Current dynamic time in seconds.
    pub seconds: f64,
}

impl DynamicTime {
    /// Create a new DynamicTime with default scale factor 1.0.
    pub fn new() -> Self {
        Self {
            scale_factor: 1.0,
            ref_scale: 1.0,
            offset: 0.0,
            seconds: 0.0,
        }
    }

    /// Update dynamic time from the current simulation time.
    ///
    /// Matches JEOD `TimeDyn::update()`.
    pub fn update(&mut self, simtime: f64) {
        self.seconds = simtime * self.ref_scale + self.offset;
    }

    /// Check for and apply changes to scale_factor.
    ///
    /// Returns `true` if the scale factor changed (time direction/rate change).
    /// Matches JEOD `TimeDyn::update_offset()`.
    ///
    /// # Panics
    /// Panics if `scale_factor` or `simtime` is not finite.
    pub fn update_offset(&mut self, simtime: f64) -> bool {
        assert!(
            self.scale_factor.is_finite(),
            "scale_factor must be finite, got {}",
            self.scale_factor
        );
        assert!(
            simtime.is_finite(),
            "simtime must be finite, got {}",
            simtime
        );
        if self.ref_scale != self.scale_factor {
            self.offset = self.seconds - (self.scale_factor * simtime);
            let _direction_changed = self.ref_scale * self.scale_factor < 0.0;
            self.ref_scale = self.scale_factor;
            true // any scale change returns true
        } else {
            false
        }
    }
}

impl Default for DynamicTime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyn_time_default_scale() {
        let mut dyn_time = DynamicTime::new();
        dyn_time.update(100.0);
        assert!((dyn_time.seconds - 100.0).abs() < 1e-15);
    }

    #[test]
    fn dyn_time_double_speed() {
        let mut dyn_time = DynamicTime::new();
        dyn_time.scale_factor = 2.0;
        // Must call update_offset first to apply the change
        dyn_time.update_offset(0.0);
        dyn_time.update(50.0);
        assert!(
            (dyn_time.seconds - 100.0).abs() < 1e-15,
            "At 2x speed, 50s simtime should give 100s dyn, got {}",
            dyn_time.seconds
        );
    }

    #[test]
    fn dyn_time_mid_sim_scale_change() {
        let mut dyn_time = DynamicTime::new();
        // Run 100s at 1x
        dyn_time.update(100.0);
        assert!((dyn_time.seconds - 100.0).abs() < 1e-15);

        // Switch to 2x at simtime=100
        dyn_time.scale_factor = 2.0;
        let changed = dyn_time.update_offset(100.0);
        assert!(changed);

        // Run another 50s simtime
        dyn_time.update(150.0);
        // dyn = 150*2 + offset. offset = 100 - (2*100) = -100
        // dyn = 300 - 100 = 200
        assert!(
            (dyn_time.seconds - 200.0).abs() < 1e-15,
            "Expected 200, got {}",
            dyn_time.seconds
        );
    }

    #[test]
    fn dyn_time_reversal() {
        let mut dyn_time = DynamicTime::new();
        dyn_time.update(100.0);

        dyn_time.scale_factor = -1.0;
        dyn_time.update_offset(100.0);

        // After another 100s simtime, dyn should be back to 0
        dyn_time.update(200.0);
        assert!(
            dyn_time.seconds.abs() < 1e-15,
            "Expected 0, got {}",
            dyn_time.seconds
        );
    }
}
