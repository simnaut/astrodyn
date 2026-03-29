use glam::DVec3;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TranslationalState {
    pub position: DVec3, // m, in integration frame
    pub velocity: DVec3, // m/s, in integration frame
}

impl TranslationalState {
    /// Returns true if both position and velocity are exactly zero.
    ///
    /// In JEOD, `check_for_uninitialized_states()` fatally fails if required
    /// state is not set. An exact-zero translational state is almost certainly
    /// unintentional for orbital mechanics (it would place the vehicle at the
    /// center of the integration frame with zero velocity).
    ///
    /// This is a heuristic — a vehicle intentionally at the origin with zero
    /// velocity would be a false positive.
    // JEOD_INV: DM.05 — partial: detects zero translational state (JEOD fatally checks all states)
    // JEOD_INV: DB.11 — partial: zero-state heuristic only (no initialized_states bitfield)
    pub fn is_likely_uninitialized(&self) -> bool {
        self.position == DVec3::ZERO && self.velocity == DVec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_likely_uninitialized() {
        assert!(TranslationalState::default().is_likely_uninitialized());
    }

    #[test]
    fn nonzero_position_is_initialized() {
        let s = TranslationalState {
            position: DVec3::new(6.7e6, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        assert!(!s.is_likely_uninitialized());
    }

    #[test]
    fn nonzero_velocity_is_initialized() {
        let s = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::new(0.0, 7.5e3, 0.0),
        };
        assert!(!s.is_likely_uninitialized());
    }
}
