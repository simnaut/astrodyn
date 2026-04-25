use glam::DVec3;
use jeod_quantities::aliases::{Position, Velocity};
use jeod_quantities::frame::{Frame, Inertial};

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

/// Typed sibling of [`TranslationalState`] carrying a frame phantom on
/// the position and velocity components. Defaults to the inertial frame
/// to match the existing untyped storage convention; override with an
/// explicit frame tag for non-inertial integrations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationalStateTyped<F: Frame = Inertial> {
    /// Position in frame `F`.
    pub position: Position<F>,
    /// Velocity in frame `F`.
    pub velocity: Velocity<F>,
}

impl<F: Frame> Default for TranslationalStateTyped<F> {
    #[inline]
    fn default() -> Self {
        Self {
            position: Position::<F>::zero(),
            velocity: Velocity::<F>::zero(),
        }
    }
}

impl<F: Frame> TranslationalStateTyped<F> {
    /// Drop the frame phantom and emit the untyped storage form. The
    /// numeric values (in base SI units: m, m/s) are preserved exactly.
    #[inline]
    pub fn to_untyped(&self) -> TranslationalState {
        TranslationalState {
            position: self.position.raw_si(),
            velocity: self.velocity.raw_si(),
        }
    }

    /// Wrap an untyped [`TranslationalState`] as a typed one. **The
    /// caller asserts** that the untyped state is expressed in frame
    /// `F` — there is no runtime check.
    #[inline]
    pub fn from_untyped_unchecked(s: &TranslationalState) -> Self {
        Self {
            position: Position::<F>::from_raw_si(s.position),
            velocity: Velocity::<F>::from_raw_si(s.velocity),
        }
    }

    /// JEOD initialization heuristic — both position and velocity zero.
    ///
    /// Mirrors [`TranslationalState::is_likely_uninitialized`] for
    /// downstream code holding the typed form.
    // JEOD_INV: DM.05 — partial: detects zero translational state
    // JEOD_INV: DB.11 — partial: zero-state heuristic only
    #[inline]
    pub fn is_likely_uninitialized(&self) -> bool {
        self.position.raw_si() == DVec3::ZERO && self.velocity.raw_si() == DVec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_quantities::frame::Ecef;

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

    #[test]
    fn typed_round_trips_through_untyped() {
        let untyped = TranslationalState {
            position: DVec3::new(7e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7500.0, 0.0),
        };
        let typed = TranslationalStateTyped::<Inertial>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();
        assert_eq!(back, untyped);
    }

    #[test]
    fn typed_default_is_likely_uninitialized() {
        let s = TranslationalStateTyped::<Inertial>::default();
        assert!(s.is_likely_uninitialized());
    }

    #[test]
    fn typed_inertial_and_ecef_are_distinct_types() {
        // Compile-time check that `TranslationalStateTyped<Inertial>` and
        // `TranslationalStateTyped<Ecef>` are not assignable to one another;
        // we only verify the same-frame case compiles here.
        let s_inertial = TranslationalStateTyped::<Inertial>::default();
        let s_ecef = TranslationalStateTyped::<Ecef>::default();
        assert!(s_inertial.is_likely_uninitialized());
        assert!(s_ecef.is_likely_uninitialized());
    }
}
